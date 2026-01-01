import faiss
import numpy as np
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from app.embeddings import get_embeddings, get_embeddings_batch
from app.cache import (
    load_faiss_index, save_faiss_index, 
    load_documents, save_documents, load_document_metadata,
    get_cached_query_result, set_cached_query_result,
    save_embedding_cache, clear_query_cache
)

DIMENSION = 768
_executor = ThreadPoolExecutor(max_workers=4)

index = load_faiss_index(DIMENSION)
documents = load_documents()
document_metadata = load_document_metadata()

if index.ntotal != len(documents):
    print(f"WARNING: Index/document mismatch! Rebuilding...")
    if documents:
        from app.embeddings import get_embeddings_batch
        embeddings = get_embeddings_batch(documents)
        index = faiss.IndexFlatL2(DIMENSION)
        index.add(np.array(embeddings).astype("float32"))
        save_faiss_index(index)
        print(f"Index rebuilt with {index.ntotal} vectors")
    else:
        index = faiss.IndexFlatL2(DIMENSION)

print(f"Retriever initialized: {len(documents)} documents, {index.ntotal} vectors")


def reload_from_disk():
    global index, documents, document_metadata, bm25
    
    index = load_faiss_index(DIMENSION)
    documents = load_documents()
    document_metadata = load_document_metadata()
    
    if documents:
        bm25.fit(documents)
    else:
        bm25.fit([])
    
    clear_query_cache()
    print(f"Reloaded: {len(documents)} documents, {index.ntotal} vectors")
    return len(documents)


class Reranker:
    def __init__(self):
        self.cross_encoder = None
        self._cross_encoder_loaded = False
    
    def _load_cross_encoder_lazy(self):
        if self._cross_encoder_loaded:
            return
        self._cross_encoder_loaded = True
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2', max_length=256)
            print("Cross-encoder loaded")
        except Exception as e:
            print(f"Cross-encoder not available: {e}")
    
    def rerank_local(self, query: str, docs: list[str], top_k: int = None) -> list[str]:
        if not docs:
            return docs
        
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))
        query_terms_list = list(query_terms)
        
        scores = []
        for doc in docs:
            doc_lower = doc.lower()
            doc_terms = set(re.findall(r'\w+', doc_lower))
            
            overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            
            pos_score = 0.0
            for term in query_terms_list[:5]:
                pos = doc_lower.find(term)
                if pos != -1:
                    pos_score += 1 - (pos / (len(doc_lower) + 1))
            pos_score /= (min(5, len(query_terms_list)) + 1)
            
            score = 0.7 * overlap + 0.3 * pos_score
            scores.append(score)
        
        sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _ in sorted_pairs]
        
        return reranked[:top_k] if top_k else reranked
    
    def rerank_cross_encoder(self, query: str, docs: list[str], top_k: int = None) -> list[str]:
        self._load_cross_encoder_lazy()
        
        if not docs or self.cross_encoder is None:
            return self.rerank_local(query, docs, top_k)
        
        truncated_docs = [doc[:512] for doc in docs]
        pairs = [[query, doc] for doc in truncated_docs]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _ in sorted_pairs]
        
        return reranked[:top_k] if top_k else reranked
    
    def rerank(self, query: str, docs: list[str], top_k: int = None, use_cross_encoder: bool = False) -> list[str]:
        if use_cross_encoder:
            return self.rerank_cross_encoder(query, docs, top_k)
        return self.rerank_local(query, docs, top_k)


reranker = Reranker()


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.term_freqs = []
        self.num_docs = 0
        
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
    
    def fit(self, documents: list[str]):
        self.num_docs = len(documents)
        self.term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()
        
        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def get_scores(self, query: str) -> np.ndarray:
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.num_docs)
        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
                
            df = self.doc_freqs[token]
            idf = np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
            
            for i, tf in enumerate(self.term_freqs):
                if token in tf:
                    freq = tf[token]
                    doc_len = self.doc_lengths[i]
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += idf * numerator / denominator
        
        return scores


bm25 = BM25()
if documents:
    bm25.fit(documents)


def add_document(doc_id: str, text: str, metadata: dict = None):
    embedding = get_embeddings(text)
    vector = np.array([embedding]).astype("float32")

    index.add(vector)
    documents.append(text)
    document_metadata.append(metadata or {"doc_id": doc_id})
    
    bm25.fit(documents)
    save_faiss_index(index)
    save_documents(documents, document_metadata)
    
    print(f"Added document. Total: {len(documents)} docs")


def add_documents_batch(texts: list[str], metadata_list: list[dict] = None):
    if not texts:
        return 0
    
    embeddings = get_embeddings_batch(texts)
    vectors = np.array(embeddings).astype("float32")
    
    index.add(vectors)
    documents.extend(texts)
    
    if metadata_list:
        document_metadata.extend(metadata_list)
    else:
        document_metadata.extend([{"doc_id": f"doc_{i}"} for i in range(len(texts))])
    
    bm25.fit(documents)
    clear_query_cache()
    
    save_faiss_index(index)
    save_documents(documents, document_metadata)
    save_embedding_cache()
    
    print(f"Added {len(texts)} chunks. Total: {len(documents)} docs")
    return len(texts)


def retrieve_docs(query: str, top_k: int = 3, search_mode: str = "hybrid", use_reranking: bool = True):
    print(f"Retrieval: {len(documents)} docs, {index.ntotal} vectors")
    
    if index.ntotal == 0 or len(documents) == 0:
        print("WARNING: No documents in index!")
        return []
    
    cache_key = f"{query}_{search_mode}_{top_k}_{use_reranking}"
    cached_result = get_cached_query_result(cache_key)
    if cached_result is not None:
        print(f"Query cache hit")
        return cached_result

    fetch_k = top_k * 3 if use_reranking else top_k
    results = []
    
    if search_mode == "keyword":
        scores = bm25.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:fetch_k]
        results = [documents[i] for i in top_indices if scores[i] > 0]
        
    elif search_mode == "vector":
        query_embedding = get_embeddings(query)
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = index.search(query_vector, min(fetch_k, index.ntotal))
        results = [documents[idx] for idx in indices[0] if idx != -1]
        
    else:
        query_embedding = get_embeddings(query)
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = index.search(query_vector, min(fetch_k * 2, index.ntotal))
        
        max_dist = max(distances[0]) if len(distances[0]) > 0 else 1
        vector_scores = {}
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                vector_scores[idx] = 1 - (dist / (max_dist + 1e-6))
        
        bm25_scores = bm25.get_scores(query)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        
        combined_scores = {}
        for idx in range(len(documents)):
            v_score = vector_scores.get(idx, 0)
            k_score = bm25_scores[idx] / (max_bm25 + 1e-6) if max_bm25 > 0 else 0
            combined_scores[idx] = 0.6 * v_score + 0.4 * k_score
        
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        results = [documents[i] for i in sorted_indices[:fetch_k] if combined_scores[i] > 0]
    
    if use_reranking and results:
        results = reranker.rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]
    
    set_cached_query_result(cache_key, results)
    return results


def get_document_count():
    return len(documents)


def get_all_documents():
    return [
        {"id": i, "text": doc[:200] + "..." if len(doc) > 200 else doc, "metadata": document_metadata[i] if i < len(document_metadata) else {}}
        for i, doc in enumerate(documents)
    ]


def delete_document(doc_index: int):
    global index, documents, document_metadata
    
    if doc_index < 0 or doc_index >= len(documents):
        return False
    
    documents.pop(doc_index)
    if doc_index < len(document_metadata):
        document_metadata.pop(doc_index)
    
    index = faiss.IndexFlatL2(DIMENSION)
    if documents:
        embeddings = get_embeddings_batch(documents)
        vectors = np.array(embeddings).astype("float32")
        index.add(vectors)
        bm25.fit(documents)
    
    save_faiss_index(index)
    save_documents(documents, document_metadata)
    return True


def delete_documents_by_filename(filename: str):
    global index, documents, document_metadata
    
    indices_to_delete = []
    for i, meta in enumerate(document_metadata):
        if meta.get("filename") == filename:
            indices_to_delete.append(i)
    
    if not indices_to_delete:
        print(f"No documents found for: {filename}")
        return 0
    
    print(f"Deleting {len(indices_to_delete)} chunks for {filename}")
    
    indices_to_keep = [i for i in range(len(documents)) if i not in indices_to_delete]
    
    if indices_to_keep and index.ntotal > 0:
        try:
            kept_vectors = np.array([index.reconstruct(i) for i in indices_to_keep]).astype("float32")
        except:
            kept_docs = [documents[i] for i in indices_to_keep]
            embeddings = get_embeddings_batch(kept_docs)
            kept_vectors = np.array(embeddings).astype("float32")
    else:
        kept_vectors = None
    
    new_documents = [documents[i] for i in indices_to_keep]
    new_metadata = [document_metadata[i] for i in indices_to_keep]
    
    documents.clear()
    documents.extend(new_documents)
    document_metadata.clear()
    document_metadata.extend(new_metadata)
    
    index = faiss.IndexFlatL2(DIMENSION)
    if kept_vectors is not None and len(kept_vectors) > 0:
        index.add(kept_vectors)
    
    if documents:
        bm25.fit(documents)
    else:
        bm25.fit([])
    
    clear_query_cache()
    save_faiss_index(index)
    save_documents(documents, document_metadata)
    
    print(f"After delete: {len(documents)} docs")
    return len(indices_to_delete)


def clear_all():
    global index, documents, document_metadata
    index = faiss.IndexFlatL2(DIMENSION)
    documents = []
    document_metadata = []
    bm25.fit([])
    save_faiss_index(index)
    save_documents(documents, document_metadata)
    save_embedding_cache()
