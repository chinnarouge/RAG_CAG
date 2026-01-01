import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.cache import get_cached_embedding, set_cached_embedding

EMBED_MODEL = "nomic-embed-text"
_executor = ThreadPoolExecutor(max_workers=4)


def get_embeddings(text: str) -> list[float]:
    cached = get_cached_embedding(text)
    if cached is not None:
        return cached
    
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    embedding = response["embedding"]
    set_cached_embedding(text, embedding)
    return embedding


def _generate_single_embedding(text: str) -> tuple[str, list[float]]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return text, response["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []
    
    for i, text in enumerate(texts):
        cached = get_cached_embedding(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)
    
    if uncached_texts:
        futures = {_executor.submit(_generate_single_embedding, text): i 
                   for i, text in zip(uncached_indices, uncached_texts)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                text, embedding = future.result()
                results[idx] = embedding
                set_cached_embedding(text, embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                results[idx] = [0.0] * 768
    
    return results
