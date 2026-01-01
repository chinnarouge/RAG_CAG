import os
import json
import hashlib
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

STORAGE_DIR = Path(__file__).parent.parent / "data"
STORAGE_DIR.mkdir(exist_ok=True)

EMBEDDING_CACHE_FILE = STORAGE_DIR / "embedding_cache.pkl"
QUERY_CACHE_FILE = STORAGE_DIR / "query_cache.json"
DOCUMENTS_FILE = STORAGE_DIR / "documents.json"
INDEX_FILE = STORAGE_DIR / "faiss_index.bin"
CHAT_HISTORY_FILE = STORAGE_DIR / "chat_history.json"
METADATA_FILE = STORAGE_DIR / "document_metadata.json"
RESPONSE_CACHE_FILE = STORAGE_DIR / "response_cache.json"

MAX_EMBEDDING_CACHE_SIZE = 10000
MAX_QUERY_CACHE_SIZE = 500
QUERY_CACHE_TTL = 3600
RESPONSE_CACHE_TTL = 3600

_embedding_cache = OrderedDict()
_query_cache = OrderedDict()
_response_cache = OrderedDict()
_save_counter = 0


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_embedding_cache():
    global _embedding_cache
    if EMBEDDING_CACHE_FILE.exists():
        try:
            with open(EMBEDDING_CACHE_FILE, "rb") as f:
                _embedding_cache = pickle.load(f)
            print(f"Loaded {len(_embedding_cache)} cached embeddings")
        except Exception as e:
            print(f"Error loading embedding cache: {e}")
            _embedding_cache = {}


def save_embedding_cache():
    try:
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump(_embedding_cache, f)
    except Exception as e:
        print(f"Error saving embedding cache: {e}")


def get_cached_embedding(text: str):
    text_hash = _hash_text(text)
    if text_hash in _embedding_cache:
        _embedding_cache.move_to_end(text_hash)
        return _embedding_cache[text_hash]
    return None


def set_cached_embedding(text: str, embedding: list):
    global _save_counter
    text_hash = _hash_text(text)
    
    if text_hash in _embedding_cache:
        _embedding_cache.move_to_end(text_hash)
        _embedding_cache[text_hash] = embedding
        return
    
    while len(_embedding_cache) >= MAX_EMBEDDING_CACHE_SIZE:
        _embedding_cache.popitem(last=False)
    
    _embedding_cache[text_hash] = embedding
    
    _save_counter += 1
    if _save_counter >= 50:
        _save_counter = 0
        save_embedding_cache()


def get_cached_embeddings_batch(texts: list[str]):
    return [get_cached_embedding(text) for text in texts]


def set_cached_embeddings_batch(texts: list[str], embeddings: list[list]):
    for text, embedding in zip(texts, embeddings):
        text_hash = _hash_text(text)
        _embedding_cache[text_hash] = embedding
    save_embedding_cache()


def load_query_cache():
    global _query_cache
    if QUERY_CACHE_FILE.exists():
        try:
            with open(QUERY_CACHE_FILE, "r") as f:
                _query_cache = json.load(f)
            print(f"Loaded {len(_query_cache)} cached queries")
        except Exception as e:
            print(f"Error loading query cache: {e}")
            _query_cache = {}


def save_query_cache():
    try:
        with open(QUERY_CACHE_FILE, "w") as f:
            json.dump(dict(_query_cache), f)
    except Exception as e:
        print(f"Error saving query cache: {e}")


def clear_query_cache():
    global _query_cache
    _query_cache.clear()
    if QUERY_CACHE_FILE.exists():
        try:
            QUERY_CACHE_FILE.unlink()
        except:
            pass
    print("Query cache cleared")


def get_cached_query_result(query: str):
    query_hash = _hash_text(query)
    cached = _query_cache.get(query_hash)
    if cached:
        timestamp = cached.get("timestamp", 0)
        if datetime.now().timestamp() - timestamp < QUERY_CACHE_TTL:
            return cached.get("docs")
    return None


def set_cached_query_result(query: str, docs: list):
    query_hash = _hash_text(query)
    _query_cache[query_hash] = {
        "docs": docs,
        "timestamp": datetime.now().timestamp()
    }
    save_query_cache()


def _get_response_cache_key(query: str, model: str, provider: str) -> str:
    key_data = f"{query}:{model or 'default'}:{provider or 'ollama'}"
    return hashlib.md5(key_data.encode()).hexdigest()


def load_response_cache():
    global _response_cache
    if RESPONSE_CACHE_FILE.exists():
        try:
            with open(RESPONSE_CACHE_FILE, "r", encoding="utf-8") as f:
                _response_cache = OrderedDict(json.load(f))
            print(f"Loaded {len(_response_cache)} cached responses (CAG)")
        except Exception as e:
            print(f"Error loading response cache: {e}")
            _response_cache = OrderedDict()


def save_response_cache():
    try:
        with open(RESPONSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(_response_cache), f, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving response cache: {e}")


def get_cached_response(query: str, model: str = None, provider: str = None):
    cache_key = _get_response_cache_key(query, model, provider)
    cached = _response_cache.get(cache_key)
    if cached:
        timestamp = cached.get("timestamp", 0)
        if datetime.now().timestamp() - timestamp < RESPONSE_CACHE_TTL:
            _response_cache.move_to_end(cache_key)
            print(f"CAG CACHE HIT: {query[:50]}...")
            return {"answer": cached["answer"], "docs": cached["docs"], "cached": True}
    return None


def set_cached_response(query: str, answer: str, docs: list, model: str = None, provider: str = None):
    cache_key = _get_response_cache_key(query, model, provider)
    
    while len(_response_cache) >= MAX_QUERY_CACHE_SIZE:
        _response_cache.popitem(last=False)
    
    _response_cache[cache_key] = {
        "answer": answer,
        "docs": docs,
        "timestamp": datetime.now().timestamp()
    }
    save_response_cache()
    print(f"CAG CACHED: {query[:50]}...")


def clear_response_cache():
    global _response_cache
    _response_cache.clear()
    if RESPONSE_CACHE_FILE.exists():
        try:
            RESPONSE_CACHE_FILE.unlink()
        except:
            pass
    print("Response cache cleared")


def save_documents(documents: list, metadata: list = None):
    try:
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        if metadata is not None:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving documents: {e}")


def load_documents():
    if DOCUMENTS_FILE.exists():
        try:
            with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading documents: {e}")
    return []


def load_document_metadata():
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading document metadata: {e}")
    return []


def save_faiss_index(index):
    import faiss
    try:
        faiss.write_index(index, str(INDEX_FILE))
    except Exception as e:
        print(f"Error saving FAISS index: {e}")


def load_faiss_index(dimension: int):
    import faiss
    if INDEX_FILE.exists():
        try:
            index = faiss.read_index(str(INDEX_FILE))
            print(f"Loaded FAISS index with {index.ntotal} vectors")
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
    return faiss.IndexFlatL2(dimension)


def save_chat_history(chats: dict, uploaded_files: set):
    try:
        data = {
            "chats": {},
            "uploaded_files": list(uploaded_files)
        }
        for chat_id, chat_data in chats.items():
            data["chats"][chat_id] = {
                "name": chat_data["name"],
                "messages": chat_data["messages"],
                "created": chat_data["created"].isoformat() if isinstance(chat_data["created"], datetime) else chat_data["created"]
            }
        
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            chats = {}
            for chat_id, chat_data in data.get("chats", {}).items():
                chats[chat_id] = {
                    "name": chat_data["name"],
                    "messages": chat_data["messages"],
                    "created": datetime.fromisoformat(chat_data["created"]) if isinstance(chat_data["created"], str) else chat_data["created"]
                }
            
            uploaded_files = set(data.get("uploaded_files", []))
            return chats, uploaded_files
        except Exception as e:
            print(f"Error loading chat history: {e}")
    return {}, set()


load_embedding_cache()
load_query_cache()
load_response_cache()
