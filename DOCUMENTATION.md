# RAG Chat System Documentation

## Overview

A **Multi-File RAG (Retrieval-Augmented Generation) Chat System** with **CAG (Cache-Augmented Generation)**, **Multi-Provider LLM Support**, and **Hybrid Search** for optimal performance.

### 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | Vector similarity + BM25 keyword search combined |
| ⚡ **CAG Caching** | Persistent response cache - instant repeat queries |
| 🌐 **Multi-Provider** | OpenAI, Azure, Gemini, Claude, Ollama |
| 🔑 **Bring Your Own Key** | Use your API keys for cloud providers |
| 💬 **Conversation Memory** | Context-aware multi-turn conversations |
| 📁 **Multi-File Upload** | Upload multiple documents at once |
| 🎯 **Reranking** | Cross-encoder reranking for better relevance |
| 📊 **Progress Tracking** | Real-time progress bar for all operations |
| 💾 **Full Persistence** | Survives server/browser restarts |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                          │
│  (app.py - Chat UI, File Upload, Provider Selection)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP Requests
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                             │
│  (main.py - REST API with CAG Caching)                          │
├─────────────────────────────────────────────────────────────────┤
│  GET  /providers      - List LLM providers                      │
│  GET  /models         - List models for provider                │
│  GET  /documents      - List all documents                      │
│  DELETE /documents    - Delete by index/filename                │
│  DELETE /documents/all- Clear all documents                     │
│  POST /upload         - Upload documents (batch)                │
│  POST /upload_stream  - Upload with progress                    │
│  POST /query          - Query with CAG cache check              │
│  POST /query_stream   - Streaming query                         │
│  POST /refresh        - Reload data from disk                   │
│  GET  /health         - Health check                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   RETRIEVER   │  │   GENERATOR   │  │    CACHE      │
│ FAISS + BM25  │  │ Multi-Provider│  │  CAG Layer    │
│ + Reranking   │  │   + Streaming │  │  (Persistent) │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## Project Structure

```
rag_2/
├── app/
│   ├── app.py          # Streamlit frontend
│   ├── main.py         # FastAPI backend
│   ├── utils.py        # File parsing (PDF, DOCX, CSV, TXT)
│   ├── chunker.py      # Sentence-based chunking
│   ├── embeddings.py   # Ollama embeddings with caching
│   ├── retriever.py    # Hybrid search + reranking
│   ├── generator.py    # Multi-provider LLM generation
│   └── cache.py        # CAG persistent caching
├── data/               # Persistent storage (auto-created)
│   ├── documents.json
│   ├── document_metadata.json
│   ├── faiss_index.bin
│   ├── embedding_cache.pkl
│   ├── query_cache.json
│   ├── response_cache.json   # Full response CAG cache
│   ├── chat_history.json
│   └── api_keys.json
├── requirements.txt
├── run_optimized.py    # Start both servers
└── DOCUMENTATION.md
```

---

## 🌐 Multi-Provider LLM Support

### Supported Providers

| Provider | Models | Requirements |
|----------|--------|--------------|
| **Ollama** (Local) | gpt-oss, llama3.2, mistral, qwen3 | Local install |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo | API key |
| **Azure OpenAI** | gpt-4, gpt-4o, gpt-35-turbo | API key + Endpoint |
| **Google Gemini** | gemini-1.5-pro, gemini-1.5-flash | API key |
| **Anthropic Claude** | claude-3-5-sonnet, claude-3-5-haiku | API key |

### Getting API Keys

1. **OpenAI**: https://platform.openai.com/api-keys
2. **Azure**: Azure Portal → OpenAI resource → Keys and Endpoint
3. **Gemini**: https://makersuite.google.com/app/apikey
4. **Claude**: https://console.anthropic.com/

### Usage
1. Select provider from sidebar dropdown
2. Enter API key in the secure input field
3. Select model for that provider
4. Start chatting!

---

## ⚡ CAG (Cache-Augmented Generation)

### What is CAG?

CAG caches **complete responses** (not just embeddings) to disk, enabling instant responses for repeated queries - even across server restarts!

### Cache Layers

| Layer | File | TTL | Purpose |
|-------|------|-----|---------|
| **Embedding Cache** | `embedding_cache.pkl` | Permanent | Skip re-embedding same text |
| **Query Cache** | `query_cache.json` | 1 hour | Skip re-retrieval for same query |
| **Response Cache** | `response_cache.json` | 1 hour | Skip LLM call entirely! |

### How It Works

```
User asks question
    ↓
Check Response Cache → HIT? → Return instant answer ⚡
    ↓ MISS
Retrieve documents (check Query Cache)
    ↓
Generate answer with LLM
    ↓
Cache full response to disk
    ↓
Return answer
```

### Cache Invalidation

Caches are automatically cleared when:
- Documents are uploaded
- Documents are deleted
- User clicks "Clear All Data"

---

## 🔍 Hybrid Search

The system uses **Hybrid Search** combining:

### Vector Search (60% weight)
- FAISS IndexFlatL2 for semantic similarity
- 768-dimensional embeddings (nomic-embed-text)
- Finds conceptually similar content

### Keyword Search (40% weight)  
- BM25 algorithm for exact term matching
- Handles specific names, codes, acronyms
- Complements semantic search

### Reranking (Optional)
When enabled, results are re-scored using:
1. **Cross-Encoder** (ms-marco-MiniLM-L-6-v2) - More accurate
2. **Local Scoring** (fallback) - Faster, based on term overlap

---

## 📊 Progress Tracking

Real-time progress bar shows all stages:

| Stage | Progress | Status |
|-------|----------|--------|
| 🔍 | 0-10% | Checking CAG cache |
| ⚡ | 100% | CAG CACHE HIT (instant!) |
| 📡 | 20% | Connecting to server |
| 🔍 | 30% | Searching documents |
| 📚 | 50% | Found X relevant chunks |
| 🤖 | 60-70% | Generating response |
| ✅ | 100% | Complete |

---

## 💬 Conversation Memory

The system maintains conversation context:

- Last 6 messages included in LLM prompt
- Enables follow-up questions like "Tell me more" or "What about X?"
- Each chat session has independent history
- History persists across sessions

---

## 📁 Document Management

### Supported Formats
- `.txt` - Plain text
- `.pdf` - PDF documents
- `.docx` - Word documents  
- `.csv` - CSV files

### Multi-File Upload
- Upload multiple files at once
- Progress bar per file
- Automatic chunking and embedding

### Chunking Strategy
- Sentence-based splitting (NLTK)
- Max 2000 characters per chunk
- 2 sentence overlap for context continuity

---

## 🚀 Deployment to Streamlit Cloud

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (https://streamlit.io/cloud)
3. Repository with your code

### Setup Steps

#### 1. Create `requirements.txt`
```
streamlit
fastapi
uvicorn
python-multipart
requests
PyPDF2
python-docx
pandas
nltk
faiss-cpu
numpy
ollama
sentence-transformers
rank-bm25
```

#### 2. Create `.streamlit/secrets.toml` (for API keys)
```toml
[api_keys]
openai = "sk-..."
azure_key = "..."
azure_endpoint = "https://..."
gemini = "..."
claude = "..."
```

#### 3. Create `packages.txt` (for system dependencies)
```
build-essential
```

#### 4. Modify for Cloud Deployment

**Important**: Streamlit Cloud runs only the Streamlit app. You need to either:

**Option A: Embed FastAPI in Streamlit (Recommended)**
- Run FastAPI in a background thread
- Use `threading` module

**Option B: Use External API**
- Deploy FastAPI separately (Railway, Render, etc.)
- Update `API_BASE` URL in app.py

### Cloud Limitations
- No local Ollama (use cloud providers instead)
- File storage is temporary (use cloud storage for persistence)
- Memory limits apply

---

## Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Start Ollama (for local LLM)
```bash
ollama pull nomic-embed-text
ollama pull gpt-oss
ollama serve
```

### 3. Start the Application
```bash
# Option 1: Use the launcher script
python run_optimized.py

# Option 2: Start manually
# Terminal 1:
uvicorn app.main:app --reload --port 8000

# Terminal 2:
streamlit run app/app.py
```

### 4. Access
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

---

## API Reference

### Providers & Models
```http
GET /providers
Response: {"providers": {"ollama": {...}, "openai": {...}, ...}}

GET /models?provider=openai
Response: {"models": ["gpt-4o", "gpt-4o-mini", ...]}
```

### Documents
```http
GET /documents
Response: {"count": 10, "documents": [{...}, ...]}

DELETE /documents
Body: {"filename": "doc.pdf"}
Response: {"status": "deleted", "chunks_deleted": 5}

DELETE /documents/all
Response: {"status": "cleared"}

POST /refresh
Response: {"status": "refreshed", "document_count": 10}
```

### Upload
```http
POST /upload
Body: multipart/form-data (file)
Response: {"status": "success", "chunks_added": 15}

POST /upload_stream
Body: multipart/form-data (file)
Response: Server-Sent Events
```

### Query
```http
POST /query
Body: {
  "query": "What is...",
  "model": "gpt-4o",
  "provider": "openai",
  "api_key": "sk-...",
  "search_mode": "hybrid",
  "use_reranking": true,
  "top_k": 3
}
Response: {"answer": "...", "retrieved_docs": [...], "cached": true/false}

POST /query_stream
Response: Server-Sent Events with tokens
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "I don't know" responses | Check documents are uploaded, try refreshing |
| Slow first query | Normal - embeddings being generated |
| Instant second query | CAG cache working! ✓ |
| Connection error | Ensure FastAPI is running on port 8000 |
| Provider error | Check API key is valid |
| Memory error | Reduce chunk size or use smaller model |

---

## Performance Tips

1. **Use CAG**: Repeat queries are instant
2. **Enable Reranking**: Better relevance at slight speed cost
3. **Optimal Chunk Size**: 2000 chars balances context vs precision
4. **Hybrid Search**: Always enabled for best results
5. **Cloud Providers**: Faster than local Ollama for generation

---

## Security Notes

- API keys stored locally in `data/api_keys.json`
- Keys passed per-request, not stored on server
- For production: use environment variables or secrets manager
- Clear browser data to remove stored keys

---

## Future Improvements

- [ ] Multi-user authentication
- [ ] Cloud storage integration (S3, GCS)
- [ ] Webhook notifications
- [ ] Advanced analytics dashboard
- [ ] Custom embedding models
- [ ] PDF page citations

---

*Documentation updated: January 1, 2026*
