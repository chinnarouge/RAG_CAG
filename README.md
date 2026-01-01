# RAG Chat

A document Q&A system with multi-provider LLM support and intelligent caching.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

## What it does

Upload your documents (PDF, Word, TXT, CSV), ask questions, get answers. The system retrieves relevant chunks from your documents and generates responses using your choice of LLM.

**Supported providers:** Ollama (local), OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/rag-chat.git
cd rag-chat
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab')"

# Start Ollama (for local LLM)
ollama pull nomic-embed-text
ollama pull gpt-oss

# Run the app
python run_optimized.py
```

Open http://localhost:8501 in your browser.

## Features

- **Hybrid Search** - Combines vector similarity (FAISS) with keyword matching (BM25)
- **Response Caching** - Repeated questions return instantly from cache
- **Multi-file Upload** - Process multiple documents at once
- **Conversation Memory** - Follow-up questions understand context
- **Reranking** - Optional cross-encoder reranking for better relevance
- **Streaming** - See responses as they're generated

## Project Structure

```
rag_chat/
├── app/
│   ├── app.py          # Streamlit UI
│   ├── main.py         # FastAPI backend
│   ├── cache.py        # Caching layer
│   ├── retriever.py    # Search (FAISS + BM25)
│   ├── generator.py    # LLM providers
│   ├── embeddings.py   # Vector embeddings
│   ├── chunker.py      # Text splitting
│   └── utils.py        # File parsing
├── data/               # Persisted data (auto-created)
├── requirements.txt
└── run_optimized.py
```

## Using Cloud Providers

1. Select provider from sidebar dropdown
2. Enter your API key
3. Choose a model
4. Start chatting

Get API keys from:
- OpenAI: https://platform.openai.com/api-keys
- Azure: Azure Portal → OpenAI resource
- Gemini: https://makersuite.google.com/app/apikey
- Claude: https://console.anthropic.com/

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload document |
| `/query` | POST | Query with caching |
| `/query_stream` | POST | Streaming query |
| `/documents` | GET | List documents |
| `/documents` | DELETE | Delete by filename |
| `/health` | GET | Health check |

## Configuration

Key settings in the code:

| Setting | Location | Default |
|---------|----------|---------|
| Chunk size | `chunker.py` | 2000 chars |
| Embedding model | `embeddings.py` | nomic-embed-text |
| Cache TTL | `cache.py` | 1 hour |
| Vector weight | `retriever.py` | 60% |
| BM25 weight | `retriever.py` | 40% |

## Requirements

- Python 3.10+
- Ollama (for local LLM/embeddings)
- ~2GB RAM for embeddings

## License

MIT
