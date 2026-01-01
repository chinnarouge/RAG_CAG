import streamlit as st
import requests
import json
import uuid
import copy
import threading
from pathlib import Path
from datetime import datetime

STORAGE_DIR = Path(__file__).parent.parent / "data"
STORAGE_DIR.mkdir(exist_ok=True)
CHAT_HISTORY_FILE = STORAGE_DIR / "chat_history.json"
API_KEYS_FILE = STORAGE_DIR / "api_keys.json"
API_BASE = "http://127.0.0.1:8000"

PROVIDERS = {
    "ollama": {"name": "Ollama (Local)", "requires_key": False, "requires_endpoint": False},
    "openai": {"name": "OpenAI", "requires_key": True, "requires_endpoint": False},
    "azure": {"name": "Azure OpenAI", "requires_key": True, "requires_endpoint": True},
    "gemini": {"name": "Google Gemini", "requires_key": True, "requires_endpoint": False},
    "claude": {"name": "Anthropic Claude", "requires_key": True, "requires_endpoint": False},
}


def save_api_keys(keys: dict):
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(keys, f)
    except Exception as e:
        print(f"Error saving API keys: {e}")


def load_api_keys():
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_chat_history(chats: dict, uploaded_files: set):
    try:
        data = {"chats": {}, "uploaded_files": list(uploaded_files)}
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
            return chats, set(data.get("uploaded_files", []))
        except Exception as e:
            print(f"Error loading chat history: {e}")
    return {}, set()


def get_available_models(provider: str = "ollama"):
    try:
        response = requests.get(f"{API_BASE}/models?provider={provider}", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", ["gpt-oss"])
    except:
        pass
    fallback = {
        "ollama": ["gpt-oss", "llama3.2", "mistral"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "azure": ["gpt-4", "gpt-4o", "gpt-35-turbo"],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "claude": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
    }
    return fallback.get(provider, ["gpt-oss"])


def get_documents():
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"count": 0, "documents": []}


def delete_document_by_filename(filename: str):
    try:
        response = requests.delete(f"{API_BASE}/documents", json={"filename": filename}, timeout=10)
        return response.status_code == 200
    except:
        return False


st.set_page_config(page_title="RAG Chat", layout="wide")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    loaded_chats, loaded_files = load_chat_history()
    st.session_state.chats = loaded_chats
    st.session_state.uploaded_files = loaded_files
    st.session_state.current_chat_id = None
    st.session_state.selected_provider = "ollama"
    st.session_state.selected_model = "gpt-oss"
    st.session_state.search_mode = "hybrid"
    st.session_state.use_reranking = True
    st.session_state.api_keys = load_api_keys()

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = "ollama"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-oss"
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "hybrid"
if "use_reranking" not in st.session_state:
    st.session_state.use_reranking = True
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}


def save_state():
    try:
        chats_copy = copy.deepcopy(dict(st.session_state.chats))
        files_copy = set(st.session_state.uploaded_files)
    except:
        return
    threading.Thread(target=save_chat_history, args=(chats_copy, files_copy), daemon=True).start()


def create_new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "name": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "created": datetime.now()
    }
    st.session_state.current_chat_id = chat_id
    save_state()
    return chat_id


def get_current_chat():
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        return st.session_state.chats[st.session_state.current_chat_id]
    if st.session_state.chats:
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        return st.session_state.chats[st.session_state.current_chat_id]
    create_new_chat()
    return st.session_state.chats[st.session_state.current_chat_id]


with st.sidebar:
    st.title("💬 RAG Chat")
    
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.divider()
    st.subheader("🤖 LLM Provider")
    
    provider_options = list(PROVIDERS.keys())
    selected_provider_idx = provider_options.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_options else 0
    
    selected_provider = st.selectbox(
        "Provider",
        options=provider_options,
        format_func=lambda x: PROVIDERS[x]["name"],
        index=selected_provider_idx,
        key="provider_selector"
    )
    st.session_state.selected_provider = selected_provider
    
    provider_config = PROVIDERS[selected_provider]
    if provider_config["requires_key"]:
        api_key = st.text_input(
            f"🔑 {PROVIDERS[selected_provider]['name']} API Key",
            type="password",
            value=st.session_state.api_keys.get(selected_provider, ""),
            key=f"api_key_{selected_provider}",
            help="Your API key is stored locally"
        )
        if api_key:
            st.session_state.api_keys[selected_provider] = api_key
            save_api_keys(st.session_state.api_keys)
    
    if provider_config.get("requires_endpoint"):
        azure_endpoint = st.text_input(
            "🌐 Azure Endpoint",
            value=st.session_state.api_keys.get("azure_endpoint", ""),
            placeholder="https://your-resource.openai.azure.com",
            key="azure_endpoint_input"
        )
        if azure_endpoint:
            st.session_state.api_keys["azure_endpoint"] = azure_endpoint
            save_api_keys(st.session_state.api_keys)
    
    available_models = get_available_models(selected_provider)
    selected_model = st.selectbox("Model", options=available_models, index=0, key="model_selector")
    st.session_state.selected_model = selected_model
    
    st.divider()
    st.subheader("🔍 Search Settings")
    st.caption("🔗 Using **Hybrid Search** (Vector + Keyword)")
    
    use_reranking = st.checkbox("Enable Reranking", value=st.session_state.use_reranking, help="Reranks results for better relevance")
    st.session_state.use_reranking = use_reranking
    
    st.divider()
    st.subheader("💬 Chats")
    
    for chat_id, chat_data in sorted(st.session_state.chats.items(), key=lambda x: x[1]["created"], reverse=True):
        preview = chat_data["messages"][0]["content"][:30] + "..." if chat_data["messages"] else "Empty chat"
        col1, col2 = st.columns([4, 1])
        with col1:
            is_current = chat_id == st.session_state.current_chat_id
            if st.button(f"{'🔵 ' if is_current else ''}{chat_data['name']}\n{preview}", key=f"chat_{chat_id}", use_container_width=True, disabled=is_current):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat_id == chat_id:
                    st.session_state.current_chat_id = None
                save_state()
                st.rerun()
    
    st.divider()
    st.subheader("📁 Documents")
    
    doc_info = get_documents()
    doc_count = doc_info.get("count", 0)
    if doc_count == 0:
        st.warning("⚠️ No documents uploaded! Upload files to start asking questions.")
    else:
        st.success(f"📚 {doc_count} chunks indexed")
    
    uploaded_files = st.file_uploader("Upload files", type=["txt", "pdf", "docx", "csv"], label_visibility="collapsed", accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
            try:
                uploaded_file.seek(0)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                response = requests.post(
                    f"{API_BASE}/upload_stream",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    stream=True
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data = json.loads(line[6:])
                                if data["status"] == "parsing":
                                    status_text.text("📄 Parsing...")
                                elif data["status"] == "chunking":
                                    status_text.text(f"✂️ {data['total']} chunks")
                                elif data["status"] == "embedding":
                                    progress_bar.progress(data["current"] / data["total"])
                                    status_text.text(f"🔢 {data['current']}/{data['total']}")
                                elif data["status"] == "complete":
                                    progress_bar.progress(1.0)
                                    status_text.empty()
                                    progress_bar.empty()
                                    st.session_state.uploaded_files.add(uploaded_file.name)
                                    save_state()
                                    st.success(f"✅ {data['chunks_added']} chunks")
                                elif data["status"] == "error":
                                    st.error(data['message'])
            except Exception as e:
                st.error(str(e))
    
    if st.session_state.uploaded_files:
        st.caption("Uploaded files:")
        for f in list(st.session_state.uploaded_files):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"✅ {f}")
            with col2:
                if st.button("❌", key=f"del_doc_{f}", help="Delete this document"):
                    if delete_document_by_filename(f):
                        st.session_state.uploaded_files.discard(f)
                        save_state()
                        st.rerun()
                    else:
                        st.error("Failed to delete")

current_chat = get_current_chat()

col1, col2, col3 = st.columns([5, 2, 1])
with col1:
    st.title(current_chat["name"])
with col2:
    st.caption(f"🤖 {PROVIDERS[st.session_state.selected_provider]['name']}: {st.session_state.selected_model}")
with col3:
    if st.button("✏️ Rename"):
        st.session_state.show_rename = True

if st.session_state.get("show_rename"):
    new_name = st.text_input("New name:", value=current_chat["name"])
    if st.button("Save"):
        current_chat["name"] = new_name
        st.session_state.show_rename = False
        st.rerun()

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("docs"):
            with st.expander(f"📄 Sources ({len(msg['docs'])} chunks)"):
                for i, doc in enumerate(msg["docs"], 1):
                    st.caption(f"**Chunk {i}:**")
                    st.text(doc[:300] + "..." if len(doc) > 300 else doc)

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

prompt = st.chat_input("Ask a question about your documents...", disabled=st.session_state.is_processing)

if prompt and not st.session_state.is_processing:
    st.session_state.pending_query = prompt
    st.session_state.is_processing = True
    st.rerun()

if st.session_state.is_processing and st.session_state.pending_query:
    prompt = st.session_state.pending_query
    
    if not st.session_state.uploaded_files:
        st.warning("⚠️ Please upload a document first!")
        st.session_state.is_processing = False
        st.session_state.pending_query = None
    else:
        current_chat["messages"].append({"role": "user", "content": prompt})
        if len(current_chat["messages"]) == 1:
            current_chat["name"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
        save_state()
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0, text="🔍 Checking CAG cache...")
                
                conversation_history = [{"role": m["role"], "content": m["content"]} for m in current_chat["messages"][:-1]]
                current_provider = st.session_state.selected_provider
                api_key = st.session_state.api_keys.get(current_provider) if current_provider != "ollama" else None
                azure_endpoint = st.session_state.api_keys.get("azure_endpoint") if current_provider == "azure" else None
                
                if PROVIDERS[current_provider]["requires_key"] and not api_key:
                    st.error(f"⚠️ Please enter your {PROVIDERS[current_provider]['name']} API key in the sidebar")
                    st.session_state.is_processing = False
                    st.session_state.pending_query = None
                    st.stop()
                
                progress_bar.progress(10, text="⚡ Checking CAG cache...")
                
                cache_hit = False
                if len(conversation_history) == 0:
                    try:
                        cache_check = requests.post(
                            f"{API_BASE}/query",
                            json={
                                "query": prompt,
                                "model": st.session_state.selected_model,
                                "provider": current_provider,
                                "search_mode": "hybrid",
                                "use_reranking": st.session_state.use_reranking,
                                "top_k": 3
                            },
                            timeout=5
                        )
                        if cache_check.status_code == 200:
                            result = cache_check.json()
                            if result.get("cached"):
                                cache_hit = True
                                progress_bar.progress(100, text="⚡ CAG CACHE HIT - Instant response!")
                                import time
                                time.sleep(0.3)
                                progress_container.empty()
                                
                                st.markdown(f"⚡ *From CAG cache*\n\n{result['answer']}")
                                current_chat["messages"].append({
                                    "role": "assistant",
                                    "content": result["answer"],
                                    "docs": result["retrieved_docs"],
                                    "cached": True
                                })
                                save_state()
                                
                                if result["retrieved_docs"]:
                                    with st.expander(f"📄 Sources ({len(result['retrieved_docs'])} chunks)"):
                                        for i, doc in enumerate(result["retrieved_docs"], 1):
                                            st.caption(f"**Chunk {i}:**")
                                            st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                                
                                st.session_state.is_processing = False
                                st.session_state.pending_query = None
                    except:
                        pass
                
                if not cache_hit:
                    progress_bar.progress(20, text="📡 Connecting to server...")
                    
                    response = requests.post(
                        f"{API_BASE}/query_stream",
                        json={
                            "query": prompt,
                            "model": st.session_state.selected_model,
                            "provider": current_provider,
                            "api_key": api_key,
                            "azure_endpoint": azure_endpoint,
                            "search_mode": "hybrid",
                            "use_reranking": st.session_state.use_reranking,
                            "conversation_history": conversation_history,
                            "top_k": 3
                        },
                        stream=True,
                        timeout=300
                    )
                    
                    progress_bar.progress(30, text="🔍 Searching documents...")
                    
                    if response.status_code == 200:
                        answer_placeholder = st.empty()
                        answer_text = ""
                        retrieved_docs = []
                        docs_received = False
                        
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    data = json.loads(line[6:])
                                    
                                    if data["type"] == "docs":
                                        retrieved_docs = data["docs"]
                                        docs_received = True
                                        progress_bar.progress(50, text=f"📚 Found {len(retrieved_docs)} relevant chunks")
                                    elif data["type"] == "start":
                                        progress_bar.progress(60, text="🤖 Generating response...")
                                    elif data["type"] == "token":
                                        if answer_text == "":
                                            progress_bar.progress(70, text="🤖 Generating response...")
                                        answer_text += data["token"]
                                        answer_placeholder.markdown(answer_text + "▌")
                                    elif data["type"] == "done":
                                        progress_bar.progress(100, text="✅ Complete")
                                        progress_container.empty()
                                        answer_placeholder.markdown(answer_text)
                                        
                                        current_chat["messages"].append({
                                            "role": "assistant",
                                            "content": answer_text,
                                            "docs": retrieved_docs
                                        })
                                        save_state()
                                        
                                        st.session_state.is_processing = False
                                        st.session_state.pending_query = None
                                        st.rerun()
                    else:
                        st.session_state.last_error = f"Error {response.status_code}"
                        st.session_state.is_processing = False
                        st.session_state.pending_query = None
                        st.rerun()
                    
            except requests.exceptions.Timeout:
                st.session_state.last_error = "⏱️ Request timed out"
                st.session_state.is_processing = False
                st.session_state.pending_query = None
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.session_state.last_error = "❌ Cannot connect to API server"
                st.session_state.is_processing = False
                st.session_state.pending_query = None
                st.rerun()
            except Exception as e:
                st.session_state.last_error = f"Error: {str(e)}"
                st.session_state.is_processing = False
                st.session_state.pending_query = None
                st.rerun()

if "last_error" in st.session_state and st.session_state.last_error:
    st.error(st.session_state.last_error)
    st.session_state.last_error = None

if not current_chat["messages"]:
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #888;">
        <h3>👋 Welcome to RAG Chat!</h3>
        <p>Upload documents in the sidebar, then ask questions about them.</p>
        <p style="font-size: 0.9em; margin-top: 20px;">
            <b>Features:</b><br>
            🌐 Multi-Provider - OpenAI, Azure, Gemini, Claude, Ollama<br>
            🔑 Bring Your Own Key - Use your API keys<br>
            🔍 Hybrid Search - Vector + Keyword + Reranking<br>
            💬 Conversation Memory - Context-aware responses<br>
            📁 Document Management - Upload & delete files
        </p>
    </div>
    """, unsafe_allow_html=True)
