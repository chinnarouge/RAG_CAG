from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from app.utils import read_file
from app.chunker import chunk_text
from app.retriever import (
    add_documents_batch, retrieve_docs, 
    get_document_count, get_all_documents, 
    delete_document, delete_documents_by_filename, clear_all,
    reload_from_disk
)
from app.generator import generate_answer, get_available_models, get_providers
from app.cache import get_cached_response, set_cached_response, clear_response_cache
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="RAG Chat API")
_thread_pool = ThreadPoolExecutor(max_workers=4)


class QueryRequest(BaseModel):
    query: str
    search_mode: Optional[str] = "hybrid"
    model: Optional[str] = None
    provider: Optional[str] = "ollama"
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    conversation_history: Optional[List[dict]] = None
    top_k: Optional[int] = 3
    use_reranking: Optional[bool] = True


class DeleteRequest(BaseModel):
    doc_index: Optional[int] = None
    filename: Optional[str] = None


@app.get("/providers")
def list_providers():
    return {"providers": get_providers()}


@app.get("/models")
def list_models(provider: str = "ollama"):
    return {"models": get_available_models(provider)}


@app.get("/documents")
def list_documents():
    return {"count": get_document_count(), "documents": get_all_documents()}


@app.delete("/documents")
def delete_documents(request: DeleteRequest):
    clear_response_cache()
    
    if request.doc_index is not None:
        success = delete_document(request.doc_index)
        if success:
            return {"status": "deleted", "doc_index": request.doc_index}
        raise HTTPException(status_code=404, detail="Document not found")
    
    if request.filename:
        count = delete_documents_by_filename(request.filename)
        return {"status": "deleted", "filename": request.filename, "chunks_deleted": count}
    
    raise HTTPException(status_code=400, detail="Provide doc_index or filename")


@app.delete("/documents/all")
def clear_documents():
    clear_response_cache()
    clear_all()
    return {"status": "cleared"}


@app.post("/refresh")
def refresh_index():
    clear_response_cache()
    reload_from_disk()
    return {"status": "refreshed", "document_count": get_document_count()}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = read_file(file.file, filename=file.filename)
    chunks = chunk_text(content)
    metadata_list = [{"filename": file.filename, "chunk_id": idx} for idx in range(len(chunks))]
    add_documents_batch(chunks, metadata_list)
    return {"status": "success", "chunks_added": len(chunks)}


@app.post("/upload_stream")
async def upload_file_stream(file: UploadFile = File(...)):
    async def generate_progress():
        try:
            content = read_file(file.file, filename=file.filename)
            yield f'data: {json.dumps({"status": "parsing", "message": "File parsed"})}\n\n'
            
            chunks = chunk_text(content)
            total = len(chunks)
            yield f'data: {json.dumps({"status": "chunking", "message": f"Created {total} chunks", "total": total})}\n\n'
            
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                metadata_list = [{"filename": file.filename, "chunk_id": i+j} for j in range(len(batch))]
                add_documents_batch(batch, metadata_list)
                yield f'data: {json.dumps({"status": "embedding", "current": min(i + batch_size, total), "total": total})}\n\n'
            
            yield f'data: {json.dumps({"status": "complete", "chunks_added": total})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"status": "error", "message": str(e)})}\n\n'
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")


@app.post("/query")
async def rag_query(request: QueryRequest):
    if not request.conversation_history:
        cached = get_cached_response(request.query, request.model, request.provider)
        if cached:
            return {"answer": cached["answer"], "retrieved_docs": cached["docs"], "cached": True}
    
    loop = asyncio.get_event_loop()
    
    docs = await loop.run_in_executor(
        _thread_pool,
        lambda: retrieve_docs(
            request.query, 
            top_k=request.top_k, 
            search_mode=request.search_mode,
            use_reranking=request.use_reranking
        )
    )
    
    answer = await loop.run_in_executor(
        _thread_pool,
        lambda: generate_answer(
            request.query, 
            "\n".join(docs),
            model=request.model,
            provider=request.provider,
            api_key=request.api_key,
            azure_endpoint=request.azure_endpoint,
            conversation_history=request.conversation_history
        )
    )
    
    if not request.conversation_history:
        set_cached_response(request.query, answer, docs, request.model, request.provider)
    
    return {"answer": answer, "retrieved_docs": docs}


@app.post("/query_stream")
async def rag_query_stream(request: QueryRequest):
    from app.generator import generate_answer_stream
    
    loop = asyncio.get_event_loop()
    
    docs = await loop.run_in_executor(
        _thread_pool,
        lambda: retrieve_docs(
            request.query, 
            top_k=request.top_k, 
            search_mode=request.search_mode,
            use_reranking=request.use_reranking
        )
    )
    context = "\n".join(docs)
    
    def generate():
        yield f'data: {json.dumps({"type": "docs", "docs": docs})}\n\n'
        yield f'data: {json.dumps({"type": "start"})}\n\n'
        
        for token in generate_answer_stream(
            request.query, 
            context,
            model=request.model,
            provider=request.provider,
            api_key=request.api_key,
            azure_endpoint=request.azure_endpoint,
            conversation_history=request.conversation_history
        ):
            yield f'data: {json.dumps({"type": "token", "token": token})}\n\n'
        
        yield f'data: {json.dumps({"type": "done"})}\n\n'
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health_check():
    return {"status": "healthy", "document_count": get_document_count(), "providers": list(get_providers().keys())}
