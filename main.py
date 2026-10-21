import os
import sys
import json
import uuid
import time
import sqlite3
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import psutil
import pymupdf
import chromadb
import ollama

# Base Paths & Clients
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "chat_history.db"
CHROMA_DATA_PATH = str(BASE_DIR / "chroma_db")
OLLAMA_HOST = "http://localhost:11434"
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB guard

# Initialize FastAPI App
app = FastAPI(
    title="Local AI Dashboard",
    description="Mobile-first Local AI Dashboard with streaming Ollama chat, SQLite history, RAG, and hardware telemetry."
)

# Permissive CORS for Local Wi-Fi, Tailscale (100.x.y.z), MagicDNS hostnames
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_client = ollama.Client(host=OLLAMA_HOST)
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

DOCS_METADATA_FILE = BASE_DIR / "documents_meta.json"

# -------------------------------------------------------------
# SQLite Database Setup & Helper Functions
# -------------------------------------------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.commit()

init_db()

def load_docs_metadata() -> Dict[str, Any]:
    if DOCS_METADATA_FILE.exists():
        try:
            with open(DOCS_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_docs_metadata(meta: Dict[str, Any]):
    try:
        with open(DOCS_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"Error saving docs metadata: {e}")

# Pydantic Schemas
class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = "llama3.1:8b"

class UpdateSessionRequest(BaseModel):
    title: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    model: str
    messages: List[ChatMessage]
    use_rag: bool = True

# -------------------------------------------------------------
# 1. Telemetry API (psutil + safe pynvml fallback)
# -------------------------------------------------------------
@app.get("/api/telemetry")
async def get_telemetry():
    """
    Returns hardware metrics: CPU %, RAM %, GPU utilization, VRAM, and Temp.
    Wrapped in try/except with simulated fallback for GPU.
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count(logical=True)
    
    vmem = psutil.virtual_memory()
    ram_data = {
        "total_gb": round(vmem.total / (1024 ** 3), 2),
        "used_gb": round(vmem.used / (1024 ** 3), 2),
        "percent": vmem.percent
    }
    
    gpu_data = {
        "available": False,
        "name": "NVIDIA GPU (Unavailable / Mock)",
        "utilization_percent": 15,
        "vram_total_mb": 8192,
        "vram_used_mb": 2048,
        "vram_percent": 25.0,
        "temperature_c": 48,
        "is_mock": True
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            
            total_mb = round(mem.total / (1024 ** 2), 1)
            used_mb = round(mem.used / (1024 ** 2), 1)
            vram_pct = round((mem.used / mem.total) * 100, 1) if mem.total > 0 else 0.0

            gpu_data = {
                "available": True,
                "name": name,
                "utilization_percent": util.gpu,
                "vram_total_mb": total_mb,
                "vram_used_mb": used_mb,
                "vram_percent": vram_pct,
                "temperature_c": temp,
                "is_mock": False
            }
        pynvml.nvmlShutdown()
    except Exception as e:
        gpu_data["error"] = str(e)
    
    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": cpu_count
        },
        "ram": ram_data,
        "gpu": gpu_data
    }

# -------------------------------------------------------------
# 2. Dynamic Model Switcher API
# -------------------------------------------------------------
@app.get("/api/models")
async def get_models():
    """
    Fetches model list from Ollama API with fallback to recommended defaults.
    """
    models = []
    ollama_online = False
    
    try:
        response = ollama_client.list()
        model_list = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        for m in model_list:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                models.append(name)
        ollama_online = True
    except Exception as e:
        print(f"Warning: Ollama connection: {e}")
        ollama_online = False
    
    default_models = ["llama3.1:8b", "qwen2.5-coder:7b"]
    for dm in default_models:
        if dm not in models and not any(dm in m for m in models):
            models.append(dm)
            
    return {
        "ollama_online": ollama_online,
        "models": models,
        "recommended": ["llama3.1:8b", "qwen2.5-coder:7b"]
    }

# -------------------------------------------------------------
# 3. Persistent Chat Session Management (SQLite)
# -------------------------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    """Returns all chat sessions ordered by latest updated."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.title, s.model, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """)
        rows = cursor.fetchall()
        sessions = [
            {
                "id": r["id"],
                "title": r["title"],
                "model": r["model"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"]
            }
            for r in rows
        ]
    return {"sessions": sessions}

@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    """Creates a new chat session."""
    session_id = str(uuid.uuid4())[:8]
    title = req.title.strip() if req.title else "New Chat"
    model = req.model or "llama3.1:8b"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, model, now, now)
        )
        conn.commit()
        
    return {
        "id": session_id,
        "title": title,
        "model": model,
        "created_at": now,
        "updated_at": now,
        "message_count": 0
    }

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieves session details and its full message history."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, model, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cursor.execute("""
            SELECT id, role, content, sources, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        msg_rows = cursor.fetchall()
        
        messages = []
        for mr in msg_rows:
            sources_val = None
            if mr["sources"]:
                try:
                    sources_val = json.loads(mr["sources"])
                except Exception:
                    sources_val = None
            messages.append({
                "id": mr["id"],
                "role": mr["role"],
                "content": mr["content"],
                "sources": sources_val,
                "created_at": mr["created_at"]
            })
            
    return {
        "session": {
            "id": session_row["id"],
            "title": session_row["title"],
            "model": session_row["model"],
            "created_at": session_row["created_at"],
            "updated_at": session_row["updated_at"]
        },
        "messages": messages
    }

@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, req: UpdateSessionRequest):
    """Updates session title."""
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        conn.commit()
    return {"success": True, "id": session_id, "title": title}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Deletes a chat session and all associated messages."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    return {"success": True, "message": f"Session {session_id} deleted."}

# -------------------------------------------------------------
# 4. Document Ingestion & RAG Pipeline
# -------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            space_idx = text.rfind(" ", start + chunk_size // 2, end)
            newline_idx = text.rfind("\n", start + chunk_size // 2, end)
            split_at = max(space_idx, newline_idx)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < text_len else text_len
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests .txt and .pdf files with file size and format guards.
    """
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()
    
    if ext not in [".txt", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Only .txt and .pdf files are supported."
        )
    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the 25MB limit.")
    
    raw_text = ""
    page_count = 1
    
    if ext == ".pdf":
        try:
            doc = pymupdf.open(stream=contents, filetype="pdf")
            page_count = len(doc)
            pages_text = []
            for i, page in enumerate(doc):
                pt = page.get_text()
                if pt.strip():
                    pages_text.append(f"[Page {i+1}]\n{pt.strip()}")
            raw_text = "\n\n".join(pages_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unreadable PDF document: {str(e)}")
    else:
        try:
            raw_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = contents.decode("latin-1", errors="replace")
            
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract readable text from document.")
    
    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document contains no chunkable text content.")
    
    doc_id = str(uuid.uuid4())[:8]
    chunk_ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    try:
        for idx, chunk in enumerate(chunks):
            res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk)
            emb = res.get("embedding")
            if not emb:
                raise ValueError("Ollama returned an empty embedding vector.")
            
            c_id = f"{doc_id}_{idx}"
            chunk_ids.append(c_id)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "timestamp": int(time.time())
            })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding error: {str(e)}. Ensure Ollama has 'nomic-embed-text' pulled."
        )
        
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    meta = load_docs_metadata()
    meta[doc_id] = {
        "id": doc_id,
        "filename": filename,
        "file_type": ext[1:].upper(),
        "file_size": len(contents),
        "pages": page_count,
        "chunks_count": len(chunks),
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_docs_metadata(meta)
    
    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
        "pages": page_count,
        "message": f"Successfully ingested '{filename}' ({len(chunks)} chunks)."
    }

@app.get("/api/documents")
async def list_documents():
    meta = load_docs_metadata()
    total_vectors = collection.count()
    return {
        "documents": list(meta.values()),
        "total_chunks": total_vectors
    }

@app.delete("/api/documents")
async def clear_documents():
    global collection
    chroma_client.delete_collection("documents")
    collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    save_docs_metadata({})
    return {"success": True, "message": "All documents cleared."}

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    meta = load_docs_metadata()
    if doc_id in meta:
        filename = meta[doc_id]["filename"]
        try:
            collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            print(f"Chroma delete: {e}")
        del meta[doc_id]
        save_docs_metadata(meta)
        return {"success": True, "message": f"Deleted '{filename}'."}
    raise HTTPException(status_code=404, detail="Document not found.")

# -------------------------------------------------------------
# 5. Real-Time Chat Streaming with SQLite Persistence
# -------------------------------------------------------------
@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """
    Streams response over SSE, retrieves RAG context, and persists
    conversation turns in SQLite chat_history.db.
    """
    session_id = request.session_id
    model = request.model
    messages = [m.model_dump() for m in request.messages]
    use_rag = request.use_rag
    
    # Ensure session exists or create one
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, "New Chat", model, now, now)
            )
            conn.commit()
    else:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM sessions WHERE id = ?", (session_id,))
            s_row = cursor.fetchone()
            if not s_row:
                cursor.execute(
                    "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, "New Chat", model, now, now)
                )
                conn.commit()

    # Find the latest user message
    user_prompt = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_prompt = msg["content"]
            break

    # Persist the user message to SQLite
    user_msg_id = str(uuid.uuid4())[:8]
    if user_prompt:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (id, session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_msg_id, session_id, "user", user_prompt, None, now)
            )
            # Auto-title if session title is default
            cursor.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
            t_row = cursor.fetchone()
            if t_row and t_row["title"] == "New Chat":
                new_title = user_prompt[:32].strip() + ("..." if len(user_prompt) > 32 else "")
                cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
            cursor.execute("UPDATE sessions SET updated_at = ?, model = ? WHERE id = ?", (now, model, session_id))
            conn.commit()

    sources = []
    
    # RAG Retrieval
    if use_rag and collection.count() > 0 and user_prompt:
        try:
            q_res = ollama_client.embeddings(model="nomic-embed-text", prompt=user_prompt)
            q_emb = q_res.get("embedding")
            if q_emb:
                k = min(4, collection.count())
                results = collection.query(query_embeddings=[q_emb], n_results=k)
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0] if "distances" in results else [0] * len(docs)
                
                context_blocks = []
                for doc_text, meta, dist in zip(docs, metas, dists):
                    fname = meta.get("filename", "Document")
                    c_idx = meta.get("chunk_index", 0)
                    sources.append({
                        "filename": fname,
                        "chunk_index": c_idx,
                        "distance": round(float(dist), 3) if dist is not None else 0.0,
                        "snippet": doc_text[:200] + ("..." if len(doc_text) > 200 else "")
                    })
                    context_blocks.append(f"[Document: {fname} (Section {c_idx + 1})]\n{doc_text}")
                
                if context_blocks:
                    context_str = "\n\n---\n\n".join(context_blocks)
                    system_rag_prompt = (
                        "You are an intelligent local AI assistant. Ground your answer in the following retrieved context "
                        "whenever applicable, cite document names, and maintain high factual precision.\n\n"
                        f"### RETRIEVED CONTEXT:\n{context_str}\n\n"
                        "If the answer is not contained in the context, answer using your general knowledge."
                    )
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] += f"\n\n{system_rag_prompt}"
                    else:
                        messages.insert(0, {"role": "system", "content": system_rag_prompt})
        except Exception as e:
            print(f"RAG search error: {e}")

    async def event_generator():
        # First send session metadata event
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"

        if sources:
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            
        full_assistant_reply = ""
        try:
            stream = ollama_client.chat(
                model=model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                token = ""
                if isinstance(chunk, dict):
                    token = chunk.get("message", {}).get("content", "")
                else:
                    msg = getattr(chunk, "message", None)
                    token = getattr(msg, "content", "") if msg else ""
                
                if token:
                    full_assistant_reply += token
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                    await asyncio.sleep(0.001)
                    
            # Persist assistant response into SQLite
            assistant_msg_id = str(uuid.uuid4())[:8]
            finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            sources_json = json.dumps(sources) if sources else None
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (id, session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (assistant_msg_id, session_id, "assistant", full_assistant_reply, sources_json, finished_at)
                )
                cursor.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (finished_at, session_id))
                conn.commit()

            yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'status': 'complete'})}\n\n"
        except Exception as e:
            err_msg = f"Inference error ({model}): {str(e)}"
            yield f"event: error\ndata: {json.dumps({'error': err_msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------
# 6. Serve Mobile-First Frontend
# -------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
