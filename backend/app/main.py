"""
FastAPI entry point.

Endpoints:
  POST /ask             — full RAG pipeline (JSON response)
  POST /ask/stream      — same pipeline, SSE streaming events
  POST /feedback        — thumbs up/down stored in SQLite
  GET  /feedback/export — export feedback as JSON (for reranker fine-tuning)
  GET  /cache/stats     — LRU cache hit stats
  DELETE /cache         — clear the answer cache
  GET  /health          — liveness check
  GET  /collections     — Qdrant collection info
"""

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from app.agent.critic import (
    _answer_cache,
    _set_cached,
    ask,
    get_retriever,
)
from app.config import settings
from app.models import (
    AskRequest,
    AskResponse,
    CacheStats,
    FeedbackRequest,
)

DB_PATH = Path("data/feedback.db")


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT    DEFAULT (datetime('now')),
                question   TEXT    NOT NULL,
                answer     TEXT    NOT NULL,
                tax_year   INTEGER NOT NULL,
                is_helpful INTEGER NOT NULL,
                comment    TEXT    DEFAULT ''
            )
        """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()   # warm up models at startup
    _init_db()
    yield


app = FastAPI(title="TaxCounsel API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /ask — standard JSON response
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    history = [m.model_dump() for m in req.chat_history]
    return ask(req.question, tax_year=req.tax_year, chat_history=history or None)


# ---------------------------------------------------------------------------
# /ask/stream — SSE streaming
# Events sent: rewriting | retrieving | evaluating | generating | done | error
# ---------------------------------------------------------------------------

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    q: Queue = Queue()

    def callback(event: str, data: dict) -> None:
        q.put({"event": event, "data": data})

    def run_agent() -> None:
        try:
            history = [m.model_dump() for m in req.chat_history]
            result = ask(
                req.question,
                tax_year=req.tax_year,
                chat_history=history or None,
                stream_cb=callback,
            )
            q.put({"event": "done", "data": result.model_dump()})
        except Exception as exc:
            q.put({"event": "error", "data": {"message": str(exc)}})

    Thread(target=run_agent, daemon=True).start()

    def generator():
        while True:
            try:
                msg = q.get(timeout=60)
                payload = json.dumps(msg["data"])
                yield f"event: {msg['event']}\ndata: {payload}\n\n"
                if msg["event"] in ("done", "error"):
                    break
            except Empty:
                yield "event: error\ndata: {\"message\": \"timeout\"}\n\n"
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

@app.post("/feedback", status_code=201)
def submit_feedback(req: FeedbackRequest) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback (question, answer, tax_year, is_helpful, comment) "
            "VALUES (?, ?, ?, ?, ?)",
            (req.question, req.answer, req.tax_year, int(req.is_helpful), req.comment),
        )
    return {"status": "recorded"}


@app.get("/feedback/export")
def export_feedback() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT 1000"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# /cache
# ---------------------------------------------------------------------------

@app.get("/cache/stats", response_model=CacheStats)
def cache_stats() -> CacheStats:
    from app.agent.critic import CACHE_MAX
    return CacheStats(size=len(_answer_cache), max_size=CACHE_MAX)


@app.delete("/cache")
def clear_cache() -> dict:
    _answer_cache.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# /health  /collections
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/collections")
def collections() -> dict:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    cols = client.get_collections().collections
    info = client.get_collection(settings.qdrant_collection)
    return {
        "collections": [c.name for c in cols],
        "points": info.points_count,
    }
