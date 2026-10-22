import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from src.agent import chatbot, checkpoint
from src.tools import ingest_rag_document

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Agentic Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _text(content) -> str:
    """Gemini can return either a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _run(graph_input, thread_id: str):
    """Stream graph events as Server-Sent Events."""
    try:
        for mode, data in chatbot.stream(
            graph_input,
            config=_config(thread_id),
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, meta = data
                if (
                    isinstance(chunk, AIMessageChunk)
                    and meta.get("langgraph_node") == "chat_node"
                ):
                    text = _text(chunk.content)
                    if text:
                        yield _sse({"type": "token", "content": text})

            elif mode == "updates":
                # Human-in-the-loop pause (purchase_stock)
                if "__interrupt__" in data:
                    for item in data["__interrupt__"]:
                        yield _sse({"type": "interrupt", "value": item.value})
                    continue

                for node, update in data.items():
                    if not update:
                        continue
                    for msg in update.get("messages", []):
                        if node == "chat_node" and isinstance(msg, AIMessage):
                            for call in msg.tool_calls:
                                yield _sse(
                                    {
                                        "type": "tool_call",
                                        "id": call["id"],
                                        "name": call["name"],
                                        "args": call["args"],
                                    }
                                )
                        elif isinstance(msg, ToolMessage):
                            yield _sse(
                                {
                                    "type": "tool_result",
                                    "id": msg.tool_call_id,
                                    "name": msg.name,
                                    "content": _text(msg.content)[:4000],
                                }
                            )
    except Exception as exc:  # surface errors to the UI instead of dropping the stream
        yield _sse({"type": "error", "message": str(exc)})
    finally:
        yield _sse({"type": "done"})


def _history(thread_id: str) -> dict:
    state = chatbot.get_state(_config(thread_id))
    raw = state.values.get("messages", []) if state and state.values else []

    out, current = [], None
    for m in raw:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _text(m.content)})
            current = None
        elif isinstance(m, AIMessage):
            if current is None:
                current = {"role": "assistant", "content": "", "tools": []}
                out.append(current)
            text = _text(m.content)
            if text:
                current["content"] += ("\n\n" if current["content"] else "") + text
            for call in m.tool_calls:
                current["tools"].append(
                    {
                        "id": call["id"],
                        "name": call["name"],
                        "args": call["args"],
                        "result": None,
                    }
                )
        elif isinstance(m, ToolMessage) and current is not None:
            for tool in current["tools"]:
                if tool["id"] == m.tool_call_id:
                    tool["result"] = _text(m.content)[:4000]

    pending = None
    for task in getattr(state, "tasks", []) or []:
        for item in getattr(task, "interrupts", []) or []:
            pending = item.value

    return {"messages": out, "pending": pending}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/threads")
def create_thread():
    return {"thread_id": str(uuid.uuid4())}


@app.get("/threads")
def list_threads():
    """Conversations ordered from most to least recent, with a short title."""
    seen: list[str] = []
    for ckpt in checkpoint.list(None):
        tid = (ckpt.config.get("configurable") or {}).get("thread_id")
        if tid and tid not in seen:
            seen.append(tid)

    threads = []
    for tid in seen:
        title = "New conversation"
        try:
            msgs = chatbot.get_state(_config(tid)).values.get("messages", [])
            first = next((m for m in msgs if isinstance(m, HumanMessage)), None)
            if first:
                title = _text(first.content).strip().replace("\n", " ")
                title = title[:42] + ("…" if len(title) > 42 else "")
        except Exception:
            pass
        threads.append({"thread_id": tid, "title": title})
    return {"threads": threads}


@app.get("/threads/{thread_id}/messages")
def get_messages(thread_id: str):
    return _history(thread_id)


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")
    stream = _run({"messages": [HumanMessage(content=req.message)]}, req.thread_id)
    return StreamingResponse(stream, media_type="text/event-stream")


@app.post("/chat/resume")
def resume(req: ResumeRequest):
    """Resume a paused run (e.g. approve or decline a stock purchase)."""
    stream = _run(Command(resume="yes" if req.approved else "no"), req.thread_id)
    return StreamingResponse(stream, media_type="text/event-stream")


@app.post("/upload")
def upload(thread_id: str = Form(...), file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    folder = UPLOAD_DIR / thread_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(file.filename).name
    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ingest_rag_document(str(path), thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")
    return {"status": "indexed", "filename": path.name}
