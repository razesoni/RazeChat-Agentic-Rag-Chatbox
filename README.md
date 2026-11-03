<div align="center">

# ✦ RazeChat — Agentic RAG Chatbox

**A tool-using AI assistant for conversational research, PDF question answering, and everyday tasks.**

Built with Streamlit, FastAPI, LangGraph, and Google Gemini.

</div>

---

## Overview

RazeChat is a full-stack conversational assistant that combines a Streamlit chat interface with a FastAPI backend. Its LangGraph-powered agent can answer general questions, call tools when useful, and keep conversations across sessions. Upload a PDF to ask questions grounded in its contents, or use the assistant's web search, calculator, weather, and stock tools.

## Features

- **Agentic chat:** Google Gemini can answer directly or choose from the available tools.
- **PDF question answering (RAG):** Upload PDFs; documents are split into chunks, embedded, and indexed in a per-conversation FAISS store.
- **Web search:** Retrieve recent information using DuckDuckGo.
- **Utility tools:** Calculate expressions, look up weather and stock prices, and simulate a stock purchase.
- **Human approval:** The simulated purchase flow pauses for approval before it completes.
- **Persistent conversations:** LangGraph checkpoints conversation state in SQLite.
- **Streaming responses:** FastAPI streams response events to the Streamlit frontend.
- **API documentation:** FastAPI's interactive docs are available at `/docs` when the backend is running.

## Architecture

```text
Streamlit UI (app/app.py)
        │ HTTP + Server-Sent Events
        ▼
FastAPI API (app/api.py)
        │
        ▼
LangGraph agent (src/agent.py) ── tools (src/tools.py)
        ├── Google Gemini
        ├── FAISS + Hugging Face embeddings (PDF RAG)
        ├── SQLite conversation checkpointing
        └── DuckDuckGo, weather, stock, and calculator tools
```

## Requirements

- Python 3.10 or later
- A Google AI API key for Gemini
- Optional API keys for weather and stock-price tools

The first run may download the `all-MiniLM-L6-v2` embedding model. The project uses a local FAISS index and SQLite database, so allow disk space for downloaded models and uploaded/indexed documents.

## Getting started

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/razesoni/RazeChat-Agentic-Rag-Chatbox.git
cd RazeChat-Agentic-Rag-Chatbox

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root. Add the Google AI API key; other keys are optional unless you use their corresponding tools.

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
STOCK_API_KEY=your_alpha_vantage_api_key
```

- `GOOGLE_API_KEY` — required for Gemini-powered chat.
- `OPENWEATHER_API_KEY` — required for current weather lookups.
- `STOCK_API_KEY` — required for Alpha Vantage stock-price lookups.

Keep `.env` private and never commit API keys.

### 3. Start the API backend

From the repository root, run:

```bash
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 4. Start the Streamlit interface

In a second terminal, activate the same virtual environment and run:

```bash
streamlit run app/app.py
```

Open the local URL printed by Streamlit. By default, the frontend connects to `http://localhost:8000`. To use a different backend URL, set `BACKEND_URL` before starting Streamlit, for example:

```bash
BACKEND_URL=http://127.0.0.1:8000 streamlit run app/app.py
```

PowerShell:

```powershell
$env:BACKEND_URL = "http://127.0.0.1:8000"
streamlit run app/app.py
```

## Using the chatbox

1. Start both the API and Streamlit processes.
2. Ask a question in the chat. The agent answers directly or invokes a relevant tool.
3. Attach a PDF in the chat input to index it for the current conversation, then ask questions about its contents.
4. Review tool activity in the assistant response. For the simulated stock-purchase tool, approve or decline the pending action in the interface.
5. Use the sidebar to start a new conversation or revisit saved chats.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check backend status. |
| `POST` | `/threads` | Create a conversation ID. |
| `GET` | `/threads` | List saved conversations. |
| `GET` | `/threads/{thread_id}/messages` | Retrieve conversation messages and pending approval, if any. |
| `POST` | `/chat` | Send a message and receive a streamed response. |
| `POST` | `/chat/resume` | Resume a paused run with an approval decision. |
| `POST` | `/upload` | Upload and index a PDF for a conversation. |

See `/docs` for request schemas and interactive API exploration.

## Project structure

```text
.
├── app/
│   ├── api.py          # FastAPI backend and streaming/API routes
│   └── app.py          # Streamlit chat interface
├── src/
│   ├── agent.py        # LangGraph agent and SQLite checkpointing
│   └── tools.py        # Agent tools and PDF retrieval/indexing
├── .streamlit/         # Streamlit configuration
├── requirements.txt
└── README.md
```

## Data and privacy notes

- Chat checkpoints are stored locally in `chatbot.db`.
- Uploaded PDFs are written under `uploads/`; their FAISS indexes are stored under `faiss_db/`.
- Chat prompts and tool requests are sent to the configured model or external services when those capabilities are used. Avoid uploading sensitive information unless you have reviewed the relevant providers' privacy terms.
- The purchase action is a **simulation**, not a brokerage integration.
- The API currently enables permissive CORS. Before deploying publicly, restrict allowed origins, add appropriate authentication and upload limits, and review the API's security configuration.
- Do not expose this development setup directly to the public internet with secrets or private documents.

## Contributing

Contributions and suggestions are welcome. For changes, please open an issue to discuss the proposal, then submit a pull request with a clear description and any relevant testing notes.

## License

No license is currently specified in this repository. Unless a license is added, the project is not explicitly released for reuse, modification, or redistribution.

---

<div align="center">

Made with Python · LangGraph · Streamlit · FastAPI

</div>
