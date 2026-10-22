
"""
Emerald Assistant — simple, stable Streamlit frontend.

Run:
    streamlit run app.py

Optional:
    BACKEND_URL=http://localhost:8000
"""

import json
import os
import uuid

import requests
import streamlit as st


# =========================================================
# 1. PAGE CONFIG
# =========================================================

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Emerald Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. SIMPLE EMERALD THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --emerald: #059669;
        --emerald-dark: #064e3b;
        --emerald-light: #ecfdf5;
    }

    .stApp {
        background-color: #f8fcf9;
    }

    /* Minimal styling only. Keep Streamlit's native layout. */

    [data-testid="stSidebar"] {
        background-color: #f0faf4;
        border-right: 1px solid #d1fae5;
    }

    .st-key-sidebar_header {
        position: sticky;
        top: 0;
        z-index: 10;
        background: #f0faf4;
        padding-bottom: 0.5rem;
    }

    .st-key-sidebar_footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 300px;
        max-width: 100vw;
        box-sizing: border-box;
        z-index: 10;
        background: #f0faf4;
        padding: 0.5rem 20px 0;
        border-top: 1px solid #d1fae5;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #064e3b;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1000px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }

    h1, h2, h3 {
        color: #064e3b;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #e1f3e8;
        border-radius: 12px;
        padding: 1rem;
        background: #ffffff;
    }

    [data-testid="stChatInput"] {
        border-color: #a7f3d0;
        border-radius: 12px;
        background: #ffffff;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #059669;
    }

    div.stButton > button[kind="primary"] {
        background-color: #059669;
        border-color: #059669;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #047857;
        border-color: #047857;
        color: white;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #d1fae5;
        border-radius: 10px;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. BACKEND HELPERS
# =========================================================

def api_get(path: str):
    response = requests.get(
        f"{API_URL}{path}",
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def stream_events(path: str, payload: dict):
    """Read server-sent events from the backend."""
    with requests.post(
        f"{API_URL}{path}",
        json=payload,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                try:
                    yield json.loads(line[6:])
                except json.JSONDecodeError:
                    continue


def backend_online() -> bool:
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        return response.ok
    except requests.RequestException:
        return False


# =========================================================
# 4. SESSION STATE
# =========================================================

ss = st.session_state

ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("messages", [])
ss.setdefault("pending", None)
ss.setdefault("resume", None)
ss.setdefault("indexed", None)
ss.setdefault("doc_name", None)
ss.setdefault("queued", None)


def new_chat():
    ss.thread_id = str(uuid.uuid4())
    ss.messages = []
    ss.pending = None
    ss.resume = None
    ss.indexed = None
    ss.doc_name = None
    ss.queued = None


def load_thread(thread_id: str):
    try:
        data = api_get(f"/threads/{thread_id}/messages")

        ss.thread_id = thread_id
        ss.messages = data.get("messages", [])
        ss.pending = data.get("pending")
        ss.resume = None
        ss.queued = None

    except requests.RequestException as exc:
        st.session_state["load_error"] = str(exc)


def queue_prompt(text: str):
    ss.queued = text


def decide(approved: bool):
    ss.resume = approved
    ss.pending = None


# =========================================================
# 5. MESSAGE AND TOOL RENDERING
# =========================================================

def render_tools(tools: list[dict]):
    if not tools:
        return

    with st.expander(
        f"Tools used ({len(tools)})",
        expanded=False,
    ):
        for index, tool in enumerate(tools, start=1):
            running = tool.get("result") is None
            name = tool.get("name", "Tool")

            status = "Running" if running else "Completed"

            st.markdown(f"**{index}. {name}** — {status}")

            with st.expander(
                f"View {name} details",
                expanded=False,
            ):
                st.caption("Input")
                st.json(tool.get("args", {}))

                if not running:
                    st.caption("Output")
                    st.code(
                        str(tool.get("result", "")),
                        language="text",
                    )


def render_message(message: dict):
    role = message.get("role", "assistant")

    if role not in ("user", "assistant"):
        role = "assistant"

    with st.chat_message(role):
        if role == "assistant":
            render_tools(message.get("tools", []))

        content = message.get("content", "")
        if content:
            st.markdown(content)


# =========================================================
# 6. STREAM ASSISTANT RESPONSE
# =========================================================

def stream_assistant(
    path: str,
    payload: dict,
    base: dict | None = None,
):
    message = base or {
        "role": "assistant",
        "content": "",
        "tools": [],
    }

    message.setdefault("role", "assistant")
    message.setdefault("content", "")
    message.setdefault("tools", [])

    # Reuse the same assistant message when resuming a run.
    with st.chat_message("assistant"):
        tool_status = st.empty()
        answer_area = st.empty()

        tool_status.caption("Preparing response...")

        try:
            for event in stream_events(path, payload):
                event_type = event.get("type")

                if event_type == "token":
                    message["content"] += event.get("content", "")

                elif event_type == "tool_call":
                    message["tools"].append(
                        {
                            "id": event.get("id"),
                            "name": event.get("name", "Tool"),
                            "args": event.get("args", {}),
                            "result": None,
                        }
                    )

                elif event_type == "tool_result":
                    for tool in message["tools"]:
                        if tool.get("id") == event.get("id"):
                            tool["result"] = event.get("content", "")

                elif event_type == "interrupt":
                    ss.pending = event.get("value")

                elif event_type == "error":
                    message["content"] += (
                        "\n\n**Error:** "
                        + str(event.get("message", "Unknown error"))
                    )

                # Keep status separate from the streamed answer.
                if message["tools"]:
                    completed = sum(
                        tool.get("result") is not None
                        for tool in message["tools"]
                    )
                    total = len(message["tools"])

                    tool_status.caption(
                        f"Tools: {completed}/{total} completed"
                    )
                else:
                    tool_status.caption("Generating response...")

                if message["content"]:
                    answer_area.markdown(
                        message["content"] + " ▌"
                    )

                elif not message["tools"]:
                    answer_area.markdown(
                        "_Thinking..._"
                    )

            if message["content"]:
                answer_area.markdown(message["content"])

            if message["tools"]:
                tool_status.caption(
                    f"Tool calls: {len(message['tools'])}"
                )

        except requests.RequestException as exc:
            message["content"] += (
                f"\n\n**Backend connection error:** {exc}"
            )
            answer_area.markdown(message["content"])

        except (ValueError, KeyError, TypeError) as exc:
            message["content"] += (
                f"\n\n**Response processing error:** {exc}"
            )
            answer_area.markdown(message["content"])

    if message["content"] or message["tools"]:
        ss.messages.append(message)

    st.rerun()


# =========================================================
# 7. SIDEBAR
# =========================================================

online = backend_online()

with st.sidebar:
    with st.container(key="sidebar_header"):
        st.title("✦ Emerald Assistant")
        st.caption("Your personal agent workspace")

    st.button(
        "＋ New chat",
        type="primary",
        use_container_width=True,
        on_click=new_chat,
    )

    st.divider()

    # Conversation history
    st.subheader("Conversation history")

    if online:
        try:
            threads = api_get("/threads").get("threads", [])
        except requests.RequestException:
            threads = []
            st.caption("Unable to load conversation history.")

        if not threads:
            st.caption("No saved conversations yet.")

        for thread in threads:
            thread_id = thread.get("thread_id")

            if not thread_id:
                continue

            title = thread.get("title") or "Untitled conversation"

            st.button(
                title,
                key=f"thread_{thread_id}",
                use_container_width=True,
                on_click=load_thread,
                args=(thread_id,),
            )

    st.divider()

    with st.container(key="sidebar_footer"):
        if online:
            st.success("Backend connected")
        else:
            st.error("Backend offline")

        st.caption(f"Backend: {API_URL}")


# =========================================================
# 8. MAIN HEADER
# =========================================================

st.title("✦ Emerald Assistant")

st.caption(
    "Ask questions, research topics, retrieve live information, "
    "perform calculations, or chat with your PDF."
)

st.divider()


# =========================================================
# 9. BACKEND AVAILABILITY
# =========================================================

if not online:
    st.error(
        f"Cannot connect to the backend at {API_URL}."
    )
    st.info(
        "Start your backend with: "
        "`uvicorn api:app --port 8000`"
    )
    st.stop()


# =========================================================
# 10. WELCOME SCREEN AND SUGGESTIONS
# =========================================================

chat_input = st.chat_input(
    "Message Emerald Assistant...",
    accept_file=True,
    file_type=["pdf"],
    key="chat_input",
    disabled=bool(ss.pending),
)

prompt = chat_input.text if chat_input else ""

if chat_input:
    for uploaded_file in chat_input.files:
        upload_key = (
            ss.thread_id,
            uploaded_file.name,
            uploaded_file.size,
        )

        if ss.indexed == upload_key:
            continue

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    data={
                        "thread_id": ss.thread_id,
                    },
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        ),
                    },
                    timeout=600,
                )

                response.raise_for_status()

                ss.indexed = upload_key
                ss.doc_name = uploaded_file.name
                st.success(f"{uploaded_file.name} indexed successfully.")

            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

if ss.doc_name:
    st.caption(f"Active document: {ss.doc_name}")

if not prompt and ss.queued:
    prompt = ss.queued
    ss.queued = None


if not ss.messages and not ss.pending:
    st.subheader("What can I help you with?")

    st.write(
        "Choose a suggestion or type your own question below."
    )

    suggestions = [
        "What's the weather in London right now?",
        "Summarize the document I uploaded",
        "Calculate 15% of 2,480 plus the square root of 144",
        "What's the latest price of AAPL?",
    ]

    col1, col2 = st.columns(2)

    for index, suggestion in enumerate(suggestions):
        target = col1 if index % 2 == 0 else col2

        with target:
            st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
                on_click=queue_prompt,
                args=(suggestion,),
            )


# =========================================================
# 11. RESTORE A PAUSED ASSISTANT RUN
# =========================================================

resume_decision = ss.resume
ss.resume = None

resume_base = None

if (
    resume_decision is not None
    and ss.messages
    and ss.messages[-1].get("role") == "assistant"
):
    resume_base = ss.messages.pop()


# =========================================================
# 12. DISPLAY CONVERSATION
# =========================================================

for message in ss.messages:
    render_message(message)


# =========================================================
# 13. HUMAN APPROVAL
# =========================================================

if ss.pending:
    st.warning("This action requires your approval.")

    st.write(ss.pending)

    approve_col, decline_col = st.columns(2)

    with approve_col:
        st.button(
            "Approve",
            type="primary",
            use_container_width=True,
            key="approve_action",
            on_click=decide,
            args=(True,),
        )

    with decline_col:
        st.button(
            "Decline",
            use_container_width=True,
            key="decline_action",
            on_click=decide,
            args=(False,),
        )


# =========================================================
# 14. RESUME AGENT AFTER APPROVAL
# =========================================================

if resume_decision is not None:
    stream_assistant(
        "/chat/resume",
        {
            "thread_id": ss.thread_id,
            "approved": resume_decision,
        },
        base=resume_base,
    )


# =========================================================
# 15. HANDLE NEW USER PROMPT
# =========================================================

if prompt:
    ss.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    stream_assistant(
        "/chat",
        {
            "thread_id": ss.thread_id,
            "message": prompt,
        },
    )
