"""
app.py
Streamlit UI for the RAG chatbot: PDF upload, chat interface, retrieved
chunk display, and configurable retrieval settings.
"""

import streamlit as st
import os
from pathlib import Path

from utils.pdf_loader import load_multiple_pdfs
from utils.embeddings import chunk_pages, get_embedding_model
from utils.retriever import build_vectorstore, load_vectorstore, load_chunks, PERSIST_DIR
from utils.chat import ChatSession

DATA_DIR = "data"

import html as _html


def _esc(value) -> str:
    """Escape untrusted value (file names, PDF text) before injecting into
    styled HTML. PDF content is arbitrary — never render it raw."""
    return _html.escape(str(value), quote=True)


st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# ---- Modern dark dashboard styling ----
st.markdown("""
<style>
:root {
    --bg: #0e1117;
    --bg-soft: #161b24;
    --card: #1a2029;
    --card-hover: #212a36;
    --border: #2a3442;
    --text: #e6edf3;
    --text-dim: #9aa7b4;
    --accent: #8b5cf6;
    --accent-2: #22d3ee;
    --grad: linear-gradient(135deg, #8b5cf6, #22d3ee);
    --user: #262e3c;
    --assistant-bubble: #1c2330;
}

/* Root background + text */
.stApp {
    background:
        radial-gradient(1200px at 15% -10%, rgba(139,92,246,0.10), transparent 55%),
        radial-gradient(1000px at 95% 10%, rgba(34,211,238,0.08), transparent 55%),
        var(--bg);
    color: var(--text);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.25rem; max-width: 1050px; }

/* ------------------------------------------------------------------------- *
   Hero header
 * ------------------------------------------------------------------------- */
.hero {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 24px; margin-bottom: 18px;
    border-radius: 16px;
    background: linear-gradient(180deg, #161b24, #12161f);
    border: 1px solid var(--border);
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}
.hero .logo {
    width: 46px; height: 46px; border-radius: 12px; flex: 0 0 auto;
    background: var(--grad); display: grid; place-items: center;
    font-size: 24px; box-shadow: 0 4px 14px rgba(139,92,246,0.35);
}
.hero h1 { font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;
           background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: 2px 0 0; color: var(--text-dim); font-size: 0.9rem; }

/* ------------------------------------------------------------------------- *
   Chat bubbles — user vs assistant via inner avatar presence
 * ------------------------------------------------------------------------- */
[data-testid="stChatMessage"] {
    background: var(--assistant-bubble);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2px 14px;
    max-width: 92%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.22);
}
[data-testid="stChatMessage"]:has([data-testid="stUserAvatar"]) {
    background: var(--user);
    border-color: #3b4857;
    margin-left: auto;
}
[data-testid="stChatMessage"]:has([data-testid="stAssistantAvatar"]) {
    background: var(--assistant-bubble);
    border-color: rgba(139,92,246,0.35);
}

/* ------------------------------------------------------------------------- *
   Source cards
 * ------------------------------------------------------------------------- */
.source-box {
    background: linear-gradient(180deg, var(--card), #141922);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 12px 14px; margin: 10px 0;
    border-radius: 10px;
    font-size: 0.85em; line-height: 1.5;
    color: var(--text-dim);
}
.source-box b { color: var(--text); }

/* Expanders */
[data-testid="stExpander"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
}
[data-testid="stExpander"] summary { color: var(--text); font-weight: 600; }
[data-testid="stExpander"] details { color: var(--text-dim); }

/* ------------------------------------------------------------------------- *
   Sidebar
 * ------------------------------------------------------------------------- */
[data-testid="stSidebar"] { background: #10141c; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown { color: var(--text); }
.sidebar-brand {
    padding: 14px; margin-bottom: 10px; border-radius: 12px;
    background: var(--grad); color: #0a0e14; font-weight: 800;
    font-size: 1.1rem; text-align: center; letter-spacing: 0.02em;
}
[data-testid="stSidebar"] [data-testid="stSlider"] label { color: var(--text-dim); }
div[data-baseweb="slider"] div[role="slider"] { background: var(--accent); border-color: var(--accent); }

/* Buttons */
.stButton button, .stDownloadButton button {
    border-radius: 10px; background: var(--card); color: var(--text);
    border: 1px solid var(--border); transition: all .15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: var(--card-hover); border-color: var(--accent);
    box-shadow: 0 4px 12px rgba(139,92,246,0.25);
}
.stButton button[kind="primary"] {
    background: var(--grad); color: #fff; border: none; font-weight: 600;
}
.stButton button[kind="primary"]:hover { filter: brightness(1.08); box-shadow: 0 6px 16px rgba(139,92,246,0.4); }

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}
[data-testid="stChatInput"] textarea { color: var(--text); }

/* Alerts / uploader / spinner */
[data-testid="stAlert"], .stAlert { border-radius: 10px; border: 1px solid var(--border); }
[data-testid="stFileUploaderDropzone"] {
    background: var(--card); border: 1px dashed var(--border); border-radius: 12px;
}
.stSpinner > div { border-color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# Hero banner replacing default title
st.markdown("""
<div class="hero">
    <div class="logo">📄</div>
    <div>
        <h1>RAG Chatbot</h1>
        <p>Ask questions grounded in your uploaded documents.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar: settings + upload ----
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ RAG Studio</div>', unsafe_allow_html=True)
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=50)
    top_k = st.slider("Top-k retrieval", 1, 10, 6)

    st.divider()
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    process_btn = st.button("Process Documents", type="primary")

    st.divider()
    if st.button("Reset Conversation"):
        if "chat_session" in st.session_state:
            st.session_state.chat_session.reset()
        st.session_state.messages = []
        st.rerun()

# ---- Session state initialization ----
if "embedding_model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedding_model = get_embedding_model()

if "chunks" not in st.session_state:
    st.session_state.chunks = []  # populated when PDFs are processed

if "chunks" not in st.session_state:
    # Load persisted chunks from disk if they exist (from a previous session's
    # processed PDFs). Empty list if no vectorstore was ever built, or if it
    # was built before this persistence fix existed.
    st.session_state.chunks = load_chunks()

if "vectorstore" not in st.session_state:
    # Only reuse the saved vectorstore if we ALSO have its matching chunks —
    # HybridRetriever needs both to build its BM25 index. Without chunks,
    # a stale vectorstore is unusable, so treat it as not present instead
    # of crashing.
    if (os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)
            and st.session_state.chunks):
        st.session_state.vectorstore = load_vectorstore(st.session_state.embedding_model)
    else:
        st.session_state.vectorstore = None

if "chat_session" not in st.session_state and st.session_state.vectorstore is not None:
    st.session_state.chat_session = ChatSession(
        st.session_state.vectorstore, st.session_state.chunks, top_k=top_k
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Process uploaded documents ----
if process_btn and uploaded_files:
    with st.spinner("Processing documents... this may take a moment."):
        try:
            Path(DATA_DIR).mkdir(exist_ok=True)
            saved_paths = []
            for uploaded_file in uploaded_files:
                save_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(save_path)

            pages = load_multiple_pdfs(saved_paths)
            if not pages:
                st.error("No extractable text found in the uploaded PDF(s).")
            else:
                chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                st.session_state.chunks = chunks
                st.session_state.vectorstore = build_vectorstore(
                    chunks, st.session_state.embedding_model
                )
                st.session_state.chat_session = ChatSession(
                    st.session_state.vectorstore, st.session_state.chunks, top_k=top_k
                )
                st.success(f"Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks.")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

elif process_btn and not uploaded_files:
    st.warning("Please upload at least one PDF before processing.")

# ---- Chat interface ----
if st.session_state.vectorstore is None:
    st.info("👈 Upload and process at least one PDF to start chatting.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Retrieved sources"):
                    for chunk in msg["sources"]:
                        project = chunk.get("project")
                        tag = f"<span class='tag'>{_esc(project)}</span>" if project else ""
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<b>{_esc(chunk['source'])}</b> — page {chunk['page']} "
                            f"(score: {chunk['score']:.4f}){tag}<br>{_esc(chunk['text'][:300])}..."
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chat_session.ask(question)
                    st.markdown(result["answer"])

                    with st.expander("📚 Retrieved sources"):
                        for chunk in result["retrieved_chunks"]:
                            project = chunk.get("project")
                            tag = f"<span class='tag'>{_esc(project)}</span>" if project else ""
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<b>{_esc(chunk['source'])}</b> — page {chunk['page']} "
                                f"(score: {chunk['score']:.4f}){tag}<br>{_esc(chunk['text'][:300])}..."
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["retrieved_chunks"]
                    })
                except Exception as e:
                    error_msg = f"Something went wrong: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})