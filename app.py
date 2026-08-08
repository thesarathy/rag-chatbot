"""
app.py
Streamlit UI for the RAG chatbot: PDF upload, chat interface, retrieved
chunk display, and configurable retrieval settings.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional

from utils.pdf_loader import load_multiple_pdfs
from utils.embeddings import chunk_pages, get_embedding_model
from utils.retriever import build_vectorstore, load_vectorstore, load_chunks, PERSIST_DIR
from utils.chat import ChatSession
import utils.chat_store as chat_store

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
    --bg: #07090f;
    --bg-2: #0a0e16;
    --panel: #0d1220;
    --card: #121a2c;
    --card-hover: #182036;
    --border: rgba(255,255,255,.06);
    --border-strong: rgba(139,124,246,.32);
    --text: #eef2f8;
    --text-2: #c6cfe0;
    --text-dim: #9ca9bd;
    --accent: #8b7cf6;
    --accent-2: #2fd3e8;
    --grad: linear-gradient(120deg, #8b7cf6 0%, #6b6cf0 50%, #2fd3e8 100%);
    --grad-soft: linear-gradient(120deg, rgba(139,124,246,.10), rgba(47,211,236,.10));
    --shadow: 0 8px 28px rgba(0,0,0,.30);
    --glow: 0 0 0 1px rgba(139,124,246,.12), 0 6px 24px rgba(99,102,241,.16);
    --font: "Inter", "Segoe UI", system-ui, -apple-system, "PingFang SC", sans-serif;
}

/* Root background + font */
html, body, .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}
/* Subdue the aurora — premium apps keep the ambient glow quiet */
.stApp {
    background:
        radial-gradient(900px 540px at 85% -6%, rgba(139,124,246,.12), transparent 60%),
        radial-gradient(820px 500px at 104% -2%, rgba(47,211,236,.09), transparent 55%),
        radial-gradient(1000px 720px at 50% 114%, rgba(99,102,241,.07), transparent 60%),
        linear-gradient(180deg, var(--bg), var(--bg-2));
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.35rem 1rem 4rem; max-width: 920px; }

/* De-Streamlit: drop the default chrome so it reads like a product */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { opacity: 0; pointer-events: none; }
[data-testid="stDecoration"] { display: none; }

/* ------------------------------------------------------------------------- *
   App bar (hero header)
 * ------------------------------------------------------------------------- */
.appbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; padding: 14px 18px; margin-bottom: 20px;
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.012));
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,.16);
}
.appbar .brand { display: flex; align-items: center; gap: 13px; min-width: 0; }
.appbar .mark {
    width: 42px; height: 42px; border-radius: 12px; flex: 0 0 auto;
    background: var(--grad); display: grid; place-items: center;
    font-size: 19px; color: #0a0e14; box-shadow: var(--glow);
}
.appbar h1 { font-size: 1.28rem; font-weight: 750; margin: 0; letter-spacing: -.02em;
             background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.appbar p { margin: 2px 0 0; color: var(--text-dim); font-size: .92rem; }
.appbar .stats { display: flex; gap: 7px; flex-wrap: wrap; justify-content: flex-end; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 999px;
    background: rgba(255,255,255,.035); border: 1px solid var(--border);
    font-size: .76rem; color: var(--text-2); white-space: nowrap;
}
.chip b { color: var(--text); font-weight: 700; }
.chip .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-2);
            box-shadow: 0 0 8px var(--accent-2); }

/* ------------------------------------------------------------------------- *
   Chat bubbles — user vs assistant via inner avatar presence
 * ------------------------------------------------------------------------- */
[data-testid="stChatMessage"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 10px 14px;
    max-width: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,.18);
}
.msglabel {
    display: flex; align-items: center; gap: 7px;
    font-size: .72rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--text-dim);
    margin-bottom: 6px;
}
.msglabel .msgrad { background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing:.06em; }
.msglabel .marshot { display: inline-grid; place-items: center; width: 18px; height: 18px;
    border-radius: 6px; background: var(--grad); color: #0a0e14; font-size: 10px; font-weight: 800; }
[data-testid="stChatMessage"]:has(.ulab) {
    background: linear-gradient(135deg, rgba(139,124,246,.13), rgba(99,102,241,.09));
    border-color: rgba(139,124,246,.24);
    border-bottom-right-radius: 5px;
    margin-left: auto;
    max-width: 78%;
}
[data-testid="stChatMessage"]:has(.alab) { margin-right: auto; border-bottom-left-radius: 5px; max-width: 88%; }

/* ------------------------------------------------------------------------- *
   Source cards
 * ------------------------------------------------------------------------- */
.source-box {
    background: rgba(255,255,255,.025);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    padding: 11px 13px; margin: 8px 0;
    border-radius: 10px;
    font-size: .86rem; line-height: 1.6;
    color: var(--text-2);
}
.source-box b { color: var(--text); font-weight: 650; }
.source-box .pg { color: var(--accent-2); font-weight: 650; }

/* Expanders */
[data-testid="stExpander"] {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden;
}
[data-testid="stExpander"] summary { color: var(--text); font-weight: 600; }
[data-testid="stExpander"] details { color: var(--text-dim); }

/* ------------------------------------------------------------------------- *
   Sidebar
 * ------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--panel), var(--bg-2));
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown { color: var(--text); }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.1rem .9rem 2rem; }
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 14px; margin-bottom: 14px; border-radius: 12px;
    background: var(--grad); color: #0a0e14; font-weight: 800;
    font-size: 1.0rem; letter-spacing: .01em; box-shadow: var(--glow);
}
[data-testid="stSidebar"] [data-testid="stSlider"] label { color: var(--text-dim); }
div[data-baseweb="slider"] div[role="slider"] { background: var(--accent); border-color: var(--accent); }
[data-testid="stSidebar"] summary { color: var(--text); }
[data-testid="stSidebar"] [data-testid="stHeading"] {
    text-transform: uppercase; letter-spacing: .13em; font-size: .82rem;
    color: var(--text-2); margin-bottom: 10px; font-weight: 700;
}

/* Buttons — uniform height, quiet interactions */
.stButton button, .stDownloadButton button {
    border-radius: 10px; background: var(--card); color: var(--text);
    border: 1px solid var(--border); font-weight: 500; min-height: 38px;
    transition: border-color .12s ease, box-shadow .12s ease, transform .08s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--border-strong); box-shadow: 0 2px 10px rgba(0,0,0,.22);
}
.stButton button:active, .stDownloadButton button:active { transform: translateY(1px); }
.stButton button[kind="primary"] {
    background: var(--grad); color: #fff; border: none; font-weight: 600;
    box-shadow: 0 3px 12px rgba(139,124,246,.30);
}
.stButton button[kind="primary"]:hover { filter: brightness(1.05); box-shadow: 0 4px 14px rgba(139,124,246,.42); }

/* Chat input — anchored, refined, not floating heavy */
[data-testid="stChatInput"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 13px; box-shadow: 0 2px 12px rgba(0,0,0,.22); overflow: hidden;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(139,124,246,.14), 0 2px 12px rgba(0,0,0,.22);
}
[data-testid="stChatInput"] textarea { color: var(--text); }

/* Alerts / uploader / spinner */
[data-testid="stAlert"], .stAlert { border-radius: 12px; border: 1px solid var(--border); }
[data-testid="stFileUploaderDropzone"] {
    background: var(--card); border: 1px dashed var(--border); border-radius: 12px;
    padding: 14px; transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent); background: rgba(139,124,246,.05);
}
[data-testid="stFileUploaderDropzone"] small { color: var(--text-dim); }
.stSpinner > div { border-color: var(--accent); }

/* ------------------------------------------------------------------------- *
   Conversations sidebar — non-boring cards
 * ------------------------------------------------------------------------- */
.conv-title {
    font-size: .8rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-2); display: flex; align-items: center; gap: 10px; margin: 2px 0 10px;
}
.conv-title::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(139,124,246,.32), transparent);
}

/* New chat button gets a touch of flair */
[data-testid="stSidebar"] button[kind="primary"] { font-weight: 600; letter-spacing: .01em; }

/* Conversation switch buttons read like message-app cards */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    text-align: left; white-space: normal; line-height: 1.3;
    min-height: 50px; justify-content: flex-start; padding: 7px 11px;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p { margin: 0; font-size: .84rem; color: var(--text-2); }

/* Compact rename/delete ghost buttons */
[data-testid="stSidebar"] [data-testid="stButton"] button[disabled="true"] { opacity: .5; }

/* ------------------------------------------------------------------------- *
   Tag chips on source cards
 * ------------------------------------------------------------------------- */
.tag {
    display: inline-block; background: rgba(139,92,246,.14); color: #c4b5fd;
    border: 1px solid rgba(139,92,246,.32); border-radius: 999px;
    padding: 1px 9px; font-size: .72rem; margin-left: 6px; vertical-align: middle;
}

/* ------------------------------------------------------------------------- *
   Empty / get-started state
 * ------------------------------------------------------------------------- */
.emptyhero {
    margin: 8vh auto; text-align: center;
    padding: 40px 36px 34px; border-radius: 18px;
    background:
        radial-gradient(620px 300px at 50% 0%, rgba(139,124,246,.09), transparent 60%),
        linear-gradient(180deg, var(--card), var(--panel));
    border: 1px solid var(--border);
    box-shadow: 0 4px 18px rgba(0,0,0,.20);
    max-width: 560px;
}
.emptyhero .eyebrow {
    font-size: .72rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: var(--accent-2); margin-bottom: 16px;
}
.emptyhero .orb {
    width: 52px; height: 52px; margin: 0 auto 18px; border-radius: 15px;
    background: var(--grad); display: grid; place-items: center; font-size: 22px;
    color: #0a0e14; box-shadow: var(--glow); animation: floaty 4s ease-in-out infinite;
}
.emptyhero h2 { font-size: 1.3rem; font-weight: 750; margin: 0 0 8px; letter-spacing: -.01em; }
.emptyhero p { color: var(--text-dim); font-size: .97rem; line-height: 1.62; margin: 0 auto; max-width: 46ch; }
.emptyhero .steps { display: flex; gap: 10px; margin: 24px 0 0; }
.emptyhero .step {
    flex: 1 1 0; min-width: 0; text-align: left;
    background: rgba(255,255,255,.03); border: 1px solid var(--border);
    border-radius: 11px; padding: 10px 12px;
}
.emptyhero .step .n { font-size: .78rem; font-weight: 800; color: var(--text); }
.emptyhero .step .n::after { content: ""; display: block; width: 14px; height: 2px;
    border-radius: 2px; background: var(--grad); margin: 4px 0 7px; }
.emptyhero .step span { font-size: .8rem; color: var(--text-2); }
@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }

/* Message author avatars cleaned up */
[data-testid="stChatMessageAvatar"] { border-radius: 9px; }

/* Scrollbar polish */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,.42); }
.appbar .pulse { animation: floaty 4s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# --- App bar header (presentational, reads existing session state) ----
_ready = st.session_state.get("vectorstore") is not None
_n_convos = len(st.session_state.get("conversations", {}))
_n_chunks = len(st.session_state.get("chunks", []))
_n_docs = len({c.get("source") for c in st.session_state.get("chunks", []) if c.get("source")})
if _ready:
    _stats_html = f"""<div class="stats">
        <span class="chip"><span class="dot"></span>Index ready</span>
        <span class="chip">💬 <b>{_n_convos}</b></span>
        <span class="chip">📄 <b>{_n_docs}</b></span>
        <span class="chip">🧩 <b>{_n_chunks}</b> chunks</span>
    </div>"""
else:
    _stats_html = f"""<div class="stats">
        <span class="chip"><span class="dot"></span>Awaiting index</span>
        <span class="chip">💬 <b>{_n_convos}</b></span>
    </div>"""

st.markdown(f"""
<div class="appbar">
    <div class="brand">
        <div class="mark">✦</div>
        <div>
            <h1>RAG Chatbot</h1>
            <p>Ask anything, grounded strictly in your documents.</p>
        </div>
    </div>
    {_stats_html}
</div>
""", unsafe_allow_html=True)

# ---- Conversation store init (must precede sidebar use) ----
if "conversations" not in st.session_state:
    st.session_state.conversations = chat_store.load_conversations()
if "active_id" not in st.session_state:
    ids = chat_store.ordered_ids(st.session_state.conversations)
    st.session_state.active_id = ids[0] if ids else None
if "rename_id" not in st.session_state:
    st.session_state.rename_id = None
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None


def _active_conv():
    """Return the currently-selected conversation dict (or None)."""
    cid = st.session_state.active_id
    return st.session_state.conversations.get(cid) if cid else None


def _cid_of(convo) -> Optional[str]:
    """Return the id key whose value is `convo` (identity-safe)."""
    for cid, c in st.session_state.conversations.items():
        if c is convo:
            return cid
    return None


def _persist():
    chat_store.save_conversations(st.session_state.conversations)


def _preview(convo) -> str:
    """Body line for a sidebar card: role + last-message excerpt, or a hint."""
    msgs = convo.get("messages") or []
    if not msgs:
        return "💬 No messages yet"
    last = msgs[-1]
    who = "You" if last.get("role") == "user" else "Bot"
    text = " ".join(str(last.get("content", "")).split())
    return f"{who}: {text[:52]}"


def _rel_time(iso: str) -> str:
    """Compact human relative time from an ISO-8601 timestamp."""
    if not iso:
        return ""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        return ""
    s = (datetime.now() - dt).total_seconds()
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86_400:
        return f"{int(s // 3600)}h ago"
    if s < 7 * 86_400:
        return f"{int(s // 86_400)}d ago"
    return dt.strftime("%b %d")


# ---- Sidebar: conversations + settings + upload ----
with st.sidebar:
    st.markdown('<div class="sidebar-brand">✦ RAG Studio</div>', unsafe_allow_html=True)

    # Upload stays pinned near the top so it's never buried under a long chat list
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    process_btn = st.button("Process Documents", type="primary")

    st.divider()

    st.markdown('<div class="conv-title">💬 Conversations</div>', unsafe_allow_html=True)
    if st.button("＋ New conversation", type="primary", use_container_width=True):
        convo = chat_store.create_conversation(st.session_state.conversations)
        st.session_state.active_id = _cid_of(convo)
        st.session_state.rename_id = None
        st.session_state.confirm_delete = None
        _persist()
        st.rerun()

    # Long chat lists scroll inside this fixed-height box instead of shoving
    # the settings/upload sections out of view.
    with st.container(height=340):
        for cid in chat_store.ordered_ids(st.session_state.conversations):
            convo = st.session_state.conversations[cid]
            is_active = cid == st.session_state.active_id
            col_sel, col_ren, col_del = st.columns([6, 1, 1])
            # Switch card: name on the first line, preview + relative time below
            card = f"{convo['name']}\n{_preview(convo)} · {_rel_time(convo.get('updated',''))}"
            if col_sel.button(
                card, key=f"open_{cid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_id = cid
                st.session_state.rename_id = None
                st.session_state.confirm_delete = None
                st.rerun()
            if col_ren.button("✎", key=f"ren_{cid}", help="Rename"):
                st.session_state.rename_id = cid
            if col_del.button(
                "🗑", key=f"del_{cid}", help="Delete", type="secondary",
            ):
                if st.session_state.confirm_delete == cid:
                    del st.session_state.conversations[cid]
                    if st.session_state.active_id == cid:
                        after = chat_store.ordered_ids(st.session_state.conversations)
                        st.session_state.active_id = after[0] if after else None
                    st.session_state.confirm_delete = None
                    _persist()
                    st.rerun()
                else:
                    st.session_state.confirm_delete = cid
            if st.session_state.confirm_delete == cid:
                st.warning(f"Click 🗑 again to delete “{convo['name']}”.", icon="⚠️")
            # Inline rename
            if st.session_state.rename_id == cid:
                new_name = st.text_input(
                    "Name", value=convo["name"], key=f"rename_input_{cid}", label_visibility="collapsed"
                )
                if st.button("Save name", key=f"save_{cid}", type="primary", use_container_width=True):
                    convo["name"] = (new_name or convo["name"]).strip() or "New conversation"
                    chat_store.touch(convo)
                    st.session_state.rename_id = None
                    st.session_state.confirm_delete = None
                    _persist()
                    st.rerun()

    st.divider()
    if st.button("Clear current chat"):
        convo = _active_conv()
        if convo:
            convo["messages"] = []
            chat_store.touch(convo)
            _persist()
            if "chat_session" in st.session_state:
                st.session_state.chat_session.set_history(convo["messages"])
            st.rerun()

    st.divider()
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=50)
    top_k = st.slider("Top-k retrieval", 1, 10, 6)

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
    st.markdown("""
    <div class="emptyhero">
        <div class="eyebrow">Get started</div>
        <div class="orb">✦</div>
        <h2>Your knowledge, one query away</h2>
        <p>Upload a PDF, build a searchable index, then ask anything — grounded<br>strictly in your own files.</p>
        <div class="steps">
            <div class="step"><div class="n">01</div><span>Upload a PDF in the sidebar</span></div>
            <div class="step"><div class="n">02</div><span>Hit <b>Process Documents</b></span></div>
            <div class="step"><div class="n">03</div><span>Ask with confidence</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    conv = _active_conv()
    # Display chat history for the active conversation only
    if conv:
        for msg in conv["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown('<div class="msglabel ulab"><span class="marshot">✎</span> You</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="msglabel alab"><span class="marshot">✦</span><span class="msgrad">RAG Assistant</span></div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Retrieved sources"):
                        for chunk in msg["sources"]:
                            project = chunk.get("project")
                            tag = f"<span class='tag'>{_esc(project)}</span>" if project else ""
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<b>{_esc(chunk['source'])}</b> · <span class='pg'>page {_esc(chunk['page'])}</span> "
                                f"(score: {chunk['score']:.4f}){tag}<br>{_esc(chunk['text'][:300])}..."
                                f"</div>",
                                unsafe_allow_html=True
                            )

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        if conv is None:
            conv = chat_store.create_conversation(st.session_state.conversations)
            st.session_state.active_id = _cid_of(conv)
        conv["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown('<div class="msglabel ulab"><span class="marshot">✎</span> You</div>', unsafe_allow_html=True)
            st.markdown(question)

        # Scope the shared ChatSession's memory to THIS conversation (all prior
        # turns, not the just-appended one) so follow-up rewriting stays local.
        hist = [{"role": m["role"], "content": m["content"]} for m in conv["messages"][:-1]]
        st.session_state.chat_session.set_history(hist)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chat_session.ask(question)
                    st.markdown('<div class="msglabel alab"><span class="marshot">✦</span><span class="msgrad">RAG Assistant</span></div>', unsafe_allow_html=True)
                    st.markdown(result["answer"])

                    with st.expander("📚 Retrieved sources"):
                        for chunk in result["retrieved_chunks"]:
                            project = chunk.get("project")
                            tag = f"<span class='tag'>{_esc(project)}</span>" if project else ""
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<b>{_esc(chunk['source'])}</b> · <span class='pg'>page {_esc(chunk['page'])}</span> "
                                f"(score: {chunk['score']:.4f}){tag}<br>{_esc(chunk['text'][:300])}..."
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    conv["messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["retrieved_chunks"]
                    })
                except Exception as e:
                    error_msg = f"Something went wrong: {str(e)}"
                    st.error(error_msg)
                    conv["messages"].append({"role": "assistant", "content": error_msg})

        chat_store.touch(conv)
        _persist()