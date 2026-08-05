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
from utils.retriever import build_vectorstore, load_vectorstore, PERSIST_DIR
from utils.chat import ChatSession

DATA_DIR = "data"

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# ---- Dark theme styling ----
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stChatMessage { background-color: #1a1d24; border-radius: 8px; }
    .source-box {
        background-color: #1a1d24;
        border-left: 3px solid #4f8bf9;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 RAG Chatbot")
st.caption("Ask questions grounded in your uploaded documents.")

# ---- Sidebar: settings + upload ----
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=50)
    top_k = st.slider("Top-k retrieval", 1, 10, 4)

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

if "vectorstore" not in st.session_state:
    # Reuse existing vectorstore on disk if one exists, so we don't
    # re-embed everything every time the app restarts.
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.session_state.vectorstore = load_vectorstore(st.session_state.embedding_model)
    else:
        st.session_state.vectorstore = None

if "chat_session" not in st.session_state and st.session_state.vectorstore is not None:
    st.session_state.chat_session = ChatSession(st.session_state.vectorstore, top_k=top_k)

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

                # Merge into the existing vectorstore if one exists, otherwise create new.
                # Both cases use build_vectorstore, which persists to PERSIST_DIR —
                # ChromaDB's persistence handles the merge automatically since it's
                # backed by the same directory.
                st.session_state.vectorstore = build_vectorstore(
                    chunks, st.session_state.embedding_model
                )
                st.session_state.chat_session = ChatSession(
                    st.session_state.vectorstore, top_k=top_k
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
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<b>{chunk['source']}</b> — page {chunk['page']} "
                            f"(score: {chunk['score']:.4f})<br>{chunk['text'][:300]}..."
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
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<b>{chunk['source']}</b> — page {chunk['page']} "
                                f"(score: {chunk['score']:.4f})<br>{chunk['text'][:300]}..."
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