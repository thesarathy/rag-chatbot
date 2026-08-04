"""
retriever.py
Handles storing document chunk embeddings in ChromaDB and performing
similarity search to retrieve relevant chunks for a given query.
"""

from langchain_community.vectorstores import Chroma
from typing import List, Dict


PERSIST_DIR = "vectorstore"


def build_vectorstore(chunks: List[Dict], embedding_model) -> Chroma:
    """
    Embed and store text chunks in a persistent ChromaDB collection.

    Args:
        chunks: List of {"text", "source", "page"} dicts (from chunk_pages()).
        embedding_model: A LangChain-compatible embeddings object.

    Returns:
        A Chroma vectorstore instance, already persisted to disk.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR
    )

    return vectorstore


def load_vectorstore(embedding_model) -> Chroma:
    """
    Load an existing ChromaDB collection from disk, without re-embedding.
    Use this on app startup if a vectorstore already exists.
    """
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )


def similarity_search(vectorstore: Chroma, query: str, top_k: int = 4) -> List[Dict]:
    """
    Retrieve the top_k most relevant chunks for a given query.

    Args:
        vectorstore: A Chroma instance.
        query: The user's question.
        top_k: Number of chunks to retrieve (configurable — exposed in UI later).

    Returns:
        List of dicts: {"text", "source", "page", "score"} — score is the
        similarity distance (lower = more similar, for Chroma's default metric).
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    retrieved = []
    for doc, score in results:
        retrieved.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "score": score
        })

    return retrieved