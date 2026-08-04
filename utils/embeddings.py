"""
embeddings.py
Handles chunking of extracted page text and generation of vector embeddings
using a local Sentence Transformers model.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict


def chunk_pages(
    pages_data: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict]:
    """
    Split page-level text into smaller overlapping chunks, preserving
    source and page metadata on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages_data:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load and return a local Sentence Transformers embedding model via LangChain's
    HuggingFaceEmbeddings wrapper. This model runs on your machine — no API key,
    no network call, no cost.

    Args:
        model_name: HuggingFace model identifier. all-MiniLM-L6-v2 is a strong
                    default: fast, small (~80MB), and produces 384-dimensional
                    vectors with solid semantic quality.

    Returns:
        A LangChain-compatible embeddings object, usable directly with ChromaDB.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}  # improves similarity search quality
    )