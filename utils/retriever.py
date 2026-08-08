"""
retriever.py
Hybrid retrieval: ChromaDB dense (semantic) search + BM25 sparse (keyword) search,
fused together, with a diversity cap so answers to enumeration-style questions
("which projects use X") aren't dominated by one document/section that happens
to have high term-frequency for the query.

Chunks are also persisted to disk alongside the vectorstore (chunks.json),
so restarting the app can reload BOTH the embeddings AND the raw chunk text/
metadata needed to rebuild the BM25 index — fixing the ZeroDivisionError that
occurred when only the vectorstore was reloaded but chunks stayed empty.
"""

import json
import os
import re
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np

PERSIST_DIR = "vectorstore"
CHUNKS_FILE = os.path.join(PERSIST_DIR, "chunks.json")

# Match whole alphanumeric words only, discarding punctuation. This is CRITICAL
# for sparse (BM25) matching: tokenizing on whitespace alone leaves punctuation
# glued to tokens ("langgraph?"), so a query ending in a question mark never
# matches the clean chunk token "langgraph" — the exact-keyword signal goes
# missing and dense (semantic) results knock the keyword-heavy page out of the
# top-k. Both the chunk index and every query must tokenize identically.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on word boundaries, dropping punctuation."""
    return _TOKEN_RE.findall((text or "").lower())


def build_vectorstore(chunks: List[Dict], embedding_model) -> Chroma:
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    vectorstore = Chroma.from_texts(
        texts=texts, embedding=embedding_model, metadatas=metadatas,
        persist_directory=PERSIST_DIR
    )
    save_chunks(chunks)
    return vectorstore


def load_vectorstore(embedding_model) -> Chroma:
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)


def save_chunks(chunks: List[Dict]) -> None:
    """Persist chunks (text + source + page + project metadata) to disk
    alongside the vectorstore, so they can be reloaded on restart."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def load_chunks() -> List[Dict]:
    """Load persisted chunks from disk. Returns an empty list if the file
    doesn't exist (e.g. a vectorstore built before this persistence fix)."""
    if not os.path.exists(CHUNKS_FILE):
        return []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class HybridRetriever:
    """
    Combines ChromaDB dense (semantic) search with BM25 sparse (keyword) search.
    Fixes cases where exact terms/acronyms (e.g. "CI/CD", "LoRA") get outranked
    by generic but semantically-similar text, AND applies a diversity cap so
    one dense, high-term-frequency section can't crowd out other relevant
    sections when a question asks for enumeration across multiple sources.
    """

    def __init__(self, vectorstore: Chroma, chunks: List[Dict]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 4, alpha: float = 0.5,
               candidate_pool: int = 20, max_per_cluster: int = 2) -> List[Dict]:
        dense_results = self.vectorstore.similarity_search_with_score(query, k=len(self.chunks))
        dense_scores = np.zeros(len(self.chunks))
        text_to_idx = {c["text"]: i for i, c in enumerate(self.chunks)}
        for doc, score in dense_results:
            idx = text_to_idx.get(doc.page_content)
            if idx is not None:
                dense_scores[idx] = 1.0 / (1.0 + score)

        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)))

        def normalize(arr):
            if arr.max() - arr.min() < 1e-9:
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())

        combined = alpha * normalize(dense_scores) + (1 - alpha) * normalize(bm25_scores)
        ranked_idx = np.argsort(-combined)[:candidate_pool]

        cluster_counts = {}
        selected = []
        for i in ranked_idx:
            cluster = self.chunks[i]["page"] // 5
            if cluster_counts.get(cluster, 0) >= max_per_cluster:
                continue
            selected.append(i)
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            if len(selected) == top_k:
                break

        return [{
            "text": self.chunks[i]["text"],
            "source": self.chunks[i]["source"],
            "page": self.chunks[i]["page"],
            "project": self.chunks[i].get("project"),
            "score": float(combined[i])
        } for i in selected]


def similarity_search(vectorstore: Chroma, query: str, top_k: int = 4) -> List[Dict]:
    """Kept for backward compatibility / dense-only fallback."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [{
        "text": doc.page_content,
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "score": score
    } for doc, score in results]