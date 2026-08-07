"""
retriever.py
Hybrid retrieval: ChromaDB dense (semantic) search + BM25 sparse (keyword) search,
fused together, with a diversity cap so answers to enumeration-style questions
("which projects use X") aren't dominated by one document/section that happens
to have high term-frequency for the query.
"""

from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np

PERSIST_DIR = "vectorstore"


def build_vectorstore(chunks: List[Dict], embedding_model) -> Chroma:
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    vectorstore = Chroma.from_texts(
        texts=texts, embedding=embedding_model, metadatas=metadatas,
        persist_directory=PERSIST_DIR
    )
    return vectorstore


def load_vectorstore(embedding_model) -> Chroma:
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)


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
        self.chunks = chunks  # same chunks used to build the vectorstore, in the same order
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 4, alpha: float = 0.5,
               candidate_pool: int = 20, max_per_cluster: int = 2) -> List[Dict]:
        """
        alpha: weight for dense score (0-1). 0.5 = equal weight to dense + keyword.
        candidate_pool: how many top-scoring chunks to consider before diversity filtering.
        max_per_cluster: max chunks allowed from the same ~5-page neighborhood in the
                          final result, so a single densely-covered section can't crowd
                          out other relevant sections.
        """
        # Dense (semantic) results — over-fetch so fusion has enough candidates
        dense_results = self.vectorstore.similarity_search_with_score(query, k=len(self.chunks))
        dense_scores = np.zeros(len(self.chunks))
        text_to_idx = {c["text"]: i for i, c in enumerate(self.chunks)}
        for doc, score in dense_results:
            idx = text_to_idx.get(doc.page_content)
            if idx is not None:
                # Chroma's default distance: lower = more similar -> invert for scoring
                dense_scores[idx] = 1.0 / (1.0 + score)

        # Sparse (keyword) results
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))

        def normalize(arr):
            if arr.max() - arr.min() < 1e-9:
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())

        combined = alpha * normalize(dense_scores) + (1 - alpha) * normalize(bm25_scores)

        # Over-fetch a larger candidate pool, ranked by fused score
        ranked_idx = np.argsort(-combined)[:candidate_pool]

        # Greedy diversity selection: skip a candidate if its "cluster"
        # (a ~5-page neighborhood, used as a cheap proxy for "which project/
        # section this belongs to") already contributed max_per_cluster chunks
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