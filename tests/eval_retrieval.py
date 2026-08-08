"""
eval_retrieval.py
8-question retrieval regression eval for the hybrid retriever.

Ground-truth is page presence: a query PASSES if at least one retrieved chunk
(pages up to top_k=6) comes from an expected page. Expected pages were read
from the uploaded PDF (BASWE_15_AI_Engineering_Projects_Guide) per-page map.

Run:  python tests/eval_retrieval.py
Prints per-query PASS/FAIL + retrieved pages, and an overall recall score.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.retriever import load_vectorstore, load_chunks, HybridRetriever
from utils.embeddings import get_embedding_model

# (question, {expected pages})  — pages grounded in the uploaded PDF.
EVAL_SET = [
    ("Which projects use LangGraph?", {15, 16, 17, 56, 57}),
    ("What does the LLM Output Arbitration System build?", {15}),
    ("How do you build a RAG pipeline with hybrid search?", {20, 21, 22, 23}),
    ("Describe the Text-to-SQL interface with guardrails", {27, 28, 29, 30}),
    ("How do you fine-tune with LoRA on a domain dataset?", {35, 36, 37}),
    ("Explain the LLM Gateway rate limiting and fallback routing", {40, 41, 42}),
    ("What is the automated eval dataset generator from logs?", {48, 49, 50}),
    ("Build a multi-modal document processor with OCR", {52, 53, 54, 55}),
]

TOP_K = 6


def main() -> int:
    chunks = load_chunks()
    if not chunks:
        print("No chunks loaded. Process/have a vectorstore first.")
        return 1
    hr = HybridRetriever(load_vectorstore(get_embedding_model()), chunks)

    passed = 0
    print(f"{'QUERY':<62} EXPECTED      RETRIEVED                 RESULT")
    print("-" * 110)
    for question, expected in EVAL_SET:
        retrieved = [r["page"] for r in hr.search(question, top_k=TOP_K)]
        hit = set(retrieved) & expected
        ok = len(hit) > 0
        passed += ok
        exp = " ".join(str(p) for p in sorted(expected))
        ret = " ".join(str(p) for p in retrieved)
        print(f"{question[:60]:<62} {exp:<13} {ret:<24} {'PASS' if ok else 'FAIL'}")

    recall = round(100 * passed / len(EVAL_SET), 1)
    print("\n" + ("-" * 62))
    print(f"Recall: {passed}/{len(EVAL_SET)} ({recall}%)  "
          f"{'ALL PASS' if passed == len(EVAL_SET) else 'SOME FAIL'}")
    return 0 if passed == len(EVAL_SET) else 1


if __name__ == "__main__":
    raise SystemExit(main())