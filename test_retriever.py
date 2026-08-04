# test_retriever.py
from utils.pdf_loader import load_pdf
from utils.embeddings import chunk_pages, get_embedding_model
from utils.retriever import build_vectorstore, similarity_search

pages = load_pdf("data/sample.pdf")
chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)

model = get_embedding_model()
vectorstore = build_vectorstore(chunks, model)

results = similarity_search(vectorstore, "What is RAG?", top_k=2)
for r in results:
    print(f"[{r['source']} - page {r['page']}] (score: {r['score']:.4f})")
    print(r["text"][:150], "...\n")