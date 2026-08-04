# test_embeddings.py
from utils.pdf_loader import load_pdf
from utils.embeddings import chunk_pages, get_embedding_model

pages = load_pdf("data/sample.pdf")
chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)
print(f"{len(pages)} pages -> {len(chunks)} chunks")

model = get_embedding_model()
sample_vector = model.embed_query(chunks[0]["text"])
print(f"Embedding dimension: {len(sample_vector)}")
print(sample_vector[:5], "...")