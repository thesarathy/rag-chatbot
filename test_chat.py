# test_chat.py
from utils.pdf_loader import load_pdf
from utils.embeddings import chunk_pages, get_embedding_model
from utils.retriever import build_vectorstore
from utils.chat import ChatSession

pages = load_pdf("data/sample.pdf")
chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)
model = get_embedding_model()
vectorstore = build_vectorstore(chunks, model)

session = ChatSession(vectorstore, top_k=2)

result1 = session.ask("What is RAG?")
print("Q1 answer:", result1["answer"])

result2 = session.ask("Why does that matter for chatbots?")
print("Q2 standalone question:", result2["standalone_question"])
print("Q2 answer:", result2["answer"])