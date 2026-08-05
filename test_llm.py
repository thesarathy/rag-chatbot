from utils.llm import ask_llm

fake_chunks = [
    {"text": "RAG combines retrieval with generation to answer questions using external documents.", "source": "sample.pdf", "page": 1}
]

answer = ask_llm("What is RAG?", fake_chunks)
print(answer)