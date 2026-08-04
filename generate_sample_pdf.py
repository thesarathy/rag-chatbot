# generate_sample_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def make_sample_pdf(path="data/sample.pdf"):
    c = canvas.Canvas(path, pagesize=letter)

    pages_content = [
        "Page 1: Introduction to Retrieval Augmented Generation.\n"
        "RAG combines a retrieval system with a language model to answer "
        "questions using external documents instead of relying solely on "
        "the model's internal knowledge.",

        "Page 2: How Vector Databases Work.\n"
        "Vector databases store embeddings — numerical representations of "
        "text — and allow fast similarity search to find the most relevant "
        "chunks of text for a given query.",

        "Page 3: Chunking Strategy.\n"
        "Documents are split into smaller overlapping chunks before being "
        "embedded, so that each chunk represents a focused, coherent idea "
        "for accurate retrieval."
    ]

    for text in pages_content:
        c.setFont("Helvetica", 12)
        lines = text.split("\n")
        y = 700
        for line in lines:
            c.drawString(50, y, line)
            y -= 20
        c.showPage()

    c.save()
    print(f"Sample PDF created at {path}")

if __name__ == "__main__":
    make_sample_pdf()