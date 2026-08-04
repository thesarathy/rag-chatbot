from utils.pdf_loader import load_pdf

pages = load_pdf("data/sample.pdf")
print(len(pages), "pages extracted")
print(pages[0])