"""
pdf_loader.py
Handles extraction of text from PDF documents.
Each page is returned as a separate unit with metadata (source filename + page number),
so that later stages (chunking, retrieval, citation) can trace any piece of text
back to its exact origin.
"""

from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a single PDF file, page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dicts, one per page:
        {
            "text": "<page text>",
            "source": "<filename>",
            "page": <page number, 1-indexed>
        }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages_data = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""  # extract_text() can return None on some pages
        text = text.strip()

        if text:  # skip blank pages (e.g. scanned pages with no extractable text)
            pages_data.append({
                "text": text,
                "source": path.name,
                "page": i + 1  # human-friendly, 1-indexed
            })

    return pages_data


def load_multiple_pdfs(file_paths: List[str]) -> List[Dict]:
    """
    Extract text from multiple PDFs and combine into a single list.

    Args:
        file_paths: List of PDF file paths.

    Returns:
        Combined list of page-level dicts across all documents.
    """
    all_pages = []
    errors = []

    for file_path in file_paths:
        try:
            pages = load_pdf(file_path)
            all_pages.extend(pages)
        except Exception as e:
            # Don't let one bad PDF kill the whole batch — collect and report errors
            errors.append(f"{file_path}: {str(e)}")

    if errors:
        print(f"[pdf_loader] Warning: {len(errors)} file(s) failed to load:")
        for err in errors:
            print(f"  - {err}")

    return all_pages