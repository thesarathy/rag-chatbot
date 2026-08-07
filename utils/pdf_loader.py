"""
pdf_loader.py
Handles extraction of text from PDF documents.
Each page is returned as a separate unit with metadata (source filename,
page number, and — where detectable — which named "PROJECT N" section it
belongs to), so downstream stages can trace text back to its origin and
label it correctly in citations.
"""

import re
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict


PROJECT_HEADING_PATTERN = re.compile(r"PROJECT\s+(\d+)\s*\n\s*(.+)")


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a single PDF file, page by page, and tag each page
    with the project section it falls under (if the document uses a
    "PROJECT N <title>" heading convention).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages_data = []
    current_project_label = None  # carries forward until a new heading appears

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            match = PROJECT_HEADING_PATTERN.search(text)
            if match:
                project_num, project_title = match.group(1), match.group(2).strip()
                current_project_label = f"Project {project_num}: {project_title}"

            pages_data.append({
                "text": text,
                "source": path.name,
                "page": i + 1,
                "project": current_project_label  # None until first heading is found
            })

    return pages_data


def load_multiple_pdfs(file_paths: List[str]) -> List[Dict]:
    """
    Extract text from multiple PDFs and combine into a single list.
    """
    all_pages = []
    errors = []

    for file_path in file_paths:
        try:
            pages = load_pdf(file_path)
            all_pages.extend(pages)
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")

    if errors:
        print(f"[pdf_loader] Warning: {len(errors)} file(s) failed to load:")
        for err in errors:
            print(f"  - {err}")

    return all_pages