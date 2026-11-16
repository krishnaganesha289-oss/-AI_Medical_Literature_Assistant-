from __future__ import annotations

from typing import Any, Dict, List

import fitz


def read_pdf_bytes(pdf_file: Any) -> bytes:
    """Read bytes from a PDF-like object and reset the pointer for reuse."""
    if hasattr(pdf_file, "read"):
        pdf_bytes = pdf_file.read()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        return pdf_bytes
    if isinstance(pdf_file, (bytes, bytearray)):
        return bytes(pdf_file)
    raise TypeError("Unsupported PDF input. Pass a file-like object or bytes.")


def extract_pdf_document(pdf_file: Any) -> Dict[str, Any]:
    """Return extracted document text, page count, and scan indicators."""
    pdf_bytes = read_pdf_bytes(pdf_file)
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover - depends on external PDF format
        raise ValueError(f"The PDF could not be opened: {exc}") from exc

    if document.is_encrypted:
        try:
            document.authenticate("")
        except Exception as exc:  # pragma: no cover
            raise ValueError("The PDF is encrypted and cannot be read without a password.") from exc

    page_count = len(document)
    page_texts: List[str] = []
    page_numbers: List[int] = []
    scanned_pages = 0

    for page_index in range(page_count):
        page = document[page_index]
        text = page.get_text("text").strip()
        if text:
            page_texts.append(text)
            page_numbers.append(page_index + 1)
        else:
            scanned_pages += 1

    document.close()

    extracted_text = "\n\n".join(page_texts)
    return {
        "text": extracted_text,
        "page_count": page_count,
        "page_texts": page_texts,
        "page_numbers": page_numbers,
        "scanned_pages": scanned_pages,
        "is_scanned": scanned_pages > 0 and not page_texts,
    }


def extract_text_from_pdf(pdf_file: Any) -> str:
    """Compatibility wrapper that returns paragraph text only."""
    return extract_pdf_document(pdf_file)["text"]


def get_page_count(pdf_file: Any) -> int:
    """Return the number of pages in the PDF."""
    pdf_bytes = read_pdf_bytes(pdf_file)
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"The PDF could not be opened: {exc}") from exc
    page_count = len(document)
    document.close()
    return page_count