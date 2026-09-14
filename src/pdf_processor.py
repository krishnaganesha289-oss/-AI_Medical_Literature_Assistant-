import fitz


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.

    Args:
        pdf_file: Streamlit UploadedFile or file-like object.

    Returns:
        str: Extracted text from the PDF.
    """

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:
        text = page.get_text()

        if text.strip():
            pages.append(text)

    document.close()

    return "\n\n".join(pages)


def get_page_count(pdf_file):
    """
    Return the number of pages in a PDF.
    """

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page_count = len(document)

    document.close()

    return page_count