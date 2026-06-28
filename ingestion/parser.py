import io
from loguru import logger


def parse_document(contents: bytes, suffix: str) -> str:
    """Parse raw file bytes into plain text based on file type."""
    if suffix == ".pdf":
        return _parse_pdf(contents)
    elif suffix == ".docx":
        return _parse_docx(contents)
    elif suffix in (".txt", ".md"):
        return contents.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(contents: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(contents))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    full_text = "\n\n".join(pages)
    logger.debug(f"Parsed PDF: {len(reader.pages)} pages, {len(full_text)} chars")
    return full_text


def _parse_docx(contents: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(contents))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    logger.debug(f"Parsed DOCX: {len(paragraphs)} paragraphs, {len(full_text)} chars")
    return full_text