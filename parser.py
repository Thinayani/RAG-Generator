"""Extract raw text from an uploaded file.

Supports PDF and plain text/markdown. No format-specific assumptions
about document content -- just text extraction.
"""

import io

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Return the full raw text of a document, given its filename and bytes.

    Dispatches on file extension. Raises ValueError for unsupported types.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    elif ext in (".txt", ".md"):
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for '{filename}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = text.strip()
    if not text:
        raise ValueError(f"No extractable text found in '{filename}'.")
    return text
