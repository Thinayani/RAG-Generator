"""Split raw text into overlapping fixed-size chunks."""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Return a list of text chunks.

    Simple character-based splitting with overlap -- no sentence/semantic
    awareness needed at this scope. Overlap keeps context from being cut
    mid-thought at chunk boundaries.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
