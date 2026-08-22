"""
Document ingestion — chunking with overlap + metadata, not naive
fixed-size splitting with no context.

Each chunk keeps:
    - source filename
    - chunk index (for citation / "source tracking")
    - char span (for potential highlighting later)
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int

    @property
    def chunk_id(self) -> str:
        return f"{self.source}::chunk_{self.chunk_index}"


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """
    Sliding-window chunking with overlap so context isn't severed at
    chunk boundaries. Tries to break on sentence/paragraph boundaries
    near the target size rather than mid-word, when possible.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    text = text.strip()
    if not text:
        return chunks

    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break on a sentence boundary within the last 20% of the window
        if end < text_len:
            search_window_start = start + int(chunk_size * 0.8)
            boundary = text.rfind(". ", search_window_start, end)
            if boundary != -1:
                end = boundary + 1  # include the period

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(Chunk(
                text=chunk_str,
                source=source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
            ))
            chunk_index += 1

        if end >= text_len:
            break
        start = end - overlap  # step forward with overlap

    return chunks


def ingest_file(file_path: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Read a plain-text file and chunk it. PDF/docx ingestion can be added
    here later by branching on file extension — kept simple/real for now
    rather than pretending to support formats untested in this build."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    source = file_path.split("/")[-1]
    return chunk_text(text, source, chunk_size, overlap)
