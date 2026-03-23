"""
rag/chunker.py — Load .md / .txt files and split into overlapping word chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def load_documents(docs_dir: str) -> List[dict]:
    """Return [{text, source}, ...] for every .md and .txt file found."""
    docs: List[dict] = []
    for path in sorted(Path(docs_dir).glob("**/*")):
        if path.suffix in (".md", ".txt") and path.is_file():
            docs.append({"text": path.read_text(encoding="utf-8"), "source": path.name})
    return docs


def split_into_chunks(
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Word-level sliding-window chunker.
    chunk_size : target number of words per chunk
    overlap    : words shared between consecutive chunks
    """
    words = text.split()
    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(Chunk(
            text=" ".join(words[start:end]),
            source=source,
            chunk_id=chunk_id,
        ))
        if end == len(words):
            break
        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def load_and_chunk(
    docs_dir: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Chunk]:
    """Convenience: load + chunk every document in docs_dir."""
    all_chunks: List[Chunk] = []
    for doc in load_documents(docs_dir):
        all_chunks.extend(split_into_chunks(doc["text"], doc["source"], chunk_size, overlap))
    return all_chunks
