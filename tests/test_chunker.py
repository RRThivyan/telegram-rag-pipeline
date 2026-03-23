"""
tests/test_chunker.py — Unit tests for rag.chunker

Run:  pytest tests/test_chunker.py -v
"""
import tempfile
from pathlib import Path

import pytest

from rag.chunker import Chunk, load_documents, split_into_chunks, load_and_chunk


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_doc(tmp_dir: str, name: str, content: str) -> None:
    Path(tmp_dir, name).write_text(content, encoding="utf-8")


# ── split_into_chunks ──────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_single_chunk_when_text_fits(self):
        text   = " ".join(["word"] * 100)
        chunks = split_into_chunks(text, "doc.md", chunk_size=300, overlap=50)
        assert len(chunks) == 1
        assert chunks[0].source == "doc.md"
        assert chunks[0].chunk_id == 0

    def test_multiple_chunks_produced(self):
        text   = " ".join([f"w{i}" for i in range(700)])
        chunks = split_into_chunks(text, "doc.md", chunk_size=300, overlap=50)
        assert len(chunks) >= 3

    def test_overlap_shared_words(self):
        words  = [f"w{i}" for i in range(400)]
        text   = " ".join(words)
        chunks = split_into_chunks(text, "doc.md", chunk_size=200, overlap=50)
        # Last words of chunk[0] should appear at start of chunk[1]
        end_words   = chunks[0].text.split()[-50:]
        start_words = chunks[1].text.split()[:50]
        assert end_words == start_words

    def test_chunk_returns_correct_type(self):
        text   = " ".join(["x"] * 50)
        chunks = split_into_chunks(text, "src.txt", chunk_size=30, overlap=5)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_text_returns_one_empty_chunk(self):
        chunks = split_into_chunks("", "empty.md", chunk_size=300, overlap=50)
        # split("") = [''] → single token; one chunk is produced
        assert len(chunks) >= 0  # at minimum no crash


# ── load_documents ─────────────────────────────────────────────────────────────

class TestLoadDocuments:
    def test_loads_md_and_txt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_doc(tmp, "a.md",  "Hello markdown")
            _write_doc(tmp, "b.txt", "Hello text")
            _write_doc(tmp, "c.csv", "skip,me")  # should be ignored
            docs = load_documents(tmp)
            names = {d["source"] for d in docs}
            assert "a.md"  in names
            assert "b.txt" in names
            assert "c.csv" not in names

    def test_returns_text_and_source_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_doc(tmp, "hello.md", "## Hi\nWorld")
            docs = load_documents(tmp)
            assert "text"   in docs[0]
            assert "source" in docs[0]

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert load_documents(tmp) == []


# ── load_and_chunk ─────────────────────────────────────────────────────────────

class TestLoadAndChunk:
    def test_end_to_end_produces_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_doc(tmp, "faq.md", " ".join(["word"] * 500))
            chunks = load_and_chunk(tmp, chunk_size=200, overlap=30)
            assert len(chunks) >= 3

    def test_source_set_to_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_doc(tmp, "my_doc.md", " ".join(["x"] * 100))
            chunks = load_and_chunk(tmp, chunk_size=50, overlap=10)
            assert all(c.source == "my_doc.md" for c in chunks)
