"""
tests/test_store.py — Unit tests for rag.store (VectorStore)

Uses an in-memory / temp SQLite DB — no file I/O side-effects.
Run:  pytest tests/test_store.py -v
"""
import tempfile
import os

import numpy as np
import pytest

# Patch DB_PATH before importing store so the real DB is never touched
import config as _cfg

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setattr(_cfg, "DB_PATH", db)
    # Also patch the module-level default used in VectorStore.__init__
    import rag.store as _store
    monkeypatch.setattr(_store, "DB_PATH", db, raising=False)
    yield db


from rag.store import VectorStore


DIM = 8  # small dimension for fast tests


def _rand_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.random(DIM).astype(np.float32)
    return v / np.linalg.norm(v)          # L2 normalise


# ── Chunk operations ───────────────────────────────────────────────────────────

class TestChunkStorage:
    def test_is_indexed_false_on_empty_db(self):
        store = VectorStore()
        assert store.is_indexed() is False

    def test_add_chunks_sets_is_indexed(self):
        store   = VectorStore()
        vecs    = np.vstack([_rand_vec(i) for i in range(3)])
        store.add_chunks(["a", "b", "c"], ["s1", "s1", "s2"], vecs)
        assert store.is_indexed() is True

    def test_clear_chunks_resets_index(self):
        store = VectorStore()
        vecs  = np.vstack([_rand_vec(0)])
        store.add_chunks(["x"], ["src.md"], vecs)
        store.clear_chunks()
        assert store.is_indexed() is False


# ── Similarity search ──────────────────────────────────────────────────────────

class TestSimilaritySearch:
    def test_returns_top_k_results(self):
        store = VectorStore()
        vecs  = np.vstack([_rand_vec(i) for i in range(10)])
        texts = [f"chunk {i}" for i in range(10)]
        sources = ["doc.md"] * 10
        store.add_chunks(texts, sources, vecs)

        query  = _rand_vec(99)
        results = store.similarity_search(query, top_k=3)
        assert len(results) == 3

    def test_results_have_required_keys(self):
        store = VectorStore()
        vecs  = np.vstack([_rand_vec(0)])
        store.add_chunks(["hello world"], ["test.md"], vecs)
        results = store.similarity_search(_rand_vec(1), top_k=1)
        assert {"id", "source", "text", "score"} <= set(results[0].keys())

    def test_most_similar_vector_ranked_first(self):
        store = VectorStore()
        query = _rand_vec(7)

        # One vector identical to query, others random
        others = np.vstack([_rand_vec(i) for i in range(5)])
        all_vecs = np.vstack([query[np.newaxis, :], others])
        texts  = ["identical"] + [f"other {i}" for i in range(5)]
        store.add_chunks(texts, ["s.md"] * 6, all_vecs)

        results = store.similarity_search(query, top_k=1)
        assert results[0]["text"] == "identical"

    def test_empty_store_returns_empty_list(self):
        store = VectorStore()
        assert store.similarity_search(_rand_vec(0), top_k=3) == []


# ── Query cache ────────────────────────────────────────────────────────────────

class TestQueryCache:
    def test_cache_miss_returns_none(self):
        store = VectorStore()
        result = store.get_cached(_rand_vec(0), threshold=0.95)
        assert result is None

    def test_cache_hit_on_identical_vector(self):
        store = VectorStore()
        vec   = _rand_vec(42)
        store.cache_answer(vec, "cached answer", "doc.md")
        result = store.get_cached(vec, threshold=0.95)
        assert result is not None
        assert result["answer"] == "cached answer"

    def test_cache_miss_below_threshold(self):
        store  = VectorStore()
        vec_a  = _rand_vec(1)
        # Create an orthogonal vector → similarity ~0
        vec_b  = np.zeros(DIM, dtype=np.float32)
        vec_b[0] = 1.0
        store.cache_answer(vec_a, "answer", "doc.md")
        result = store.get_cached(vec_b, threshold=0.95)
        assert result is None


# ── User history ───────────────────────────────────────────────────────────────

class TestUserHistory:
    def test_add_and_retrieve_history(self):
        store = VectorStore()
        store.add_history(1, "user",      "What is RAG?")
        store.add_history(1, "assistant", "RAG stands for …")
        hist = store.get_history(1, n_turns=3)
        assert len(hist) == 2
        assert hist[0]["role"] == "user"
        assert hist[1]["role"] == "assistant"

    def test_history_respects_n_turns_limit(self):
        store = VectorStore()
        for i in range(10):
            store.add_history(1, "user",      f"q{i}")
            store.add_history(1, "assistant", f"a{i}")
        hist = store.get_history(1, n_turns=2)
        assert len(hist) <= 4        # 2 turns × 2 messages

    def test_history_is_user_scoped(self):
        store = VectorStore()
        store.add_history(1, "user", "user 1 message")
        store.add_history(2, "user", "user 2 message")
        hist_1 = store.get_history(1, n_turns=5)
        hist_2 = store.get_history(2, n_turns=5)
        assert all(m["content"] == "user 1 message" for m in hist_1)
        assert all(m["content"] == "user 2 message" for m in hist_2)

    def test_clear_history_removes_entries(self):
        store = VectorStore()
        store.add_history(1, "user", "hello")
        store.clear_history(1)
        assert store.get_history(1) == []
