"""
tests/test_embedder.py — Unit tests for rag.embedder (OpenAI version)

Uses unittest.mock to patch the OpenAI client so no real API calls are made.
Run:  pytest tests/test_embedder.py -v
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_response(n: int, dim: int = 1536) -> MagicMock:
    """Build a fake openai embeddings response with n vectors."""
    rng  = np.random.default_rng(42)
    resp = MagicMock()
    resp.data = []
    for i in range(n):
        v = rng.random(dim).astype(np.float32)
        item = MagicMock()
        item.embedding = v.tolist()
        resp.data.append(item)
    return resp


@pytest.fixture
def embedder():
    with patch("rag.embedder.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        from rag.embedder import Embedder
        emb = Embedder()
        emb._mock_client = mock_client   # expose for per-test setup
        yield emb


class TestEmbedder:
    def test_embed_returns_correct_shape(self, embedder):
        embedder._mock_client.embeddings.create.return_value = _make_mock_response(3)
        vecs = embedder.embed(["a", "b", "c"])
        assert vecs.shape == (3, 1536)

    def test_embed_one_returns_1d(self, embedder):
        embedder._mock_client.embeddings.create.return_value = _make_mock_response(1)
        vec = embedder.embed_one("test")
        assert vec.ndim == 1
        assert vec.shape[0] == 1536

    def test_embeddings_are_l2_normalised(self, embedder):
        embedder._mock_client.embeddings.create.return_value = _make_mock_response(2)
        vecs  = embedder.embed(["x", "y"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_newlines_replaced_before_api_call(self, embedder):
        embedder._mock_client.embeddings.create.return_value = _make_mock_response(1)
        embedder.embed(["line one\nline two"])
        call_args = embedder._mock_client.embeddings.create.call_args
        sent_input = call_args.kwargs.get("input") or call_args.args[0]
        assert "\n" not in sent_input[0]

    def test_dim_property_is_1536(self, embedder):
        assert embedder.dim == 1536
