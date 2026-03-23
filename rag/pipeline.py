"""
rag/pipeline.py — Orchestrates chunking → embedding → retrieval → generation.

Flow for each /ask query:
  1. Embed user query        (OpenAI text-embedding-3-small)
  2. Check query cache       (cosine threshold via SQLite)
  3. Retrieve top-k chunks   (NumPy dot-product on stored blobs)
  4. Build context window    (retrieved chunks + conversation history)
  5. Call OpenAI Chat        (gpt-3.5-turbo)
  6. Persist history + cache, return structured result
"""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from config import (
    CACHE_SIMILARITY_THRESHOLD,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HISTORY_SIZE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)
from rag.chunker import load_and_chunk
from rag.embedder import Embedder
from rag.store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompt template ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a knowledgeable assistant. Answer the user's question using ONLY the
information in the Context section below.

Rules:
- If the answer is not in the context, respond: "I don't have information about \
that in my knowledge base."
- Be concise and direct (2-4 sentences unless detail is requested).
- Never make up facts not present in the context.

Context:
{context}
"""


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store    = VectorStore()
        self.client   = OpenAI(api_key=OPENAI_API_KEY)

    # ── Indexing ────────────────────────────────────────────────────────────

    def index_documents(self, docs_dir: str, force: bool = False) -> None:
        """Chunk, embed, and persist all documents. Skips if already indexed."""
        if self.store.is_indexed() and not force:
            logger.info("Knowledge base already indexed — skipping.")
            return

        chunks = load_and_chunk(docs_dir, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            logger.warning("No documents found in '%s'.", docs_dir)
            return

        texts   = [c.text   for c in chunks]
        sources = [c.source for c in chunks]
        vectors = self.embedder.embed(texts)

        self.store.clear_chunks()
        self.store.add_chunks(texts, sources, vectors)
        logger.info(
            "Indexed %d chunks from %d document(s).",
            len(chunks),
            len(set(sources)),
        )

    # ── Query ───────────────────────────────────────────────────────────────

    def query(self, question: str, user_id: int) -> dict:
        """
        Execute a RAG query and return a result dict:
          {answer, sources, chunks, from_cache}
        """
        query_vec = self.embedder.embed_one(question)

        # ── 1. Cache lookup ────────────────────────────────────────────────
        cached = self.store.get_cached(query_vec, CACHE_SIMILARITY_THRESHOLD)
        if cached:
            self.store.add_history(user_id, "user",      question)
            self.store.add_history(user_id, "assistant", cached["answer"])
            return {**cached, "from_cache": True, "chunks": []}

        # ── 2. Retrieve ────────────────────────────────────────────────────
        hits = self.store.similarity_search(query_vec, TOP_K)
        if not hits:
            return {
                "answer":     "The knowledge base is empty. Please add documents first.",
                "sources":    "",
                "chunks":     [],
                "from_cache": False,
            }

        context = "\n\n---\n\n".join(
            f"[Source: {h['source']}]\n{h['text']}" for h in hits
        )
        unique_sources: List[str] = list(dict.fromkeys(h["source"] for h in hits))

        # ── 3. Build messages (system + history + current question) ────────
        history  = self.store.get_history(user_id, HISTORY_SIZE)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT.format(context=context)}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        # ── 4. Generate ────────────────────────────────────────────────────
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

        # ── 5. Persist ─────────────────────────────────────────────────────
        self.store.add_history(user_id, "user",      question)
        self.store.add_history(user_id, "assistant", answer)
        self.store.cache_answer(query_vec, answer, ", ".join(unique_sources))

        return {
            "answer":     answer,
            "sources":    ", ".join(unique_sources),
            "chunks":     hits,
            "from_cache": False,
        }

    # ── Summarise ───────────────────────────────────────────────────────────

    def summarize_history(self, user_id: int) -> str:
        """Return a one-paragraph summary of the user's recent conversation."""
        history = self.store.get_history(user_id, HISTORY_SIZE)
        if not history:
            return "No recent conversation found."

        dialogue = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history
        )
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise the following conversation in 2-3 sentences.",
                },
                {"role": "user", "content": dialogue},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
