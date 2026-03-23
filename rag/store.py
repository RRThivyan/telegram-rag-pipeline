"""
rag/store.py — SQLite-backed vector store.

Three tables:
  chunks        — raw text + source + binary embedding blob
  query_cache   — cached (query_vec -> answer) pairs for fast repeated queries
  user_history  — per-user conversation turns (last N kept)

Cosine similarity is computed in Python using NumPy dot products.
(Vectors are L2-normalised at embed time so cosine == dot product.)
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import DB_PATH, TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_tables()

    # ── Schema ─────────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                source  TEXT    NOT NULL,
                text    TEXT    NOT NULL,
                vector  BLOB    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash  TEXT PRIMARY KEY,
                query_vec   BLOB NOT NULL,
                answer      TEXT NOT NULL,
                sources     TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── Chunk management ───────────────────────────────────────────────────

    def is_indexed(self) -> bool:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0

    def clear_chunks(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def add_chunks(
        self,
        texts: List[str],
        sources: List[str],
        vectors: np.ndarray,
    ) -> None:
        """Batch-insert chunks. vectors shape: (N, dim), dtype float32."""
        rows = [
            (src, txt, vectors[i].astype(np.float32).tobytes())
            for i, (txt, src) in enumerate(zip(texts, sources))
        ]
        self.conn.executemany(
            "INSERT INTO chunks (source, text, vector) VALUES (?,?,?)", rows
        )
        self.conn.commit()
        logger.info("Inserted %d chunks into store.", len(rows))

    def _load_all_chunks(self) -> List[tuple]:
        """Return [(id, source, text, vec_array), ...]"""
        cur = self.conn.execute("SELECT id, source, text, vector FROM chunks")
        return [
            (r[0], r[1], r[2], np.frombuffer(r[3], dtype=np.float32))
            for r in cur.fetchall()
        ]

    def similarity_search(
        self,
        query_vec: np.ndarray,
        top_k: int = TOP_K,
    ) -> List[dict]:
        """
        Return the top-k most similar chunks to query_vec.
        Each result: {id, source, text, score}
        """
        chunks = self._load_all_chunks()
        if not chunks:
            return []

        matrix = np.vstack([c[3] for c in chunks])   # (N, dim)
        scores  = matrix @ query_vec                  # (N,)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id":     chunks[i][0],
                "source": chunks[i][1],
                "text":   chunks[i][2],
                "score":  float(scores[i]),
            }
            for i in top_idx
        ]

    # ── Query cache ────────────────────────────────────────────────────────

    def get_cached(
        self,
        query_vec: np.ndarray,
        threshold: float = 0.95,
    ) -> Optional[dict]:
        cur = self.conn.execute("SELECT query_vec, answer, sources FROM query_cache")
        for row in cur.fetchall():
            cached_vec = np.frombuffer(row[0], dtype=np.float32)
            sim = float(cached_vec @ query_vec)
            if sim >= threshold:
                logger.debug("Cache hit (sim=%.4f)", sim)
                return {"answer": row[1], "sources": row[2]}
        return None

    def cache_answer(
        self,
        query_vec: np.ndarray,
        answer: str,
        sources: str,
    ) -> None:
        h = hashlib.md5(query_vec.tobytes()).hexdigest()
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache (query_hash, query_vec, answer, sources) VALUES (?,?,?,?)",
            (h, query_vec.astype(np.float32).tobytes(), answer, sources),
        )
        self.conn.commit()

    # ── User history ───────────────────────────────────────────────────────

    def add_history(self, user_id: int, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO user_history (user_id, role, content) VALUES (?,?,?)",
            (user_id, role, content),
        )
        self.conn.commit()

    def get_history(self, user_id: int, n_turns: int = 3) -> List[dict]:
        cur = self.conn.execute(
            """SELECT role, content
               FROM user_history
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, n_turns * 2),
        )
        rows = cur.fetchall()[::-1]
        return [{"role": r[0], "content": r[1]} for r in rows]

    def clear_history(self, user_id: int) -> None:
        self.conn.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
        self.conn.commit()
