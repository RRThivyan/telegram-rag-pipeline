"""
config.py — All settings read from environment / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Bot ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ── OpenAI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ── Storage ─────────────────────────────────────────────────────────────────
DB_PATH: str   = os.getenv("DB_PATH",   "data/rag.db")
DOCS_DIR: str  = os.getenv("DOCS_DIR",  "data/docs")

# ── RAG knobs ───────────────────────────────────────────────────────────────
TOP_K: int         = int(os.getenv("TOP_K",         "3"))
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE",    "300"))   # words per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))    # words overlap

# ── History / cache ─────────────────────────────────────────────────────────
HISTORY_SIZE: int                  = int(os.getenv("HISTORY_SIZE",   "3"))
CACHE_SIMILARITY_THRESHOLD: float  = float(os.getenv("CACHE_THRESHOLD", "0.95"))
