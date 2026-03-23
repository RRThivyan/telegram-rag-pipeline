# 🤖 Telegram RAG Knowledge Bot (OpenAI Edition)

A lightweight, production-grade Telegram bot that answers questions from a local
knowledge base using **Retrieval-Augmented Generation (RAG)** — powered entirely
by OpenAI. No PyTorch, no Hugging Face, no local model downloads.

---

## ✨ Features

| Feature | Details |
|---|---|
| `/ask <question>` | Embed query → retrieve top-k chunks → generate answer via GPT |
| `/summarize` | Summarise your last 3 conversation turns |
| `/help` | Usage guide |
| `/start` | Welcome message |
| Message history | Last 3 user+assistant turns injected into every prompt |
| Query caching | Semantically similar queries (cosine ≥ 0.95) served from SQLite cache |
| Source snippets | Every answer shows which document it came from + a 200-char preview |
| Typing indicator | Bot shows "typing…" while processing |

---

## 🏗 System Architecture

```
User (Telegram)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                Telegram Bot Layer                    │
│  python-telegram-bot v20 (async)                    │
│  Handlers: /ask  /summarize  /help  /start          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                   RAG Pipeline                       │
│                                                      │
│  1. embed_one(query)  ← OpenAI text-embedding-3-small│
│  2. cache lookup      ← SQLite query_cache table    │
│  3. similarity_search ← NumPy dot-product on blobs  │
│  4. build_prompt      ← system + history + context  │
│  5. OpenAI ChatCompletion  ← gpt-3.5-turbo          │
│  6. persist history + cache                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                SQLite (data/rag.db)                  │
│  chunks        — text + source + vector BLOB        │
│  query_cache   — query_vec → answer                 │
│  user_history  — per-user conversation turns        │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Windows / Mac / Linux)

### 1. Prerequisites

- Python 3.10+
- A Telegram bot token — get one from [@BotFather](https://t.me/BotFather)
- An OpenAI API key — from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 2. Install dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# Install (fast — no torch/huggingface)
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in your two keys:

```env
TELEGRAM_BOT_TOKEN=7123456789:AAFyourtokenhere
OPENAI_API_KEY=sk-proj-yourkeyhere
```

### 4. Run

```bash
python app.py
```

You'll see:
```
INFO | Initialising RAG pipeline ...
INFO | Indexed 42 chunks from 5 document(s).
INFO | Knowledge base ready.
INFO | Bot is polling. Press Ctrl+C to stop.
```

Open Telegram, search for your bot, and try:
```
/help
/ask What is retrieval-augmented generation?
/ask How does gradient descent work?
/ask What is model drift in MLOps?
/summarize
```

---

## 🐳 Docker (optional)

```bash
cp .env.example .env    # fill in tokens
docker compose up --build
```

The `data/` folder is mounted as a volume — the SQLite DB persists across restarts.

---

## 🧠 Models Used

| Component | Model | Why |
|---|---|---|
| Embeddings | `text-embedding-3-small` | 1536-dim, fast, $0.00002/1K tokens, better than local MiniLM |
| Generation | `gpt-3.5-turbo` | Fast, cheap, reliable. Swap for `gpt-4o` for higher quality |
| Vector DB | SQLite + NumPy | Zero extra infrastructure. Cosine sim on normalised vecs = dot product |
| Bot | `python-telegram-bot` v20 | Native async, clean handler API |

### 💰 Cost estimate

Your 5 sample docs index once at startup — roughly **$0.0001 total** (a fraction of a cent).
Each `/ask` query costs ~$0.001 (embedding + GPT response combined).

---

## 📁 Project Structure

```
rag-bot-v2/
├── app.py                   # Entry point
├── config.py                # All settings from .env
├── requirements.txt         # 4 dependencies only (no torch!)
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── bot/
│   ├── __init__.py
│   └── handlers.py          # /ask /summarize /help /start handlers
│
├── rag/
│   ├── __init__.py
│   ├── chunker.py           # Word-level sliding-window chunker
│   ├── embedder.py          # OpenAI text-embedding-3-small wrapper ← UPDATED
│   ├── store.py             # SQLite vector store (3 tables)
│   └── pipeline.py          # Orchestrator: embed → retrieve → generate
│
├── data/
│   └── docs/                # Your knowledge base (.md / .txt files)
│       ├── ai_basics_faq.md
│       ├── llm_guide.md
│       ├── rag_concepts.md
│       ├── mlops_faq.md
│       └── python_ai_tips.md
│
└── tests/
    ├── conftest.py
    ├── test_chunker.py      # 9 tests
    ├── test_store.py        # 12 tests
    └── test_embedder.py     # 5 tests (mocked, no real API calls)
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required** |
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Swap to `gpt-4o` for better answers |
| `TOP_K` | `3` | Chunks retrieved per query |
| `CHUNK_SIZE` | `300` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `HISTORY_SIZE` | `3` | Conversation turns injected into prompt |
| `CACHE_THRESHOLD` | `0.95` | Cosine similarity for cache hit |

---

## ➕ Adding Your Own Documents

Drop any `.md` or `.txt` file into `data/docs/` and restart the bot.
It re-indexes automatically on startup. Delete `data/rag.db` first if you want
to force a full re-index.

---

## 🧪 Running Tests

```bash
pip install pytest
pytest
```

Expected output:
```
tests/test_chunker.py  .........   9 passed
tests/test_store.py    ............  12 passed
tests/test_embedder.py .....        5 passed
```

---

## 📝 Example Interaction

```
User:   /ask What is retrieval-augmented generation?

Bot:    💬 Answer:
        RAG is an AI architecture that combines a retrieval system with a
        generative LLM. At query time, relevant document chunks are fetched
        from a vector store and injected into the prompt, grounding the
        model's response in real documents rather than parametric memory.

        📚 Sources: rag_concepts.md
        📎 Snippet from rag_concepts.md:
        _Retrieval-Augmented Generation (RAG) is an AI architecture that
        augments an LLM's answers with relevant information retrieved from
        an external knowledge base at query time…_

User:   /ask How does MLOps help with model drift?

Bot:    💬 Answer:
        MLOps addresses model drift through continuous monitoring of
        prediction distributions and input feature statistics. When metrics
        deviate beyond thresholds, automated alerts trigger retraining
        or rollback pipelines.

        📚 Sources: mlops_faq.md

User:   /summarize

Bot:    📋 Conversation Summary:
        The user asked about RAG architecture and MLOps practices around
        model drift detection. Both topics were answered using the
        knowledge base documents.
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `TELEGRAM_BOT_TOKEN is not set` | Check your `.env` file exists and has the token |
| `openai.AuthenticationError` | Your OpenAI API key is wrong or has no credits |
| Bot doesn't respond | Make sure `python app.py` is still running |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the venv |
| Want better answers | Change `OPENAI_MODEL=gpt-4o` in `.env` |
