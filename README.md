<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenAI-GPT--3.5%20%7C%20Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/Telegram-Bot%20API-26A5E4?style=for-the-badge&logo=telegram&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite-Vector%20Store-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>

<br/><br/>

# telegram-rag-pipeline

### A production-grade Retrieval-Augmented Generation bot on Telegram  
**Zero PyTorch · OpenAI-native · Runs on any machine in under 2 minutes**

<br/>

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Configuration](#-configuration) · [Demo](#-example-interaction) · [Tests](#-tests)

</div>

---

## Overview

`telegram-rag-pipeline` is a fully self-contained conversational AI bot that lets users query a private knowledge base through Telegram using natural language. Instead of relying on an LLM's static training data, every query is grounded in real documents via a **Retrieval-Augmented Generation (RAG)** pipeline backed by OpenAI embeddings and a lightweight SQLite vector store.

Built for portfolio demonstration but architected for production — modular, testable, and deployable via Docker with a single command.

---

## Features

| Capability | Implementation |
|---|---|
| **Semantic document search** | `text-embedding-3-small` → cosine similarity over SQLite BLOB store |
| **Grounded answer generation** | Top-k chunks injected into GPT-3.5-turbo system prompt |
| **Conversation memory** | Last 3 user/assistant turns persisted per user in SQLite |
| **Query caching** | Repeated/similar queries (cosine ≥ 0.95) bypass the LLM entirely |
| **Source transparency** | Every answer cites its source document with a 200-char snippet |
| **Bring your own docs** | Drop any `.md` or `.txt` file into `data/docs/` — auto-indexed on startup |
| **Async bot layer** | Full async I/O with `python-telegram-bot` v20, typing indicators included |

**Bot commands**

```
/ask <question>   →  Query the knowledge base
/summarize        →  Summarise your recent conversation
/help             →  Usage guide
/start            →  Welcome message
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Telegram Client                          │
└─────────────────────────────┬────────────────────────────────────┘
                              │  python-telegram-bot v20 (async)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Bot Handler Layer                         │
│                                                                    │
│   app.py  ──►  bot/handlers.py  ──►  rag/pipeline.py             │
│   (entry)       (commands)           (orchestrator)               │
└─────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
    │  embedder   │  │   chunker    │  │    store     │
    │             │  │              │  │              │
    │ OpenAI API  │  │ Sliding-     │  │ SQLite +     │
    │ embed-3-sm  │  │ window split │  │ NumPy dot    │
    │ 1536-dim    │  │ word overlap │  │ product sim  │
    └──────┬──────┘  └──────────────┘  └──────┬───────┘
           │                                   │
           │         ┌─────────────────────────┤
           │         │       3 tables           │
           │         │  ┌─────────────────┐    │
           │         │  │ chunks          │    │
           │         │  │ query_cache     │    │
           │         │  │ user_history    │    │
           │         │  └─────────────────┘    │
           │         └─────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │  OpenAI     │
    │  Chat API   │
    │  gpt-3.5    │
    └─────────────┘
```

**Indexing flow (startup, runs once)**
```
data/docs/*.md  →  chunker  →  embedder  →  SQLite (chunks table)
```

**Query flow (per /ask)**
```
User query  →  embed  →  cache check  →  top-k retrieval
           →  prompt build (context + history)  →  GPT
           →  persist history + cache  →  reply with source
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Telegram bot token → [@BotFather](https://t.me/BotFather)
- OpenAI API key → [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 1. Clone & install

```bash
git clone https://github.com/RRThivyan/telegram-rag-pipeline.git
cd telegram-rag-pipeline

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **No PyTorch. No Hugging Face.** `pip install` completes in under 30 seconds.

### 2. Configure

Create a `.env` file in the project root and add your keys:

```env
TELEGRAM_BOT_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
```

### 3. Run

```bash
python app.py
```

```
INFO | Initialising RAG pipeline …
INFO | Indexed 42 chunks from 5 document(s).
INFO | Knowledge base ready.
INFO | Bot is polling.
```

### Docker

```bash
docker compose up --build
```

The `data/` directory is volume-mounted — your SQLite DB persists across container restarts.

---

## Project Structure

```
telegram-rag-pipeline/
│
├── app.py                    # Entry point — wires pipeline + Telegram
├── config.py                 # Centralised settings via .env
├── requirements.txt          # 4 dependencies only
├── Dockerfile
├── docker-compose.yml
│
├── bot/
│   └── handlers.py           # Async command handlers (/ask /summarize /help)
│
├── rag/
│   ├── chunker.py            # Word-level sliding-window document splitter
│   ├── embedder.py           # OpenAI text-embedding-3-small wrapper
│   ├── store.py              # SQLite vector store — chunks, cache, history
│   └── pipeline.py           # RAG orchestrator — embed → retrieve → generate
│
├── data/
│   └── docs/                 # Knowledge base — drop .md or .txt files here
│       ├── ai_basics_faq.md
│       ├── llm_guide.md
│       ├── rag_concepts.md
│       ├── mlops_faq.md
│       └── python_ai_tips.md
│
└── tests/
    ├── test_chunker.py       # 9 unit tests
    ├── test_store.py         # 12 unit tests
    └── test_embedder.py      # 5 unit tests (OpenAI mocked)
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required.** From @BotFather |
| `OPENAI_API_KEY` | — | **Required.** From OpenAI dashboard |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Swap to `gpt-4o` for higher quality answers |
| `TOP_K` | `3` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `300` | Words per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping words between consecutive chunks |
| `HISTORY_SIZE` | `3` | Conversation turns injected into each prompt |
| `CACHE_THRESHOLD` | `0.95` | Minimum cosine similarity to trigger cache hit |

---

## Models & Cost

| Component | Model | Cost |
|---|---|---|
| Embeddings | `text-embedding-3-small` | $0.00002 / 1K tokens |
| Generation | `gpt-3.5-turbo` | $0.001 / 1K tokens |

Full document indexing (5 docs) costs **< $0.001** and runs once at startup.  
Each `/ask` query costs approximately **$0.001–0.002** end-to-end.

---

## Example Interaction

```
User   → /ask What is retrieval-augmented generation?

Bot    → 💬 Answer:
         RAG is an AI architecture that combines a retrieval system with a
         generative LLM. At query time, relevant document chunks are fetched
         from a vector store and injected into the prompt, grounding the
         model's response in real documents rather than parametric memory.

         📚 Sources: rag_concepts.md
         📎 rag_concepts.md: "Retrieval-Augmented Generation (RAG) is an AI
         architecture that augments an LLM's answers with relevant information
         retrieved from an external knowledge base at query time…"

─────────────────────────────────────────────────

User   → /ask How does MLOps handle model drift?

Bot    → 💬 Answer:
         MLOps addresses model drift through continuous monitoring of
         prediction distributions and input feature statistics. Alerts fire
         when metrics deviate beyond thresholds, triggering automated
         retraining or rollback pipelines.

         📚 Sources: mlops_faq.md

─────────────────────────────────────────────────

User   → /summarize

Bot    → 📋 Conversation Summary:
         The user explored RAG architecture and MLOps practices around
         drift detection. Both answers were grounded in the knowledge base.
```

---

## Tests

```bash
pip install pytest
pytest
```

```
tests/test_chunker.py   .........    9 passed
tests/test_store.py     ............  12 passed
tests/test_embedder.py  .....         5 passed

26 passed in 1.42s
```

Test coverage includes: chunk sizing and overlap correctness, L2 normalisation, vector similarity ranking, cache hit/miss thresholds, user history scoping, and OpenAI client mocking.

---

## Adding Your Own Knowledge Base

Drop any `.md` or `.txt` files into `data/docs/`. Delete `data/rag.db` to force a full re-index, then restart.

```bash
cp my_company_policy.md data/docs/
rm -f data/rag.db
python app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Bot framework | `python-telegram-bot` v20 |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-3.5-turbo` |
| Vector store | SQLite + NumPy dot product |
| Async runtime | Python `asyncio` |
| Containerisation | Docker + Compose |
| Testing | `pytest` + `unittest.mock` |

---

## Author

**RR Thivyan** — AI/ML Engineer  
[GitHub](https://github.com/RRThivyan) · [LinkedIn](https://linkedin.com/in/thivyan-rr)
