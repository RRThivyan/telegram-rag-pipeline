# Python AI Development — Tips & Best Practices

## Use virtual environments for every project
Always create an isolated environment with `python -m venv venv` or `conda create -n myenv python=3.11`. This prevents dependency conflicts between projects and makes requirements.txt reproducible.

## Pin your dependencies
Use `pip freeze > requirements.txt` to pin exact versions. For production, use `pip-tools` to separate direct dependencies (requirements.in) from the full locked tree (requirements.txt). This ensures deterministic builds.

## Profile before optimising
Before rewriting slow code, profile it: `python -m cProfile -s cumtime script.py`. The bottleneck is rarely where you think. Common hotspots in ML code: data loading (fix with `num_workers` in DataLoader), embedding batching (vectorise, don't loop), and repeated model loading (load once, reuse).

## Batch your embedding calls
Never embed one text at a time in a loop. Sentence-Transformers and OpenAI both accept lists. A batch of 100 embeddings takes nearly the same time as one, because GPU/CPU compute is vectorised. Use: `model.encode(list_of_texts)`.

## Use environment variables for secrets
Never hardcode API keys. Use `python-dotenv` and `.env` files locally. On servers, use environment variables or secret managers (AWS Secrets Manager, HashiCorp Vault). Always add `.env` to `.gitignore`.

## Structured logging beats print statements
Use Python's `logging` module instead of `print`. Set log levels (DEBUG, INFO, WARNING, ERROR) and format logs with timestamps. This is essential in async bot code where multiple requests process concurrently.

## Async Python for bot development
Telegram and Discord bots use async I/O. Use `async def` for handlers and `await` for I/O operations (API calls, DB queries). Blocking calls inside async handlers will stall the event loop — wrap them with `asyncio.run_in_executor` or use async DB libraries.

## Use Pydantic for data validation
Pydantic's BaseModel provides automatic type checking, default values, and clean serialisation. It is used extensively in FastAPI and is great for validating LLM-structured outputs and configuration objects.

## Cache expensive operations
Use `functools.lru_cache` for pure functions (tokenisation, config lookups). Use Redis or SQLite for cross-process caching (embedding cache, query cache). In RAG pipelines, caching semantically similar queries can reduce LLM API costs by 30–50%.
