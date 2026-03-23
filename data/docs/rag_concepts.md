# Retrieval-Augmented Generation (RAG) — Concepts

## What is RAG?
Retrieval-Augmented Generation (RAG) is an AI architecture that augments an LLM's answers with relevant information retrieved from an external knowledge base at query time. Instead of relying solely on the model's parametric memory (knowledge baked in during training), RAG fetches fresh, grounded context and injects it into the prompt before generating a response.

## Why use RAG instead of fine-tuning?
Fine-tuning updates model weights and is expensive to repeat every time knowledge changes. RAG separates knowledge (stored in a vector DB) from the model. Updating the knowledge base is as simple as re-indexing documents. RAG is also more transparent — you can trace which source chunks contributed to the answer.

## What is a vector database?
A vector database stores high-dimensional embedding vectors alongside their original text. It supports fast approximate nearest-neighbour (ANN) search to find the most semantically similar chunks to a query vector. Examples include: Chroma, Pinecone, Weaviate, Qdrant, and FAISS. For lightweight use cases, SQLite with numpy is sufficient.

## What is chunking?
Chunking is the process of splitting long documents into smaller pieces before embedding. Smaller chunks lead to more precise retrieval. Common strategies: fixed-size word/character chunks with overlap, sentence-level splitting, and recursive character splitting (used in LangChain). Overlap between consecutive chunks prevents context from being cut off at boundaries.

## What is an embedding?
An embedding is a dense, real-valued vector representation of text that captures semantic meaning. Similar texts have embeddings that are close in vector space (measured by cosine similarity or dot product). Models like `all-MiniLM-L6-v2` produce 384-dimensional embeddings in ~50 ms on CPU.

## What is the RAG pipeline flow?
1. **Indexing (offline)**: Load documents → chunk → embed → store in vector DB.
2. **Querying (online)**: Receive user question → embed query → retrieve top-k chunks → build prompt (system + context + history + question) → call LLM → return answer.

## What is top-k retrieval?
Top-k retrieval returns the k chunks with the highest cosine similarity to the query vector. Typical values are k=3 to k=5. More chunks provide richer context but increase prompt length and LLM cost. Re-ranking (using a cross-encoder after initial retrieval) can improve precision.

## What is a re-ranker?
A re-ranker is a cross-encoder model that takes a (query, chunk) pair and produces a relevance score. It is more accurate than bi-encoder cosine similarity but slower. Typical RAG pipelines first retrieve top-20 candidates with fast ANN search, then re-rank to select the final top-3.
