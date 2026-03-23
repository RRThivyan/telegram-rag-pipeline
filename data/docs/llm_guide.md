# Large Language Models (LLMs) — Guide

## What is a Large Language Model?
A Large Language Model (LLM) is a type of deep learning model trained on massive text corpora to understand and generate human language. LLMs are based on the Transformer architecture and are pre-trained using self-supervised objectives like next-token prediction (autoregressive models: GPT, LLaMA) or masked language modelling (encoder models: BERT).

## How does the Transformer architecture work?
The Transformer uses a mechanism called **self-attention** to weigh the importance of every word in a sequence relative to every other word. This allows it to capture long-range dependencies efficiently. A Transformer block consists of multi-head self-attention followed by a feed-forward network, with residual connections and layer normalisation around each sub-layer.

## What is tokenisation?
Tokenisation is the process of converting raw text into discrete tokens (sub-word units) that the model can process. Modern LLMs use Byte-Pair Encoding (BPE) or SentencePiece. A word like "unhappiness" might be split into ["un", "happiness"] or ["un", "happi", "ness"] depending on the vocabulary.

## What is prompt engineering?
Prompt engineering is the practice of crafting input text (prompts) to guide an LLM towards desired outputs. Techniques include: zero-shot prompting (no examples), few-shot prompting (provide 2–5 examples), chain-of-thought prompting (ask the model to reason step-by-step), and role prompting (assign the model a persona).

## What is fine-tuning?
Fine-tuning adapts a pre-trained LLM to a specific task or domain by continuing training on a smaller, task-specific dataset. This is much cheaper than training from scratch. Techniques include full fine-tuning, LoRA (Low-Rank Adaptation), and QLoRA for memory-efficient tuning.

## What is context length / context window?
The context window is the maximum number of tokens an LLM can process in a single forward pass. GPT-4 Turbo supports 128K tokens; Claude 3 supports up to 200K. A larger context window allows the model to reference more history, documents, or code in one prompt.

## What are hallucinations in LLMs?
LLM hallucinations occur when the model generates text that sounds plausible but is factually incorrect or fabricated. This happens because LLMs learn statistical patterns, not ground truth. Mitigations include: grounding the model with external knowledge (RAG), using system prompts to restrict scope, and applying output verification.

## What is temperature in LLM inference?
Temperature is a hyperparameter that controls the randomness of LLM output. Temperature = 0 makes output nearly deterministic (picks the most likely token). Higher temperature (e.g., 0.8–1.0) increases diversity and creativity. For factual Q&A, temperature 0.0–0.3 is recommended.
