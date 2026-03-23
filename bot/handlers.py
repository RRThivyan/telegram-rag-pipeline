"""
bot/handlers.py — Telegram command handlers wired to the RAG pipeline.

Commands:
  /start     — welcome message
  /help      — usage guide
  /ask       — RAG query
  /summarize — summarise recent conversation
  (text)     — fallback nudge
"""
from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── Static texts ─────────────────────────────────────────────────────────────

_WELCOME = (
    "👋 *Welcome to the RAG Knowledge Bot!*\n\n"
    "I can answer questions from a curated knowledge base powered by AI.\n\n"
    "Type /help to see all available commands."
)

_HELP = (
    "🤖 *RAG Knowledge Bot — Help*\n\n"
    "*Commands:*\n"
    "• `/ask <question>` — Query the knowledge base\n"
    "• `/summarize` — Summarise your recent conversation\n"
    "• `/help` — Show this help message\n\n"
    "*Examples:*\n"
    "`/ask What is retrieval-augmented generation?`\n"
    "`/ask How does MLOps improve model reliability?`\n"
    "`/ask What are transformer attention mechanisms?`"
)

_FALLBACK = (
    "💡 Use `/ask <your question>` to query the knowledge base.\n"
    "Type /help for the full command list."
)


class BotHandlers:
    def __init__(self, rag: RAGPipeline) -> None:
        self.rag = rag

    # ── /start ──────────────────────────────────────────────────────────────

    async def start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(_WELCOME, parse_mode=ParseMode.MARKDOWN)

    # ── /help ───────────────────────────────────────────────────────────────

    async def help_cmd(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(_HELP, parse_mode=ParseMode.MARKDOWN)

    # ── /ask ────────────────────────────────────────────────────────────────

    async def ask(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user_id  = update.effective_user.id
        question = " ".join(ctx.args).strip()

        if not question:
            await update.message.reply_text(
                "❓ Please provide a question after `/ask`.\n\n"
                "Example: `/ask What is RAG?`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            result = self.rag.query(question, user_id)
        except Exception as exc:
            logger.error("RAG query error for user %d: %s", user_id, exc, exc_info=True)
            await update.message.reply_text(
                "❌ Something went wrong while querying. Please try again."
            )
            return

        answer     = result["answer"]
        sources    = result.get("sources", "")
        from_cache = result.get("from_cache", False)
        chunks     = result.get("chunks", [])

        lines = [f"💬 *Answer:*\n{answer}"]

        if sources:
            lines.append(f"\n📚 *Sources:* `{sources}`")

        if chunks:
            snippet = chunks[0]["text"][:200].replace("\n", " ") + "…"
            lines.append(f"\n📎 *Snippet from* `{chunks[0]['source']}`:\n_{snippet}_")

        if from_cache:
            lines.append("\n⚡ _(served from cache)_")

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── /summarize ──────────────────────────────────────────────────────────

    async def summarize(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        await update.message.chat.send_action(ChatAction.TYPING)

        try:
            summary = self.rag.summarize_history(user_id)
        except Exception as exc:
            logger.error("Summarize error for user %d: %s", user_id, exc, exc_info=True)
            await update.message.reply_text("❌ Could not generate summary.")
            return

        await update.message.reply_text(
            f"📋 *Conversation Summary:*\n{summary}",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── Fallback ─────────────────────────────────────────────────────────────

    async def fallback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(_FALLBACK, parse_mode=ParseMode.MARKDOWN)
