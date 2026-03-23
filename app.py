#!/usr/bin/env python3
"""
Telegram RAG Bot — Entry Point
Initialises the pipeline, indexes documents, and starts polling.
"""
import logging
from pathlib import Path

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import TELEGRAM_TOKEN, DOCS_DIR
from rag.pipeline import RAGPipeline
from bot.handlers import BotHandlers

logging.basicConfig(
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env")

    # ── Initialise RAG pipeline ────────────────────────────────────────────
    logger.info("Initialising RAG pipeline …")
    rag = RAGPipeline()
    rag.index_documents(DOCS_DIR)
    logger.info("Knowledge base ready.")

    # ── Build Telegram application ─────────────────────────────────────────
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    h = BotHandlers(rag)

    app.add_handler(CommandHandler("start",     h.start))
    app.add_handler(CommandHandler("help",      h.help_cmd))
    app.add_handler(CommandHandler("ask",       h.ask))
    app.add_handler(CommandHandler("summarize", h.summarize))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, h.fallback))

    logger.info("Bot is polling. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
