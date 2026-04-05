# src/telegram_bot.py
"""
Telegram bot interface for the Medical RAG assistant.
Commands:
  /start    — greeting and current model info
  /setmodel — hot-swap the Ollama model
  /help     — usage instructions
Any other text → answer_question() via RAG pipeline.
"""

import os
import asyncio
import logging

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject

from rag_engine import answer_question, FOLLOWUP_QUESTIONS, SAFETY_TEXT
from model     import generate_answer, set_model, CURRENT_MODEL, ALLOWED_MODELS


# ── Config ────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set in .env")

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bot")

bot = Bot(token=TOKEN)
dp  = Dispatcher()


# ── Handlers ──────────────────────────────────────────────────────────────────

@dp.message(Command("start"))
async def cmd_start(msg: types.Message):
    models_list = ", ".join(f"`{m}`" for m in ALLOWED_MODELS)
    text = (
        "👋 Здравствуйте! Я *МедАссистент* — помощник по клиническим протоколам МЗ РК.\n\n"
        "Опишите симптомы, и я найду информацию из протоколов и дам общие рекомендации.\n\n"
        "⚠️ Я *не врач* и не ставлю диагнозы. При опасных симптомах немедленно звоните *103*.\n\n"
        f"Текущая модель: *{CURRENT_MODEL}*\n"
        f"Сменить модель: /setmodel `<{models_list}>`"
    )
    await msg.answer(text, parse_mode="Markdown")


@dp.message(Command("help"))
async def cmd_help(msg: types.Message):
    text = (
        "*Как пользоваться:*\n"
        "Просто напишите симптомы — например:\n"
        "_«Температура 38, кашель и слабость уже 3 дня»_\n\n"
        "*Команды:*\n"
        "/start — приветствие\n"
        "/help — эта справка\n"
        f"/setmodel — сменить модель ({', '.join(ALLOWED_MODELS)})\n\n"
        "*Уточняющие вопросы для лучшего ответа:*\n"
        + "\n".join(f"• {q}" for q in FOLLOWUP_QUESTIONS)
    )
    await msg.answer(text, parse_mode="Markdown")


@dp.message(Command("setmodel"))
async def cmd_setmodel(msg: types.Message, command: CommandObject):
    choice = (command.args or "").strip().lower()
    if not choice:
        await msg.answer(
            f"❗ Укажите модель. Варианты: {', '.join(ALLOWED_MODELS)}\n"
            f"Пример: `/setmodel mistral`",
            parse_mode="Markdown",
        )
        return
    try:
        set_model(choice)
        await msg.answer(f"✅ Модель сменена на *{choice}*.", parse_mode="Markdown")
    except ValueError as e:
        await msg.answer(f"❗ Ошибка: {e}")


@dp.message()
async def handle_question(msg: types.Message):
    question = (msg.text or "").strip()
    if not question:
        return

    if len(question) > 1500:
        await msg.answer(
            "❗ Сообщение слишком длинное. Опишите симптомы кратко (до 1500 символов)."
        )
        return

    logger.info(f"Question from {msg.from_user.id}: {question!r}")

    loop   = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, answer_question, question)

    if len(answer) > 4000:
        answer = answer[:3997] + "…"

    await msg.answer(answer, parse_mode="Markdown")


# ── Startup ───────────────────────────────────────────────────────────────────

async def main():
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, generate_answer, " ")
        logger.info("Ollama model warmed up")
    except Exception:
        logger.exception("Model warmup failed; first request may be slow")

    logger.info("Starting bot polling …")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())