# src/rag_engine.py
"""
RAG orchestrator: retrieval → prompt building → generation → logging.
"""

import os
import logging
import yaml
from collections import defaultdict
from datetime import datetime

from retriever import retrieve
from model     import generate_answer


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rag_engine")

cfg  = load_config()
ROOT = project_root()

TOP_DOCS   = cfg["rag"]["top_docs"]
RED_FLAGS  = cfg["rag"]["red_flags"]   # dict[str, list[str]]
LOG_DIR    = os.path.join(ROOT, "logs")

SAFETY_TEXT = (
    "⚠️ Я не врач и не ставлю диагноз. Я помогаю ориентироваться по симптомам "
    "и показываю выдержки из клинических протоколов.\n"
    "Если есть боль в груди, сильная одышка, обморок, признаки инсульта "
    "(онемение/слабость одной стороны, нарушение речи или лица) "
    "или состояние резко ухудшается — немедленно звоните в скорую: *103*."
)

SYSTEM_PROMPT = (
    "Ты — МедАссистент, виртуальный помощник медицинского портала МЗ РК.\n"
    "Ты НЕ врач и НЕ ставишь диагноз. Отвечай ТОЛЬКО на русском языке.\n\n"
    "Условия:\n"
    "- Отвечай только на основании предоставленного контекста из клинических протоколов.\n"
    "- Никогда не выдумывай информацию. Если нет точного ответа — честно скажи об этом.\n"
    "- Используй формат: (1) возможные заболевания, (2) рекомендации на дому, "
    "(3) когда обратиться к врачу, (4) напоминание что ты не врач.\n"
    "- Если в контексте нет подходящей информации, напиши: "
    "'Извините, такой информации нет в базе протоколов МЗ РК.'\n"
)

FOLLOWUP_QUESTIONS = [
    "Как давно начались симптомы и усиливаются ли они?",
    "Есть ли температура? Какая максимальная?",
    "Были ли контакты с больными за последние 2 недели?",
    "Есть ли хронические заболевания или принимаете лекарства?",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_interaction(question: str, answer: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "interactions.log"), "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}]\nQ: {question}\nA: {answer}\n\n")


def detect_red_flags(text: str) -> list[str]:
    t = text.lower()
    return [key for key, patterns in RED_FLAGS.items() if any(p in t for p in patterns)]


def build_prompt(symptoms: str, docs: list[dict]) -> str:
    # Group excerpts by disease source for cleaner context
    by_source: dict = defaultdict(list)
    for d in docs:
        by_source[d["source"]].append(d)

    parts = []
    for src, chunks in by_source.items():
        name    = src.replace(".txt", "").replace(".pdf", "").replace(".docx", "")
        best    = max(chunks, key=lambda c: c["score"])
        preview = best["text"][:600].strip()
        parts.append(f"=== {name} ===\n{preview}")

    context = "\n\n".join(parts) if parts else "Подходящие протоколы не найдены."

    # Mistral instruct format
    return (
        "<s>[INST] "
        f"{SYSTEM_PROMPT}\n"
        f"Пользователь описал симптомы: {symptoms}\n\n"
        f"Выдержки из клинических протоколов МЗ РК:\n\n{context}\n\n"
        "Ответ должен быть чётким, по пунктам, на русском языке."
        " [/INST]"
    )


def _fallback_answer(docs: list[dict]) -> str:
    by_source: dict = defaultdict(list)
    for d in docs:
        by_source[d["source"]].append(d)
    names  = [src.replace(".txt", "").replace(".pdf", "").replace(".docx", "") for src in by_source]
    bullet = "\n".join(f"• {n}" for n in names) if names else "• Не определено"
    return (
        "На основе описанных симптомов найдены следующие возможные направления (НЕ диагноз):\n\n"
        f"{bullet}\n\n"
        f"Рекомендую уточнить симптомы и обратиться к врачу для точной оценки.\n\n"
        f"{SAFETY_TEXT}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Full RAG pipeline:
      1. Check for red-flag symptoms → return emergency message immediately.
      2. Retrieve relevant protocol chunks.
      3. Build prompt and generate answer via Ollama.
      4. Log and return answer.
    """
    flags = detect_red_flags(question)
    if flags:
        answer = (
            "⚠️ В описании есть симптомы, при которых нужна неотложная помощь.\n\n"
            f"{SAFETY_TEXT}"
        )
        log_interaction(question, answer)
        return answer

    docs = retrieve(question)
    logger.info(f"Retrieved {len(docs)} docs for: {question!r}")
    if not docs:
        answer = "Извините, такой информации нет в базе протоколов МЗ РК."
        log_interaction(question, answer)
        return answer

    prompt = build_prompt(question, docs[:TOP_DOCS])

    try:
        answer = generate_answer(prompt)
    except Exception:
        logger.exception("Generation failed")
        answer = ""

    if not answer or len(answer.strip()) < 20:
        answer = _fallback_answer(docs)

    log_interaction(question, answer)
    return answer
