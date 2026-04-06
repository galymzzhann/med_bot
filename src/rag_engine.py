# src/rag_engine.py
"""
RAG orchestrator with intent-aware prompting.

Handles three types of queries:
  1. Symptom descriptions → suggests possible diseases from protocols
  2. Treatment questions  → provides treatment info from protocols
  3. Disease info queries → explains the disease from protocols
"""

import os
import logging
from collections import defaultdict
from datetime import datetime

import yaml

from retriever import retrieve, detect_intent
from model     import generate_answer


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rag_engine")

cfg  = _load_config()
ROOT = _project_root()

TOP_DOCS   = cfg["rag"]["top_docs"]
RED_FLAGS  = cfg["rag"]["red_flags"]
LOG_DIR    = os.path.join(ROOT, cfg["data"]["logs_dir"])

SECTION_LABELS = {
    "definition":     "Определение",
    "etiology":       "Этиология и причины",
    "symptoms":       "Симптомы и клиническая картина",
    "diagnostics":    "Диагностика",
    "treatment":      "Лечение",
    "prevention":     "Профилактика и прогноз",
    "classification": "Классификация",
    "general":        "Общая информация",
}

SAFETY_TEXT = (
    "⚠️ Я не врач и не ставлю диагноз. Я помогаю ориентироваться по симптомам "
    "и показываю выдержки из клинических протоколов.\n"
    "Если есть боль в груди, сильная одышка, обморок, признаки инсульта "
    "(онемение/слабость одной стороны, нарушение речи) "
    "или состояние резко ухудшается — немедленно звоните в скорую: *103*."
)
# Different system prompts depending on what the user is asking
SYSTEM_PROMPTS = {
    "symptoms": (
        "Ты — МедАссистент, виртуальный помощник по клиническим протоколам МЗ РК.\n"
        "Ты НЕ врач и НЕ ставишь диагноз. Отвечай ТОЛЬКО на русском языке.\n\n"
        "СТРОГИЕ ПРАВИЛА:\n"
        "1. Отвечай ИСКЛЮЧИТЕЛЬНО на основании предоставленного контекста.\n"
        "2. НИКОГДА не выдумывай информацию, лекарства, дозировки или диагнозы.\n"
        "3. Если в контексте нет ответа — честно скажи.\n"
        "4. Указывай из какого протокола (заболевания) взята информация.\n\n"
        "Формат ответа:\n"
        "1) Возможные заболевания (только из контекста)\n"
        "2) Рекомендации на основании протоколов\n"
        "3) Когда необходимо обратиться к врачу\n"
        "4) Напоминание что ты не врач\n"
    ),
    "treatment": (
        "Ты — МедАссистент, виртуальный помощник по клиническим протоколам МЗ РК.\n"
        "Ты НЕ врач и НЕ назначаешь лечение. Отвечай ТОЛЬКО на русском языке.\n\n"
        "Пользователь спрашивает о лечении. Ответь строго на основании контекста.\n\n"
        "СТРОГИЕ ПРАВИЛА:\n"
        "1. Описывай только то лечение, которое указано в протоколах.\n"
        "2. НИКОГДА не выдумывай лекарства или дозировки.\n"
        "3. Всегда напоминай, что назначать лечение может только врач.\n"
        "4. Указывай из какого протокола взята информация.\n\n"
        "Формат ответа:\n"
        "1) Подходы к лечению из протоколов\n"
        "2) Немедикаментозные рекомендации (если есть в контексте)\n"
        "3) Напоминание обратиться к врачу для назначения лечения\n"
        "4) Напоминание что ты не врач\n"
    ),
    "info": (
        "Ты — МедАссистент, виртуальный помощник по клиническим протоколам МЗ РК.\n"
        "Ты НЕ врач. Отвечай ТОЛЬКО на русском языке.\n\n"
        "Пользователь хочет узнать информацию о заболевании. Ответь на основании контекста.\n\n"
        "СТРОГИЕ ПРАВИЛА:\n"
        "1. Отвечай ИСКЛЮЧИТЕЛЬНО на основании предоставленного контекста.\n"
        "2. НИКОГДА не выдумывай информацию.\n"
        "3. Если в контексте нет ответа — честно скажи.\n\n"
        "Формат ответа:\n"
        "1) Определение заболевания\n"
        "2) Основные симптомы и признаки\n"
        "3) Методы диагностики\n"
        "4) Общие подходы к лечению\n"
        "5) Напоминание что ты не врач\n"
    ),
}

NOT_MEDICAL_RESPONSE = (
    "Я — МедАссистент и могу помочь только с медицинскими вопросами.\n\n"
    "Вы можете:\n"
    "• Описать симптомы (например: 'температура 38, кашель, слабость')\n"
    "• Спросить о заболевании (например: 'что такое бронхиальная астма')\n"
    "• Узнать о лечении (например: 'как лечить гастрит')\n\n"
    f"{SAFETY_TEXT}"
)

FOLLOWUP_QUESTIONS = [
    "Как давно начались симптомы и усиливаются ли они?",
    "Есть ли температура? Какая максимальная?",
    "Были ли контакты с больными за последние 2 недели?",
    "Есть ли хронические заболевания или принимаете лекарства?",
]

MAX_CONTEXT_CHARS = 3000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_interaction(question: str, answer: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "interactions.log")
    with open(path, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}]\nQ: {question}\nA: {answer}\n\n")


def detect_red_flags(text: str) -> list[str]:
    t = text.lower()
    return [key for key, patterns in RED_FLAGS.items()
            if any(p in t for p in patterns)]


def _build_context(docs: list[dict]) -> str:
    by_disease: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for d in docs:
        disease = d.get("disease", "Неизвестно")
        section = d.get("section", "general")
        text    = d["text"].strip()
        if text:
            by_disease[disease][section].append(text)

    parts: list[str] = []
    total_len = 0

    for disease, sections in by_disease.items():
        block = f"=== {disease} ===\n"
        for section, texts in sections.items():
            label = SECTION_LABELS.get(section, section.title())
            combined = " ".join(texts)
            max_per_section = MAX_CONTEXT_CHARS // max(len(by_disease), 1) // max(len(sections), 1)
            if len(combined) > max_per_section:
                combined = combined[:max_per_section] + "…"
            block += f"[{label}]\n{combined}\n\n"

        if total_len + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total_len += len(block)

    return "\n".join(parts) if parts else "Подходящие протоколы не найдены."

def _build_prompt(question: str, docs: list[dict], intent: str) -> str:
        context = _build_context(docs)
        system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["symptoms"])

        intent_labels = {
            "symptoms": "описал симптомы",
            "treatment": "спрашивает о лечении",
            "info": "хочет узнать о заболевании",
        }
        user_action = intent_labels.get(intent, "описал симптомы")

        return (
            "<s>[INST] "
            f"{system_prompt}\n"
            f"Пользователь {user_action}: {question}\n\n"
            f"Выдержки из клинических протоколов МЗ РК:\n\n{context}\n\n"
            "Ответь строго по контексту выше. Если информации недостаточно — "
            "так и скажи. Не выдумывай."
            " [/INST]"
        )

def _fallback_answer(docs: list[dict]) -> str:
        diseases = {d.get("disease", "?") for d in docs}
        bullet = "\n".join(f"• {name}" for name in sorted(diseases)) or "• Не определено"
        return (
            "На основе запроса найдены следующие протоколы:\n\n"
            f"{bullet}\n\n"
            f"Рекомендую обратиться к врачу для профессиональной консультации.\n\n"
            f"{SAFETY_TEXT}"
        )

    # ── Public API ────────────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
        """
        Full RAG pipeline:
          1. Red-flag check → emergency message
          2. Detect intent (symptoms / treatment / info)
          3. Retrieve relevant chunks
          4. If no relevant results → reject non-medical queries
          5. Build intent-specific prompt and generate
          6. Fallback if generation fails
          7. Log interaction
        """
        # 1. Red flags
        flags = detect_red_flags(question)
        if flags:
            answer = (
                "⚠️ В описании есть симптомы, при которых нужна неотложная помощь.\n\n"
                f"{SAFETY_TEXT}"
            )
            _log_interaction(question, answer)
            return answer

        # 2. Detect intent
        intent = detect_intent(question)
        logger.info(f"Intent for '{question[:50]}': {intent}")

        # 3. Retrieve
        docs = retrieve(question)
        logger.info(f"Retrieved {len(docs)} docs for: {question!r}")

        # 4. No relevant results — likely not a medical question
        if not docs:
            answer = NOT_MEDICAL_RESPONSE
            _log_interaction(question, answer)
            return answer

        # 5. Generate with intent-specific prompt
        prompt = _build_prompt(question, docs[:TOP_DOCS], intent)

        try:
            answer = generate_answer(prompt)
        except Exception:
            logger.exception("Generation failed")
            answer = ""

        # 6. Fallback
        if not answer or len(answer.strip()) < 20:
            answer = _fallback_answer(docs)

        # 7. Log
        _log_interaction(question, answer)
        return answer