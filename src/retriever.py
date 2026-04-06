# src/retriever.py
"""
Smart retriever with:
  1. Relevance threshold — rejects random/non-medical queries
  2. Query intent detection — symptoms vs treatment vs disease info
  3. Section-aware boosting per intent type
  4. Diversity filtering — max 2 chunks per disease
"""

import os
import re
import pickle
import logging
from collections import defaultdict

import yaml
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


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
logger = logging.getLogger("retriever")

cfg  = _load_config()
ROOT = _project_root()

IDX_DIR   = os.path.join(ROOT, cfg["data"]["faiss_index_dir"])
IDX_PATH  = os.path.join(IDX_DIR, "index.faiss")
META_PATH = os.path.join(IDX_DIR, "metadata.pkl")

TOP_K      = cfg["retrieve"]["top_k"]
RR_K       = cfg["retrieve"]["rerank_k"]
CE_MODEL   = cfg["retrieve"]["cross_encoder_model"]
EMB_MODEL  = cfg["embed"]["model_name"]
MIN_LEN    = cfg["embed"]["min_chunk_len"]

SECTION_BOOST = 0.08
MIN_RELEVANCE_SCORE = 0.75   # below this, results are considered irrelevant


# ── Query intent detection ────────────────────────────────────────────────────

TREATMENT_PATTERNS = [
    r"как\s+лечить", r"как\s+лечится", r"лечение\b", r"чем\s+лечить",
    r"какие\s+лекарства", r"какие\s+препараты", r"что\s+принимать",
    r"что\s+пить", r"терапия\b", r"помогает\s+от", r"средств[оа]\s+от",
]

INFO_PATTERNS = [
    r"что\s+тако[ей]", r"расскажи\s+о", r"расскажите\s+о",
    r"информаци[яю]\s+о", r"опиши\s", r"что\s+за\s+болезнь",
    r"причины\s", r"профилактика\s", r"диагностика\s",
]

INTENT_SECTIONS = {
    "symptoms":  {"symptoms", "diagnostics", "definition"},
    "treatment": {"treatment", "prevention", "symptoms"},
    "info":      {"definition", "classification", "etiology", "symptoms"},
}


def detect_intent(query: str) -> str:
    q = query.lower().strip()
    for pat in TREATMENT_PATTERNS:
        if re.search(pat, q):
            return "treatment"
    for pat in INFO_PATTERNS:
        if re.search(pat, q):
            return "info"
    return "symptoms"


# ── Load index + models once at import time ───────────────────────────────────

logger.info(f"Loading FAISS index from {IDX_PATH}")
if not os.path.exists(IDX_PATH):
    raise FileNotFoundError(f"Index not found: {IDX_PATH}. Run indexer.py first.")

index = faiss.read_index(IDX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)
logger.info(f"Loaded index: {len(metadata)} entries")

ALL_DISEASES = {m.get("disease", "").lower() for m in metadata if m.get("disease")}

logger.info(f"Loading embedder: {EMB_MODEL}")
embedder = SentenceTransformer(EMB_MODEL)

reranker = CrossEncoder(CE_MODEL) if CE_MODEL else None
if reranker:
    logger.info(f"Cross-encoder loaded: {CE_MODEL}")


# ── Disease name matching ─────────────────────────────────────────────────────

def _find_disease_in_query(query: str) -> str | None:
    q = query.lower().strip()
    for prefix in ["что такое ", "расскажи о ", "расскажите о ",
                   "информация о ", "как лечить ", "как лечится ",
                   "лечение ", "причины ", "профилактика ", "диагностика ",
                   "что за болезнь "]:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    best_match = None
    best_len = 0
    for disease in ALL_DISEASES:
        if disease in q or q in disease:
            if len(disease) > best_len:
                best_match = disease
                best_len = len(disease)
    return best_match


# ── Core ──────────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int | None = None) -> list[dict]:

    real_words = [w for w in query.strip().split() if len(w) >= 3]
    if len(real_words) < 1:
        logger.info(f"Query rejected: no real words")
        return []

    k = top_k or TOP_K
    intent = detect_intent(query)
    boost_sections = INTENT_SECTIONS.get(intent, INTENT_SECTIONS["symptoms"])

    logger.info(f"Query intent: {intent}")

    disease_match = _find_disease_in_query(query)
    if disease_match:
        logger.info(f"Disease name detected: {disease_match}")

    fetch_k = min(k * 4, index.ntotal)
    qv = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(qv, fetch_k)

    raw_docs: list[dict] = []
    seen_texts: set[str] = set()

    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        entry = metadata[idx].copy()
        text  = entry.pop("text", "")
        if len(text.strip()) < MIN_LEN:
            continue

        text_key = text[:80].strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        score = float(dist)

        section = entry.get("section", "general")
        if section in boost_sections:
            score += SECTION_BOOST

        if disease_match and disease_match in entry.get("disease", "").lower():
            score += 0.15

        raw_docs.append({"score": score, "text": text, **entry})

    raw_docs.sort(key=lambda d: d["score"], reverse=True)

    if raw_docs and raw_docs[0]["score"] < MIN_RELEVANCE_SCORE:
        logger.info(f"Best score {raw_docs[0]['score']:.3f} below threshold")
        return []

    MAX_PER_DISEASE = 2
    MAX_FOR_MATCHED = 4 if disease_match else 2

    disease_count: dict[str, int] = {}
    docs: list[dict] = []

    for d in raw_docs:
        disease = d.get("disease", "unknown")
        count = disease_count.get(disease, 0)
        max_allowed = MAX_FOR_MATCHED if (disease_match and disease_match in disease.lower()) else MAX_PER_DISEASE
        if count >= max_allowed:
            continue
        disease_count[disease] = count + 1
        docs.append(d)
        if len(docs) >= k:
            break

    if reranker and docs:
        pairs  = [[query, d["text"]] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = []
        for d, s in ranked[:RR_K]:
            d["score"] = float(s)
            docs.append(d)
    else:
        docs = docs[:RR_K]

    return docs


def retrieve_grouped(query: str, top_k: int | None = None) -> dict[str, list[dict]]:
    docs = retrieve(query, top_k)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        grouped[d.get("disease", "unknown")].append(d)
    return dict(grouped)


if __name__ == "__main__":
    q = input("Query: ")
    intent = detect_intent(q)
    print(f"Intent: {intent}")
    for i, d in enumerate(retrieve(q), 1):
        sec = d.get("section", "?")
        dis = d.get("disease", "?")
        print(f"{i}. [{d['score']:.3f}] {dis} / {sec}")
        print(f"   {d['text'][:120]}…\n")