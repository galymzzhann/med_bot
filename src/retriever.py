# src/retriever.py
"""
Two-stage retriever with section-aware boosting.

Key improvements:
  1. Boosts chunks from "symptoms" and "diagnostics" sections for
     symptom-related queries (the most common user intent).
  2. Groups results by disease so the LLM gets coherent context
     instead of random fragments from 5 different diseases.
  3. Returns rich metadata (disease name, section, URL) for citations.
  4. Junk filtering is no longer needed here — it's done at scrape time.
"""

import os
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

# Sections that are most relevant when the user describes symptoms
SYMPTOM_SECTIONS = {"symptoms", "diagnostics", "definition"}
SECTION_BOOST    = 0.08   # cosine-distance bonus for relevant sections


# ── Load index + models once at import time ───────────────────────────────────

logger.info(f"Loading FAISS index from {IDX_PATH}")
if not os.path.exists(IDX_PATH):
    raise FileNotFoundError(f"Index not found: {IDX_PATH}. Run indexer.py first.")

index = faiss.read_index(IDX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)
logger.info(f"Loaded index: {len(metadata)} entries")

logger.info(f"Loading embedder: {EMB_MODEL}")
embedder = SentenceTransformer(EMB_MODEL)

reranker = CrossEncoder(CE_MODEL) if CE_MODEL else None
if reranker:
    logger.info(f"Cross-encoder loaded: {CE_MODEL}")


# ── Core ──────────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Returns the most relevant chunks for *query*.
    Enforces diversity: max 2 chunks per disease so results span
    multiple diseases instead of repeating one.

    Each result dict has keys:
      score, text, disease, section, source, url, chunk_id
    """
    k = top_k or TOP_K
    # Fetch extra candidates so we have enough after diversity filtering
    fetch_k = min(k * 4, index.ntotal)
    qv = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(qv, fetch_k)

    raw_docs: list[dict] = []
    seen_texts: set[str] = set()   # near-duplicate filter

    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        entry = metadata[idx].copy()
        text  = entry.pop("text", "")
        if len(text.strip()) < MIN_LEN:
            continue

        # Skip near-duplicates (same first 80 chars)
        text_key = text[:80].strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        # Inner Product score (higher = more similar)
        # With normalized vectors, this equals cosine similarity (0 to 1)
        score = float(dist)

        # Boost chunks from symptom-relevant sections
        section = entry.get("section", "general")
        if section in SYMPTOM_SECTIONS:
            score += SECTION_BOOST

        raw_docs.append({"score": score, "text": text, **entry})

    # Sort by (boosted) score descending
    raw_docs.sort(key=lambda d: d["score"], reverse=True)

    # Diversity filter: max 2 chunks per disease
    MAX_PER_DISEASE = 2
    disease_count: dict[str, int] = {}
    docs: list[dict] = []

    for d in raw_docs:
        disease = d.get("disease", "unknown")
        count   = disease_count.get(disease, 0)
        if count >= MAX_PER_DISEASE:
            continue
        disease_count[disease] = count + 1
        docs.append(d)
        if len(docs) >= k:
            break

    # Optional cross-encoder reranking
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
    """
    Same as retrieve(), but groups results by disease name.
    Useful for building per-disease context blocks in the prompt.
    """
    docs = retrieve(query, top_k)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        grouped[d.get("disease", "unknown")].append(d)
    return dict(grouped)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = input("Query: ")
    for i, d in enumerate(retrieve(q), 1):
        sec = d.get("section", "?")
        dis = d.get("disease", "?")
        print(f"{i}. [{d['score']:.3f}] {dis} / {sec}")
        print(f"   {d['text'][:120]}…\n")
        