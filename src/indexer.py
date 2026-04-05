# src/indexer.py
"""
Builds a FAISS index from precomputed embeddings.
Reads:  data/embeddings/embeddings.npy + metadata.pkl
Writes: data/faiss_index/index.faiss   + metadata.pkl
"""

import os
import pickle
import logging

import yaml
import numpy as np
import faiss


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
logger = logging.getLogger("indexer")

cfg  = _load_config()
ROOT = _project_root()

EMB_DIR   = os.path.join(ROOT, cfg["data"]["embeddings_dir"])
INDEX_DIR = os.path.join(ROOT, cfg["data"]["faiss_index_dir"])
FACTORY   = cfg["index"]["factory_string"]
os.makedirs(INDEX_DIR, exist_ok=True)

EMB_PATH  = os.path.join(EMB_DIR,   "embeddings.npy")
META_IN   = os.path.join(EMB_DIR,   "metadata.pkl")
IDX_PATH  = os.path.join(INDEX_DIR, "index.faiss")
META_OUT  = os.path.join(INDEX_DIR, "metadata.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(
            f"Embeddings not found: {EMB_PATH}. Run embed.py first."
        )

    logger.info(f"Loading embeddings from {EMB_PATH}")
    emb = np.load(EMB_PATH).astype("float32")
    emb = np.ascontiguousarray(emb)
    logger.info(f"Shape: {emb.shape}")

    with open(META_IN, "rb") as f:
        meta = pickle.load(f)
    logger.info(f"Metadata entries: {len(meta)}")

    d = emb.shape[1]
    logger.info(f"Building FAISS index [{FACTORY}] dim={d} metric=InnerProduct")
    idx = faiss.index_factory(d, FACTORY, faiss.METRIC_INNER_PRODUCT)

    if not idx.is_trained:
        logger.info("Training index …")
        idx.train(emb)

    idx.add(emb)
    logger.info(f"Vectors in index: {idx.ntotal}")

    faiss.write_index(idx, IDX_PATH)
    with open(META_OUT, "wb") as f:
        pickle.dump(meta, f)

    size_mb = os.path.getsize(IDX_PATH) / 1_048_576
    logger.info(f"Saved index ({size_mb:.1f} MB) → {IDX_PATH}")
    logger.info(f"Saved metadata → {META_OUT}")


if __name__ == "__main__":
    main()