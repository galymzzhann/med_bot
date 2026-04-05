#!/usr/bin/env python3
"""Quick diagnostic: shows raw FAISS results for a bronchitis query."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import faiss, pickle
from sentence_transformers import SentenceTransformer

meta = pickle.load(open("data/faiss_index/metadata.pkl", "rb"))
index = faiss.read_index("data/faiss_index/index.faiss")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

query = "кашель с мокротой, температура, насморк, слабость"
qv = model.encode([query], normalize_embeddings=True)
D, I = index.search(qv, 30)

print(f"Query: {query}")
print(f"Raw FAISS top 30 (no filtering):\n")
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    m = meta[idx]
    dis = m.get("disease", "?")
    sec = m.get("section", "?")
    mark = " <<<" if "бронхит" in dis.lower() else ""
    print(f"  {rank:2d}. [{score:.4f}] {dis} / {sec}{mark}")
