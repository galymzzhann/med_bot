# src/model.py
"""
Ollama LLM client.
Exposes generate_answer(prompt) and set_model(name).
"""

import os
import json
import logging
import requests
import yaml
from dotenv import load_dotenv


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = project_root()
load_dotenv(os.path.join(ROOT, ".env"))

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("model")

cfg = load_config()

OLLAMA_URL     = os.getenv("OLLAMA_URL", cfg["model"]["ollama_url"]).rstrip("/")
DEFAULT_MODEL  = os.getenv("OLLAMA_MODEL", cfg["model"]["default_model"])
ALLOWED_MODELS = cfg["model"]["allowed_models"]
MAX_TOKENS     = cfg["model"]["max_tokens"]
TEMPERATURE    = cfg["model"]["temperature"]

CURRENT_MODEL: str = DEFAULT_MODEL


# ── Public API ────────────────────────────────────────────────────────────────

def set_model(name: str) -> None:
    """Switch the active Ollama model. Raises ValueError for unknown names."""
    global CURRENT_MODEL
    if name not in ALLOWED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Allowed: {ALLOWED_MODELS}")
    CURRENT_MODEL = name
    logger.info(f"Model switched to: {CURRENT_MODEL}")


def generate_answer(prompt: str) -> str:
    """Send prompt to Ollama via streaming /api/generate and return full response."""
    url     = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model":       CURRENT_MODEL,
        "prompt":      prompt,
        "stream":      True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        raise

    answer = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data   = json.loads(line.decode("utf-8"))
            answer += data.get("response", "")
        except json.JSONDecodeError:
            continue

    return answer.strip()
