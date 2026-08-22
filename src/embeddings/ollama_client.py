import os
import logging

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load project .env so OLLAMA_URL / model overrides take effect without needing
# manual env exports (configs/.env.template → .env).
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Dedicated embedding model. gemma3:270m is a generation model and does not
# expose embeddings; a purpose-built embedding model keeps vector dims stable.
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Fallback dimensions tracked so we can log if an existing Chroma collection
# was built with a different embedding model (mismatch → retrieval failures).
_EMBED_DIMS_BY_MODEL = {"nomic-embed-text": 768}


def embed_text(text: str) -> list[float]:
    """Embed a single text via the Ollama embeddings endpoint.

    Uses a dedicated embedding model (default nomic-embed-text, 768-dim, CPU
    friendly) to keep query and document vectors consistent. Raises a clear
    error if Ollama is unreachable, so callers can degrade gracefully.

    Args:
        text: input string to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    if not text:
        return []
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        embedding = r.json().get("embedding", [])
        if not embedding:
            raise ValueError(f"Empty embedding returned for model {EMBED_MODEL}")
        return embedding
    except Exception as e:
        logger.error("Ollama embedding failed: %s", e)
        raise RuntimeError(
            f"Could not embed text. Is Ollama running ('{OLLAMA_URL}') with "
            f"model '{EMBED_MODEL}' pulled? Run: ollama pull {EMBED_MODEL}. "
            f"Reason: {e}"
        ) from e


def generate_answer(prompt: str, model: str | None = None) -> str:
    """Generate a completion from the configured Ollama generation model."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model or os.getenv("OLLAMA_GEN_MODEL", "gemma3:270m"),
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")
