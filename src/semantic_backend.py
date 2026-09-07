"""
semantic_backend.py
--------------------
Optional upgrade path for semantic similarity.

By default the grader uses TF-IDF cosine similarity (src/grader.py), which
works fully offline with no model downloads — good enough for keyword-rich
subject answers (Maths/Physics/Chemistry short answers, definitions, derivations).

If the `sentence-transformers` package is installed AND a model can be loaded
(the first run downloads ~80MB from huggingface.co, which needs internet
access on whatever machine actually runs this app), this module swaps in a
proper sentence-embedding similarity, which handles paraphrased / reworded
correct answers much better than TF-IDF.

This is intentionally lazy + fail-safe: if the package isn't installed, or the
model can't be loaded (no internet, first run in an offline sandbox, etc.),
`semantic_backend_available()` returns False and grader.py silently falls
back to TF-IDF. Nothing breaks either way.
"""

from __future__ import annotations

_model = None
_load_attempted = False
_available = False


def _try_load():
    global _model, _load_attempted, _available
    if _load_attempted:
        return
    _load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        # small, fast, good general-purpose model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _available = True
    except Exception:
        _model = None
        _available = False


def semantic_backend_available() -> bool:
    _try_load()
    return _available


def get_semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between sentence embeddings of two texts, in [0, 1]."""
    _try_load()
    if not _available or _model is None:
        return 0.0
    if not text_a or not text_b:
        return 0.0
    import numpy as np
    emb = _model.encode([text_a, text_b], normalize_embeddings=True)
    sim = float(np.dot(emb[0], emb[1]))
    return max(0.0, min(1.0, sim))
