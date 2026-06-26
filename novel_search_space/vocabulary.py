"""Concept vocabulary sourced from WordNet, plus embedding caching.

Candidate concepts are pulled from the WordNet lexical database (downloaded on
demand) - not hard-coded. The embedding geometry then decides which concepts
fall in the distance band around a given term.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import CACHE_DIR
from .models import get_embedder


@lru_cache(maxsize=4)
def load_concept_vocabulary(
    pos: str = "n", max_words: Optional[int] = None, seed: int = 0
) -> tuple:
    """Return a concept vocabulary from WordNet (downloaded on first use).

    Nothing is hard-coded. Multi-word and non-alphabetic lemmas are dropped so
    the geometry works over clean single-word concepts. ``pos='n'`` keeps nouns,
    which read best as "concepts". Pass ``max_words`` to work with a random
    (seeded) subset for speed.
    """
    import nltk
    from nltk.corpus import wordnet

    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        wordnet.ensure_loaded()

    words = sorted(
        {
            lemma.lower()
            for lemma in wordnet.all_lemma_names(pos=pos)
            if lemma.isalpha() and len(lemma) >= 3
        }
    )
    if max_words is not None and len(words) > max_words:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(words), size=max_words, replace=False))
        words = [words[i] for i in idx]
    # tuple so the result is hashable / cacheable
    return tuple(words)


def _vocab_cache_path(model_name: str, words: Sequence[str]) -> Path:
    digest = hashlib.md5(
        (model_name + "\x00" + "\x00".join(words)).encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"emb-{digest}.npy"


def get_vocab_embeddings(words: Sequence[str], model_name: str) -> np.ndarray:
    """Embed ``words`` once and cache the normalised matrix to disk."""
    cache_path = _vocab_cache_path(model_name, words)
    if cache_path.exists():
        return np.load(cache_path)
    embedder = get_embedder(model_name)
    emb = embedder.encode(
        list(words),
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
    ).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb
