"""Step 1: derive whitelist / blacklist from embedding geometry."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .config import EMBEDDING_MODEL_NAME
from .models import get_embedder
from .vocabulary import get_vocab_embeddings, load_concept_vocabulary


def build_concept_lists(
    term: str,
    *,
    min_distance: float = 0.55,
    max_distance: float = 0.80,
    vocabulary: Optional[Sequence[str]] = None,
    max_results: int = 12,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> tuple[List[str], List[str]]:
    """Split a vocabulary into whitelist / blacklist by cosine *distance*.

    ``distance = 1 - cosine_similarity`` to ``term`` (0 = identical, ~1 =
    unrelated).

    * blacklist = concepts closer than ``min_distance`` (the obvious, cliche
      neighbours the calling model would default to).
    * whitelist = concepts whose distance falls in ``[min_distance,
      max_distance]`` - the "somewhat nearby" band: related but not obvious.

    Both lists are ordered nearest-first and capped at ``max_results``.
    """
    if not 0.0 <= min_distance < max_distance:
        raise ValueError("require 0 <= min_distance < max_distance")

    words = (
        list(vocabulary)
        if vocabulary is not None
        else list(load_concept_vocabulary())
    )
    words = list(dict.fromkeys(words))
    if not words:
        raise ValueError("vocabulary is empty")

    embedder = get_embedder(embedding_model)
    term_vec = embedder.encode([term], normalize_embeddings=True)[0].astype(np.float32)
    vocab_vecs = get_vocab_embeddings(words, embedding_model)

    # cosine similarity (vectors are normalised -> dot product) -> distance
    distances = 1.0 - (vocab_vecs @ term_vec)
    order = np.argsort(distances)  # nearest first

    blacklist: List[str] = []
    whitelist: List[str] = []
    for i in order:
        word = words[i]
        if word.lower() == term.lower():
            continue
        d = float(distances[i])
        if d < min_distance:
            if len(blacklist) < max_results:
                blacklist.append(word)
        elif d <= max_distance:
            if len(whitelist) < max_results:
                whitelist.append(word)
        if len(blacklist) >= max_results and len(whitelist) >= max_results:
            break

    return whitelist, blacklist
