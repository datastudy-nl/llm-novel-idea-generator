"""Lazily-loaded, cached model handles.

Loading transformer models is expensive, so both loaders are memoised with
``lru_cache``. The first call downloads/initialises the model; subsequent tool
calls reuse the in-process instance.
"""

from __future__ import annotations

from functools import lru_cache

from .config import EMBEDDING_MODEL_NAME, GENERATOR_MODEL_NAME


@lru_cache(maxsize=4)
def get_embedder(model_name: str = EMBEDDING_MODEL_NAME):
    """Return a cached :class:`~sentence_transformers.SentenceTransformer`."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def get_generator(model_name: str = GENERATOR_MODEL_NAME):
    """Return a cached Hugging Face ``text-generation`` pipeline."""
    from transformers import pipeline

    return pipeline("text-generation", model=model_name)
