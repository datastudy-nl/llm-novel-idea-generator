"""Novel Search Space.

Idea
----
A large language model that calls itself for "new ideas" is trapped inside its
own prior: it keeps returning the statistically obvious associations of a term.
This package breaks that loop by deriving the candidate concepts from a
*different* mechanism - the geometry of an embedding model over a large external
lexical database (WordNet) - instead of from autoregressive token prediction.
Nothing is hard-coded.

For an input ``term`` we measure the cosine *distance* (``1 - similarity``) from
the term to every concept in the vocabulary and split by a distance band:

* **blacklist** - concepts *closer* than ``min_distance``. These are the cliche,
  "of course" associations the calling model would default to. We forbid them so
  the generator cannot fall back into the obvious.
* **whitelist** - concepts whose distance falls in ``[min_distance,
  max_distance]`` - the "somewhat nearby" band: related enough to be relevant,
  far enough to be surprising. These are the lenses the generator must blend
  with the term.

A second (generator) model is then prompted to invent concepts that combine the
term with the whitelist while strictly avoiding the blacklist.

Public API
----------
* :func:`generate_novel_concepts` - the high-level "tool" an LLM agent calls.
* :func:`build_concept_lists` - just the whitelist/blacklist geometry step.
* :func:`load_concept_vocabulary` - the WordNet-derived concept pool.
* :class:`NovelSearchSpaceResult` - the structured, JSON-serialisable result.
"""

from __future__ import annotations

from .concept_lists import build_concept_lists
from .config import (
    CACHE_DIR,
    EMBEDDING_MODEL_NAME,
    GENERATOR_MODEL_NAME,
)
from .generation import generate_novel_concepts
from .result import NovelSearchSpaceResult
from .vocabulary import load_concept_vocabulary

__all__ = [
    "generate_novel_concepts",
    "build_concept_lists",
    "load_concept_vocabulary",
    "NovelSearchSpaceResult",
    "EMBEDDING_MODEL_NAME",
    "GENERATOR_MODEL_NAME",
    "CACHE_DIR",
]

__version__ = "0.1.0"
