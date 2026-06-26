"""Shared configuration: default model ids and the on-disk cache location.

These values are intentionally small/ubiquitous defaults so the tool runs on a
laptop. Override the cache directory with the ``NOVEL_SEARCH_SPACE_CACHE``
environment variable, or pass different model ids to the public API.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default Hugging Face Hub model ids.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
"""Tiny, ubiquitous sentence-embedding model used for the distance geometry."""

GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
"""Small instruct model that synthesises concepts under the list constraints."""

# Disk cache for vocabulary embeddings. Override via the NOVEL_SEARCH_SPACE_CACHE env var.
CACHE_DIR = Path(
    os.environ.get(
        "NOVEL_SEARCH_SPACE_CACHE", Path.home() / ".cache" / "novel_search_space"
    )
)
"""Directory where vocabulary embedding matrices are cached between runs."""
