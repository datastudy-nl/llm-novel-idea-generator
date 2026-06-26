"""Step 2 + public entry point: prompt the generator under the constraints."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .concept_lists import build_concept_lists
from .config import EMBEDDING_MODEL_NAME, GENERATOR_MODEL_NAME
from .models import get_generator
from .result import NovelIdeaResult


def _build_prompt(term: str, whitelist: Sequence[str], blacklist: Sequence[str],
                  n_concepts: int) -> List[dict]:
    system = (
        "You are an idea-synthesis engine. You invent novel, non-obvious "
        "concepts by blending a seed term with a fixed set of conceptual "
        "lenses. You must obey the whitelist and blacklist exactly."
    )
    user = (
        f"Seed term: {term}\n\n"
        f"WHITELIST (you MUST draw on these lenses, blending them with the "
        f"seed term):\n- " + "\n- ".join(whitelist) + "\n\n"
        f"BLACKLIST (these are the obvious, cliche associations - you are "
        f"FORBIDDEN from producing ideas centred on any of them):\n- "
        + "\n- ".join(blacklist) + "\n\n"
        f"Produce {n_concepts} novel concepts. Each concept must fuse the seed "
        f"term with at least one whitelist lens, avoid every blacklist item, "
        f"and be surprising yet coherent.\n"
        "Return ONLY a numbered list. Each line: a short concept name, then "
        "' - ', then one sentence explaining it."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_concepts(text: str) -> List[str]:
    """Pull numbered / bulleted list items out of the raw model output."""
    concepts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(?:\d+[.)]|[-*•])\s+(.*)$", line)
        if m:
            item = m.group(1).strip()
            if item:
                concepts.append(item)
    if not concepts:  # model ignored formatting; fall back to non-empty lines
        concepts = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return concepts


def generate_novel_concepts(
    term: str,
    *,
    n_concepts: int = 6,
    min_distance: float = 0.55,
    max_distance: float = 0.80,
    vocabulary: Optional[Sequence[str]] = None,
    max_results: int = 12,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    generator_model: str = GENERATOR_MODEL_NAME,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
) -> NovelIdeaResult:
    """Generate novel concepts for ``term``.

    1. derive whitelist + blacklist from a cosine-distance band over an external
       vocabulary (external to any calling LLM's priors),
    2. prompt a separate generator model under those constraints,
    3. return the parsed concepts plus the lists for transparency.

    Parameters
    ----------
    term:
        The seed term to generate ideas around.
    n_concepts:
        How many concepts to ask the generator for.
    min_distance / max_distance:
        Cosine-distance band. Concepts closer than ``min_distance`` are the
        obvious cliches (blacklist); concepts within the band are the
        "somewhat nearby" lenses (whitelist).
    vocabulary:
        Candidate concept pool. Defaults to WordNet nouns.
    max_results:
        Cap on the size of each list.
    embedding_model / generator_model:
        Hugging Face model ids.
    """
    if not term or not term.strip():
        raise ValueError("term must be a non-empty string")
    term = term.strip()

    whitelist, blacklist = build_concept_lists(
        term,
        min_distance=min_distance,
        max_distance=max_distance,
        vocabulary=vocabulary,
        max_results=max_results,
        embedding_model=embedding_model,
    )

    generator = get_generator(generator_model)
    messages = _build_prompt(term, whitelist, blacklist, n_concepts)
    output = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        return_full_text=False,
    )
    raw = output[0]["generated_text"]
    if isinstance(raw, list):  # some pipeline versions return chat turns
        raw = raw[-1].get("content", "")

    concepts = _parse_concepts(raw)

    return NovelIdeaResult(
        term=term,
        whitelist=whitelist,
        blacklist=blacklist,
        concepts=concepts,
        raw_generation=raw,
    )
