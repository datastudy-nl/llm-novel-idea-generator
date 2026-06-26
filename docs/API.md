# API Reference

All public symbols are importable directly from the top-level package:

```python
from novel_idea_generator import (
    generate_novel_concepts,
    build_concept_lists,
    load_concept_vocabulary,
    NovelIdeaResult,
)
```

---

## `generate_novel_concepts`

```python
generate_novel_concepts(
    term: str,
    *,
    n_concepts: int = 6,
    min_distance: float = 0.55,
    max_distance: float = 0.80,
    vocabulary: Sequence[str] | None = None,
    max_results: int = 12,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    generator_model: str = GENERATOR_MODEL_NAME,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
) -> NovelIdeaResult
```

The high-level entry point — the "tool" an LLM agent calls. It (1) derives the
whitelist + blacklist from a cosine-distance band over an external vocabulary,
(2) prompts a separate generator model under those constraints, and (3) returns
the parsed concepts plus the lists for transparency.

**Parameters**

| Name              | Type            | Default | Description                                                                 |
| ----------------- | --------------- | ------- | --------------------------------------------------------------------------- |
| `term`            | `str`           | —       | Seed term to generate ideas around. Must be non-empty.                      |
| `n_concepts`      | `int`           | `6`     | How many concepts to ask the generator for.                                 |
| `min_distance`    | `float`         | `0.55`  | Below this distance → blacklist (obvious clichés).                          |
| `max_distance`    | `float`         | `0.80`  | Upper bound of the "somewhat nearby" whitelist band.                        |
| `vocabulary`      | `Sequence[str]` | `None`  | Candidate concept pool. Defaults to WordNet nouns.                          |
| `max_results`     | `int`           | `12`    | Cap on the size of each list.                                               |
| `embedding_model` | `str`           | MiniLM  | Hugging Face embedding model id.                                            |
| `generator_model` | `str`           | Qwen    | Hugging Face generator model id.                                            |
| `max_new_tokens`  | `int`           | `512`   | Generation length cap.                                                      |
| `temperature`     | `float`         | `0.9`   | Sampling temperature.                                                       |

**Returns:** [`NovelIdeaResult`](#novelidearesult)

**Raises:** `ValueError` if `term` is empty or the distance band is invalid.

---

## `build_concept_lists`

```python
build_concept_lists(
    term: str,
    *,
    min_distance: float = 0.55,
    max_distance: float = 0.80,
    vocabulary: Sequence[str] | None = None,
    max_results: int = 12,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> tuple[list[str], list[str]]
```

Just the geometry step. Returns `(whitelist, blacklist)` ordered nearest-first.
Useful if you want to plug the lists into your own generation pipeline.

`distance = 1 − cosine_similarity(term, concept)`.

**Raises:** `ValueError` if `not (0 <= min_distance < max_distance)` or the
vocabulary is empty.

---

## `load_concept_vocabulary`

```python
load_concept_vocabulary(
    pos: str = "n",
    max_words: int | None = None,
    seed: int = 0,
) -> tuple[str, ...]
```

Returns a concept vocabulary from WordNet (downloaded on first use). Multi-word
and non-alphabetic lemmas are dropped. `pos="n"` keeps nouns. Pass `max_words`
for a seeded random subset (faster experiments). The result is cached.

---

## `NovelIdeaResult`

```python
@dataclass
class NovelIdeaResult:
    term: str
    whitelist: list[str]
    blacklist: list[str]
    concepts: list[str]
    raw_generation: str = ""
```

| Method        | Description                                       |
| ------------- | ------------------------------------------------- |
| `to_dict()`   | Plain `dict` of all fields.                       |
| `to_json(**)` | JSON string (`**kwargs` forwarded to `json.dumps`). |

`raw_generation` holds the unparsed model output for debugging.

---

## Constants

| Name                   | Default                                       |
| ---------------------- | --------------------------------------------- |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2`      |
| `GENERATOR_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct`                  |
| `CACHE_DIR`            | `~/.cache/novel_idea_generator` (or `$NOVEL_IDEA_CACHE`) |
