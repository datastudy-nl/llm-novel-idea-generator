# Architecture

The package splits the original single-file prototype into focused modules,
each with one responsibility. Data flows from a seed term to a structured
result through two stages: **geometry** (deterministic) and **generation**
(stochastic).

## Modules

| Module             | Responsibility                                                       |
| ------------------ | ------------------------------------------------------------------- |
| `config.py`        | Default model ids and the on-disk cache location.                   |
| `models.py`        | Lazy, `lru_cache`-memoised loaders for the embedder and generator.  |
| `vocabulary.py`    | WordNet concept pool + per-`(model, wordlist)` embedding cache.     |
| `concept_lists.py` | **Stage 1** — distance-band whitelist/blacklist (`build_concept_lists`). |
| `generation.py`    | **Stage 2** — prompt build, parse, and `generate_novel_concepts`.   |
| `result.py`        | `NovelSearchSpaceResult` dataclass (`to_dict` / `to_json`).         |
| `cli.py`           | argparse CLI (`main`).                                              |
| `__main__.py`      | Enables `python -m novel_search_space`.                             |
| `__init__.py`      | Re-exports the public API.                                          |

## Data flow

```
term ──► build_concept_lists ──► (whitelist, blacklist) ──► _build_prompt
            │                                                    │
            ├─ load_concept_vocabulary (WordNet)                 ▼
            └─ get_vocab_embeddings (cached)            get_generator → raw text
                                                                 │
                                                          _parse_concepts
                                                                 │
                                                                 ▼
                                                         NovelSearchSpaceResult
```

## Stage 1 — geometry (deterministic)

`build_concept_lists` embeds the seed term and computes cosine distance to every
cached vocabulary vector. Vectors are L2-normalised, so cosine similarity is a
single matrix–vector dot product, and `distance = 1 − similarity`. Concepts are
bucketed:

- `distance < min_distance` → **blacklist** (obvious clichés).
- `min_distance ≤ distance ≤ max_distance` → **whitelist** (surprising lenses).

Both lists are ordered nearest-first and capped at `max_results`.

## Stage 2 — generation (stochastic)

`generate_novel_concepts` feeds the term, whitelist, and blacklist into a chat
prompt instructing a small instruct model to **fuse** the term with whitelist
lenses while **avoiding** every blacklist item. The raw output is parsed into a
clean list of concepts.

## Caching

- **Model handles** are memoised in-process via `lru_cache` (`models.py`).
- **Vocabulary embeddings** are cached to disk under `CACHE_DIR`, keyed by an
  MD5 of `(model_name, wordlist)` (`vocabulary.py`). This makes repeated tool
  calls cheap: only the single seed term is embedded per call.
