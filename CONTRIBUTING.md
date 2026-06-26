# Contributing

Thanks for your interest in improving Novel Search Space!

## Development setup

```bash
git clone https://github.com/datastudy-nl/llm-novel-idea-generator.git
cd llm-novel-idea-generator
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Unix: source .venv/bin/activate
pip install -e .
```

## Ground rules

- **Nothing is hard-coded.** The candidate concepts must always come from an
  external source (WordNet) and the geometry, never from a fixed list of "good"
  ideas. Keep that property intact.
- Keep the public API stable: `generate_novel_concepts`, `build_concept_lists`,
  `load_concept_vocabulary`, and `NovelSearchSpaceResult` are the contract.
- Prefer small, focused modules. Each file in `novel_search_space/` has a
  single responsibility (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

## Making a change

1. Open an issue describing the change (bug, feature, or tuning).
2. Create a branch and make your change.
3. Verify the package still imports and runs end-to-end:
   ```bash
   python -c "import novel_search_space"
   python -m novel_search_space umbrella --json
   ```
4. Open a pull request with a clear description and rationale.

## Ideas for contributions

- A frequency filter to drop obscure WordNet lemmas (e.g. `algolagnia`).
- Support for additional parts of speech beyond nouns.
- Alternative embedding / generator backends.
- A small evaluation harness for "novel but coherent" scoring.

## Licensing of contributions

This project is licensed under **CC BY-NC 4.0** (see [LICENSE](LICENSE)). By
submitting a contribution you agree it is licensed under the same terms, with
attribution to **Lars Cornelissen (datastudy.nl)** retained. Commercial use of
the project requires a separate license from the copyright holder
(<lars@datastudy.nl>).

## Code style

- Type hints on public functions.
- Docstrings explaining *why*, not just *what*.
- No new runtime dependencies without discussion.
