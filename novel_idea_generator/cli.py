"""Command-line interface for the novel idea generator."""

from __future__ import annotations

from typing import Optional, Sequence

from .generation import generate_novel_concepts


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate novel concepts for a term.")
    parser.add_argument("term", help="seed term to generate ideas around")
    parser.add_argument("-n", "--n-concepts", type=int, default=6)
    parser.add_argument("--min-distance", type=float, default=0.55,
                        help="closer than this = obvious cliche (blacklist)")
    parser.add_argument("--max-distance", type=float, default=0.80,
                        help="upper bound of the 'somewhat nearby' band (whitelist)")
    parser.add_argument("--max-results", type=int, default=12,
                        help="cap on each list size")
    parser.add_argument("--json", action="store_true", help="print full JSON result")
    args = parser.parse_args(argv)

    result = generate_novel_concepts(
        args.term,
        n_concepts=args.n_concepts,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        max_results=args.max_results,
    )

    if args.json:
        print(result.to_json(indent=2))
        return

    print(f"\nTerm: {result.term}")
    print(f"\nWhitelist (nearby-band lenses): {', '.join(result.whitelist)}")
    print(f"Blacklist (too-close cliches): {', '.join(result.blacklist)}")
    print("\nNovel concepts:")
    for c in result.concepts:
        print(f"  - {c}")
