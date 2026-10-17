"""Verifier-budgeted graph-repair controls for the frozen edit study."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from book import verify
from edit_study import BLUE_BOOK, N, PAIRS, RED_BOOK, edge_list, flip, load_split, score, sha256


def rollout(start: list[int], *, method: str, proposals: int, seed: int) -> dict:
    """Count each candidate scored by the exact objective as one verifier call."""
    if method not in {"random", "greedy", "anneal"} or proposals < 1:
        raise ValueError("invalid method or proposal budget")
    rng = random.Random(seed)
    rows = start.copy()
    current = score(rows)[0]
    initial = current
    evaluated = 0
    accepted = 0
    best = current
    best_rows = rows.copy()
    while evaluated < proposals and current > 0:
        if method == "greedy":
            remaining = proposals - evaluated
            candidate_pairs = PAIRS[:remaining]
        else:
            candidate_pairs = (rng.choice(PAIRS),)
        choice = None
        for u, v in candidate_pairs:
            flip(rows, u, v)
            candidate = score(rows)[0]
            flip(rows, u, v)
            evaluated += 1
            if choice is None or (candidate, u, v) < (choice[0], choice[1], choice[2]):
                choice = (candidate, u, v)
        assert choice is not None
        candidate, u, v = choice
        if method == "greedy":
            accept = candidate < current
        elif method == "random":
            accept = True
        else:
            temperature = 2.0 * (0.08 / 2.0) ** ((evaluated % (len(PAIRS) * 5)) / (len(PAIRS) * 5))
            accept = candidate <= current or rng.random() < math.exp((current - candidate) / temperature)
        if accept:
            flip(rows, u, v)
            current = candidate
            accepted += 1
            if current < best:
                best = current
                best_rows = rows.copy()
    certificate = {"n": N, "red_edges": edge_list(best_rows)}
    exact = verify(N, certificate["red_edges"], RED_BOOK, BLUE_BOOK)
    if (best == 0) != exact["valid"]:
        raise AssertionError("objective and independent checker disagree")
    return {"initial_loss": initial, "best_loss": best, "success": exact["valid"],
            "evaluated": evaluated, "accepted": accepted}


def evaluate(data: Path, split: str, *, methods: tuple[str, ...], budgets: tuple[int, ...]) -> dict:
    rows = load_split(data, split)
    results = []
    for method in methods:
        for budget in budgets:
            records = []
            for index, row in enumerate(rows):
                start = [int(word, 16) for word in row["rows_hex"]]
                records.append({"id": row["id"], **rollout(start, method=method, proposals=budget,
                                                          seed=871 + index)})
            results.append({"method": method, "proposals": budget,
                            "successes": sum(record["success"] for record in records),
                            "mean_best_loss": round(sum(record["best_loss"] for record in records) / len(records), 4),
                            "mean_evaluated": round(sum(record["evaluated"] for record in records) / len(records), 4),
                            "records": records})
    return {"split": split, "rows": len(rows), "data_sha256": sha256(data / f"{split}.jsonl"),
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 325, 1625])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = evaluate(args.data, args.split, methods=("random", "greedy", "anneal"),
                      budgets=tuple(args.budgets))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split": args.split, "rows": result["rows"],
                      "summary": [{k: v for k, v in record.items() if k != "records"}
                                  for record in result["results"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
