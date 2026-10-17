"""Export exact certificates for successful five-pass CPU greedy repairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from book import verify
from transfer_v2 import (BLUE_BOOK, N, PAIRS, RED_BOOK, edge_list, flip,
                         load_split, score, sha256)


def certify(data: Path, split: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = load_split(data, split)
    output.mkdir(parents=True)
    certificates = []
    for row in rows:
        adjacency = [int(word, 16) for word in row["rows_hex"]]
        current = row["loss"]
        for _ in range(5):
            if current == 0:
                break
            candidates = []
            for u, v in PAIRS:
                flip(adjacency, u, v)
                candidates.append((score(adjacency)[0], u, v))
                flip(adjacency, u, v)
            best, u, v = min(candidates)
            if best >= current:
                break
            flip(adjacency, u, v)
            current = best
        if current != 0:
            continue
        certificate = {"n": N, "red_edges": edge_list(adjacency)}
        exact = verify(N, certificate["red_edges"], RED_BOOK, BLUE_BOOK)
        if not exact["valid"]:
            raise AssertionError("greedy zero-loss certificate failed exact verifier")
        path = output / f"{row['id']}.json"
        path.write_text(json.dumps(certificate, separators=(",", ":")) + "\n")
        certificates.append({"id": row["id"], "sha256": sha256(path),
                             "red_edges": exact["red_edges"]})
    result = {"version": 2, "split": split,
              "dataset_sha256": sha256(data / f"{split}.jsonl"),
              "successes": len(certificates), "certificates": certificates}
    (output / "index.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = certify(args.data, args.split, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "certificates"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
