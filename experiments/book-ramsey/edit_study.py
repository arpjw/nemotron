"""Frozen one-edge repair benchmark for the 26-vertex B4/B9 constructions.

The ten external calibration graphs are split by graph6 record before any
relabeling. No graph6 source or derived rows are committed to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from book import decode_graph6, verify

N = 26
RED_BOOK = 4
BLUE_BOOK = 9
PAIRS = tuple((u, v) for u in range(N) for v in range(u + 1, N))
FULL = (1 << N) - 1
SOURCE_SHA256 = "f0b4b4ad4507835703444c0f0990733e2f7cc31524cf91d04ba4a109f1fd52ac"
SEED = 20261016
COUNTS = {"train": 1200, "dev": 150, "test": 150}
ANCHORS = {"train": tuple(range(8)), "dev": (8,), "test": (9,)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjacency(edges: list[list[int]]) -> list[int]:
    rows = [0] * N
    for u, v in edges:
        flip(rows, u, v)
    return rows


def flip(rows: list[int], u: int, v: int) -> None:
    if not 0 <= u < v < N:
        raise ValueError("invalid edge")
    rows[u] ^= 1 << v
    rows[v] ^= 1 << u


def edge_list(rows: list[int]) -> list[list[int]]:
    if len(rows) != N or any(type(row) is not int or row < 0 or row >= 1 << N for row in rows):
        raise ValueError("invalid adjacency rows")
    return [[u, v] for u, v in PAIRS if rows[u] >> v & 1]


def score(rows: list[int]) -> tuple[int, int, int]:
    """Return excess book pages and red/blue violating spine counts."""
    loss = red = blue = 0
    for u, v in PAIRS:
        if rows[u] >> v & 1:
            excess = max(0, (rows[u] & rows[v]).bit_count() - RED_BOOK + 1)
            red += excess > 0
        else:
            excess = max(0, (FULL & ~(rows[u] | rows[v]) & ~(1 << u) & ~(1 << v)).bit_count()
                         - BLUE_BOOK + 1)
            blue += excess > 0
        loss += excess
    return loss, red, blue


def action_scores(rows: list[int]) -> list[int]:
    scores = []
    for u, v in PAIRS:
        flip(rows, u, v)
        scores.append(score(rows)[0])
        flip(rows, u, v)
    return scores


def violating_spines(rows: list[int]) -> list[tuple[int, int, int, str]]:
    spines = []
    for u, v in PAIRS:
        if rows[u] >> v & 1:
            excess = max(0, (rows[u] & rows[v]).bit_count() - RED_BOOK + 1)
            color = "R"
        else:
            excess = max(0, (FULL & ~(rows[u] | rows[v]) & ~(1 << u) & ~(1 << v)).bit_count()
                         - BLUE_BOOK + 1)
            color = "B"
        if excess:
            spines.append((excess, u, v, color))
    spines.sort(key=lambda item: (-item[0], item[1], item[2]))
    return spines


def prompt(rows: list[int], current: tuple[int, int, int]) -> str:
    words = " ".join(f"{row:07x}" for row in rows)
    spines = violating_spines(rows)
    top_spines = " ".join(f"{color}{u}-{v}:{excess}" for excess, u, v, color in spines[:12])
    return ("Repair a red/blue coloring of K26. Red edges are the 1 bits in 26 adjacency rows, "
            "listed in vertex order as seven-digit hexadecimal integers; bit v of row u is edge (u,v). "
            "The rows are symmetric with zero diagonal. Blue edges are the 0 bits off diagonal. "
            "Avoid red B4 and blue B9, where Bk is k triangles sharing an edge. "
            f"Current exact excess-page loss is {current[0]}; violating red spines {current[1]}, "
            f"blue spines {current[2]}. The twelve largest violating spines are "
            f"color/u-v/excess: {top_spines}. Rows: {words}. "
            "Choose one edge to flip. Reply only JSON {\"u\":integer,\"v\":integer}, with 0<=u<v<26.")


def load_anchors(path: Path) -> list[list[int]]:
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("McKay graph6 checksum mismatch")
    decoded = [decode_graph6(line) for line in path.read_text().splitlines() if line.strip()]
    if len(decoded) != 10 or any(n != N or not verify(n, edges, RED_BOOK, BLUE_BOOK)["valid"]
                               for n, edges in decoded):
        raise ValueError("expected ten valid 26-vertex graphs")
    return [adjacency(edges) for _, edges in decoded]


def relabel(rows: list[int], rng: random.Random) -> list[int]:
    permutation = list(range(N))
    rng.shuffle(permutation)
    result = [0] * N
    for u, v in PAIRS:
        if rows[u] >> v & 1:
            a, b = sorted((permutation[u], permutation[v]))
            flip(result, a, b)
    return result


def generate(source: Path, output: Path, *, counts: dict[str, int] | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    anchors = load_anchors(source)
    selected_counts = counts or COUNTS
    if set(selected_counts) != set(COUNTS) or any(value < 1 for value in selected_counts.values()):
        raise ValueError("all split counts must be positive")
    output.mkdir(parents=True)
    files = {}
    for split, count in selected_counts.items():
        rng = random.Random(SEED + {"train": 0, "dev": 1, "test": 2}[split])
        records = []
        seen: set[tuple[int, ...]] = set()
        for index in range(count):
            for _ in range(100):
                anchor_id = ANCHORS[split][index % len(ANCHORS[split])]
                rows = relabel(anchors[anchor_id], rng)
                perturbations = rng.randint(3, 8)
                for u, v in rng.sample(PAIRS, perturbations):
                    flip(rows, u, v)
                before = score(rows)
                identity = tuple(rows)
                if before[0] > 0 and identity not in seen:
                    break
            else:
                raise RuntimeError("could not generate distinct invalid state")
            seen.add(identity)
            candidates = action_scores(rows)
            best = min(candidates)
            best_index = candidates.index(best)
            u, v = PAIRS[best_index]
            records.append({"id": f"{split}-{index:04d}", "split": split, "anchor_id": anchor_id,
                            "rows_hex": [f"{row:07x}" for row in rows], "loss": before[0],
                            "red_violations": before[1], "blue_violations": before[2],
                            "teacher_action": [u, v], "teacher_next_loss": best,
                            "perturbations": perturbations})
        target = output / f"{split}.jsonl"
        target.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                  for row in records))
        files[target.name] = sha256(target)
    manifest = {"version": 1, "seed": SEED, "source_sha256": SOURCE_SHA256,
                "source_url": "https://users.cecs.anu.edu.au/~bdm/data/sr261034.g6",
                "counts": selected_counts, "anchors": ANCHORS, "files": files,
                "teacher": "minimum exact one-flip excess-page loss; first lexicographic tie"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_split(directory: Path, split: str) -> list[dict]:
    if split not in COUNTS:
        raise ValueError("invalid split")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") != 1 or manifest.get("source_sha256") != SOURCE_SHA256 or \
            manifest.get("anchors") != {key: list(value) for key, value in ANCHORS.items()} or \
            set(manifest.get("files", {})) != {f"{part}.jsonl" for part in COUNTS}:
        raise ValueError("invalid manifest")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("frozen dataset changed")
    rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]
    if len(rows) != manifest["counts"][split]:
        raise ValueError("count mismatch")
    for index, row in enumerate(rows):
        if row["id"] != f"{split}-{index:04d}" or row["split"] != split or \
                row["anchor_id"] not in ANCHORS[split]:
            raise ValueError("split leakage or row order mismatch")
        rows_int = [int(word, 16) for word in row["rows_hex"]]
        if len(rows_int) != N or any(rows_int[u] >> u & 1 for u in range(N)) or \
                any((rows_int[u] >> v & 1) != (rows_int[v] >> u & 1) for u, v in PAIRS):
            raise ValueError("invalid adjacency encoding")
        if score(rows_int) != (row["loss"], row["red_violations"], row["blue_violations"]):
            raise ValueError("incorrect exact feedback")
        action = row["teacher_action"]
        if type(action) is not list or len(action) != 2 or any(type(x) is not int for x in action) or \
                not 0 <= action[0] < action[1] < N:
            raise ValueError("invalid teacher action")
        flip(rows_int, *action)
        if score(rows_int)[0] != row["teacher_next_loss"]:
            raise ValueError("incorrect teacher score")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
