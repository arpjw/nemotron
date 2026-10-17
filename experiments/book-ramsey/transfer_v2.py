"""Versioned out-of-distribution test for the frozen B4/B9 edit adapter.

The SRG(25,12,5,6) records are independent of the v1 SRG(26,10,3,4)
collection. They are valid B6/B6 colorings because each color has five
common neighbors at every edge of that color.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from book import decode_graph6, verify

N = 25
RED_BOOK = BLUE_BOOK = 6
SOURCE_URL = "https://users.cecs.anu.edu.au/~bdm/data/sr251256.g6"
SOURCE_SHA256 = "f55f86eb48cd61c3315c62341cc3a8c59289c58504edb3ab859d7ef2068b9aa4"
SEED = 2026101602
COUNTS = {"dev": 150, "test": 150}
ANCHORS = {"dev": tuple(range(8)), "test": tuple(range(8, 15))}
PAIRS = tuple((u, v) for u in range(N) for v in range(u + 1, N))
FULL = (1 << N) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flip(rows: list[int], u: int, v: int) -> None:
    if not 0 <= u < v < N:
        raise ValueError("invalid edge")
    rows[u] ^= 1 << v
    rows[v] ^= 1 << u


def edge_list(rows: list[int]) -> list[list[int]]:
    return [[u, v] for u, v in PAIRS if rows[u] >> v & 1]


def score(rows: list[int]) -> tuple[int, int, int]:
    loss = red = blue = 0
    for u, v in PAIRS:
        if rows[u] >> v & 1:
            excess = max(0, (rows[u] & rows[v]).bit_count() - RED_BOOK + 1)
            red += excess > 0
        else:
            pages = (FULL & ~(rows[u] | rows[v]) & ~(1 << u) & ~(1 << v)).bit_count()
            excess = max(0, pages - BLUE_BOOK + 1)
            blue += excess > 0
        loss += excess
    return loss, red, blue


def spines(rows: list[int]) -> list[tuple[int, int, int, str]]:
    result = []
    for u, v in PAIRS:
        if rows[u] >> v & 1:
            excess = max(0, (rows[u] & rows[v]).bit_count() - RED_BOOK + 1)
            color = "R"
        else:
            pages = (FULL & ~(rows[u] | rows[v]) & ~(1 << u) & ~(1 << v)).bit_count()
            excess = max(0, pages - BLUE_BOOK + 1)
            color = "B"
        if excess:
            result.append((excess, u, v, color))
    return sorted(result, key=lambda item: (-item[0], item[1], item[2]))


def prompt(row: dict) -> str:
    rows = [int(word, 16) for word in row["rows_hex"]]
    shown = " ".join(f"{color}{u}-{v}:{excess}" for excess, u, v, color in spines(rows)[:12])
    words = " ".join(row["rows_hex"])
    return ("Repair a red/blue coloring of K25. Red edges are the 1 bits in 25 adjacency rows, "
            "listed in vertex order as seven-digit hexadecimal integers; bit v of row u is edge (u,v). "
            "The rows are symmetric with zero diagonal. Blue edges are the 0 bits off diagonal. "
            "Avoid red B6 and blue B6, where Bk is k triangles sharing an edge. "
            f"Current exact excess-page loss is {row['loss']}; violating red spines "
            f"{row['red_violations']}, blue spines {row['blue_violations']}. "
            f"The twelve largest violating spines are color/u-v/excess: {shown}. Rows: {words}. "
            "Choose one edge to flip. Reply only JSON {\"u\":integer,\"v\":integer}, with 0<=u<v<25.")


def load_anchors(source: Path) -> list[list[int]]:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("SRG(25,12,5,6) source checksum mismatch")
    decoded = [decode_graph6(line) for line in source.read_text().splitlines() if line.strip()]
    if len(decoded) != 15 or any(n != N or not verify(n, edges, RED_BOOK, BLUE_BOOK)["valid"]
                                 for n, edges in decoded):
        raise ValueError("expected fifteen valid 25-vertex graphs")
    result = []
    for _, edges in decoded:
        rows = [0] * N
        for u, v in edges:
            flip(rows, u, v)
        result.append(rows)
    return result


def relabel(rows: list[int], rng: random.Random) -> list[int]:
    permutation = list(range(N))
    rng.shuffle(permutation)
    result = [0] * N
    for u, v in PAIRS:
        if rows[u] >> v & 1:
            flip(result, *sorted((permutation[u], permutation[v])))
    return result


def generate(source: Path, output: Path, *, counts: dict[str, int] | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    anchors = load_anchors(source)
    selected = counts if counts is not None else COUNTS
    if set(selected) != set(COUNTS) or any(type(value) is not int or value < 1
                                          for value in selected.values()):
        raise ValueError("positive dev/test counts required")
    output.mkdir(parents=True)
    files = {}
    for split, count in selected.items():
        rng = random.Random(SEED + (0 if split == "dev" else 1))
        seen: set[tuple[int, ...]] = set()
        records = []
        for index in range(count):
            for _ in range(100):
                anchor_id = ANCHORS[split][index % len(ANCHORS[split])]
                rows = relabel(anchors[anchor_id], rng)
                perturbations = rng.randint(3, 8)
                for u, v in rng.sample(PAIRS, perturbations):
                    flip(rows, u, v)
                current = score(rows)
                if current[0] > 0 and tuple(rows) not in seen:
                    break
            else:
                raise RuntimeError("could not generate distinct invalid start")
            seen.add(tuple(rows))
            best = None
            action = None
            for u, v in PAIRS:
                flip(rows, u, v)
                candidate = score(rows)[0]
                flip(rows, u, v)
                if best is None or candidate < best:
                    best, action = candidate, [u, v]
            records.append({"id": f"{split}-{index:04d}", "split": split,
                            "anchor_id": anchor_id, "rows_hex": [f"{row:07x}" for row in rows],
                            "loss": current[0], "red_violations": current[1],
                            "blue_violations": current[2], "teacher_action": action,
                            "teacher_next_loss": best, "perturbations": perturbations})
        target = output / f"{split}.jsonl"
        target.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                  for row in records))
        files[target.name] = sha256(target)
    manifest = {"version": 2, "seed": SEED, "source_url": SOURCE_URL,
                "source_sha256": SOURCE_SHA256, "n": N, "red_book": RED_BOOK,
                "blue_book": BLUE_BOOK, "counts": selected, "anchors": ANCHORS, "files": files,
                "teacher": "minimum exact one-flip excess-page loss; first lexicographic tie"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_split(directory: Path, split: str) -> list[dict]:
    if split not in COUNTS:
        raise ValueError("invalid split")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") != 2 or manifest.get("source_sha256") != SOURCE_SHA256 or \
            manifest.get("n") != N or manifest.get("red_book") != RED_BOOK or \
            manifest.get("blue_book") != BLUE_BOOK or \
            manifest.get("anchors") != {k: list(v) for k, v in ANCHORS.items()}:
        raise ValueError("invalid manifest")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("frozen transfer data changed")
    rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]
    if len(rows) != manifest["counts"][split]:
        raise ValueError("count mismatch")
    for index, row in enumerate(rows):
        if row["id"] != f"{split}-{index:04d}" or row["split"] != split or \
                row["anchor_id"] not in ANCHORS[split]:
            raise ValueError("split leakage")
        adjacency = [int(word, 16) for word in row["rows_hex"]]
        if len(adjacency) != N or any(value < 0 or value >= 1 << N or value >> u & 1
                                      for u, value in enumerate(adjacency)) or \
                any((adjacency[u] >> v & 1) != (adjacency[v] >> u & 1) for u, v in PAIRS):
            raise ValueError("invalid adjacency")
        if score(adjacency) != (row["loss"], row["red_violations"], row["blue_violations"]):
            raise ValueError("incorrect score")
        action = row["teacher_action"]
        if not isinstance(action, list) or len(action) != 2 or \
                any(type(vertex) is not int for vertex in action) or \
                not 0 <= action[0] < action[1] < N:
            raise ValueError("invalid teacher action")
        flip(adjacency, *action)
        if score(adjacency)[0] != row["teacher_next_loss"]:
            raise ValueError("incorrect teacher loss")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
