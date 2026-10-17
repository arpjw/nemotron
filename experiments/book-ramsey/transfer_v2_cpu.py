"""Run frozen exact one-action and five-pass CPU controls without a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from edit_model import write_json
from transfer_v2 import load_split, sha256
from transfer_v2_evaluate import controls, greedy_repair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    rows = load_split(args.data, args.split)
    result = {"version": 2, "split": args.split,
              "dataset_sha256": sha256(args.data / f"{args.split}.jsonl"),
              **controls(rows), "greedy_five_pass": greedy_repair(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({name: {key: value for key, value in item.items() if key != "records"}
                      for name, item in result.items() if isinstance(item, dict)}, sort_keys=True))


if __name__ == "__main__":
    main()
