"""Bounded SAT encoding of the two-color book avoidance constraints."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import threading
import time

from book import decode_graph6, verify
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver


def build(n: int, red_book: int, blue_book: int, symmetry_break: bool = True) -> tuple[CNF, IDPool, dict[tuple[int, int], int]]:
    pool = IDPool()
    edge = {(u, v): pool.id(("edge", u, v)) for u in range(n) for v in range(u + 1, n)}
    cnf = CNF()
    red_pages: dict[tuple[int, int], list[int]] = {pair: [] for pair in edge}
    blue_pages: dict[tuple[int, int], list[int]] = {pair: [] for pair in edge}
    for u, v, w in itertools.combinations(range(n), 3):
        e = [edge[(u, v)], edge[(u, w)], edge[(v, w)]]
        red = pool.id(("red_triangle", u, v, w))
        blue = pool.id(("blue_triangle", u, v, w))
        for bit in e:
            cnf.append([-red, bit])
            cnf.append([-blue, -bit])
        cnf.append([red] + [-bit for bit in e])
        cnf.append([blue] + e)
        for pair in ((u, v), (u, w), (v, w)):
            red_pages[pair].append(red)
            blue_pages[pair].append(blue)
    for pair in edge:
        cnf.extend(CardEnc.atmost(red_pages[pair], bound=red_book - 1,
                                  vpool=pool, encoding=EncType.seqcounter).clauses)
        cnf.extend(CardEnc.atmost(blue_pages[pair], bound=blue_book - 1,
                                  vpool=pool, encoding=EncType.seqcounter).clauses)
    # If the all-blue graph is forbidden, some red edge exists and can be relabeled 0--1.
    if symmetry_break and n >= blue_book + 2:
        cnf.append([edge[(0, 1)]])
    if n == 27 and red_book == 4 and blue_book == 9:
        # Proven by triangle counting plus parity of each red neighborhood.
        variables = list(edge.values())
        cnf.extend(CardEnc.atleast(variables, bound=157,
                                   vpool=pool, encoding=EncType.seqcounter).clauses)
        cnf.extend(CardEnc.atmost(variables, bound=171,
                                  vpool=pool, encoding=EncType.seqcounter).clauses)
    return cnf, pool, edge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=27)
    parser.add_argument("--red-book", type=int, default=4)
    parser.add_argument("--blue-book", type=int, default=9)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--solver", default="glucose4")
    parser.add_argument("--known-graph6", type=Path)
    parser.add_argument("--output", type=Path, default=Path("search-results/sat-certificate.json"))
    args = parser.parse_args()
    if not 3 <= args.n <= 32 or min(args.red_book, args.blue_book) < 1 or args.seconds <= 0:
        parser.error("invalid size, book, or time bound")
    begin = time.monotonic()
    cnf, pool, edge = build(args.n, args.red_book, args.blue_book)
    assumptions: list[int] = []
    if args.known_graph6 is not None:
        graph_n, known_edges = decode_graph6(args.known_graph6.read_text().splitlines()[0])
        if graph_n != args.n:
            parser.error("known graph has a different order")
        red = {tuple(pair) for pair in known_edges}
        assumptions = [variable if pair in red else -variable for pair, variable in edge.items()]
    build_seconds = time.monotonic() - begin
    remaining = max(0.01, args.seconds - build_seconds)
    with Solver(name=args.solver, bootstrap_with=cnf.clauses) as solver:
        timer = threading.Timer(remaining, solver.interrupt)
        timer.start()
        try:
            result = solver.solve_limited(assumptions=assumptions, expect_interrupt=True)
        finally:
            timer.cancel()
        elapsed = time.monotonic() - begin
        summary: dict[str, object] = {
            "n": args.n, "red_book": args.red_book, "blue_book": args.blue_book,
            "solver": args.solver, "result": "sat" if result is True else "unsat" if result is False else "timeout",
            "variables": pool.top, "clauses": len(cnf.clauses),
            "build_seconds": round(build_seconds, 3),
            "elapsed_seconds": round(elapsed, 3),
            "stats": solver.accum_stats(),
            "known_graph_fixed": args.known_graph6 is not None,
        }
        if result is True:
            model = set(solver.get_model())
            red_edges = [list(pair) for pair, variable in edge.items() if variable in model]
            exact = verify(args.n, red_edges, args.red_book, args.blue_book)
            if not exact["valid"]:
                raise AssertionError("SAT assignment fails independent checker")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"n": args.n, "red_edges": red_edges}, sort_keys=True) + "\n")
            summary["certificate"] = str(args.output)
            summary["exact_check"] = exact
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
