#!/usr/bin/env python3
"""CI gate for an evaluation results file: fails on any regression.

    python scripts/ci_check_eval.py data/eval_results.json [--min 0.99]

Exits 0 only if every case passed *and* every rate metric is at or above ``--min``.
Missing keys are an error rather than a silent pass: if the report shape ever changes,
the gate must shout instead of comparing nothing.  Kept as a script (not inline YAML) so
it can be run — and tested — locally.
"""
from __future__ import annotations

import argparse
import json
import sys

RATE_METRICS = ("recall@k", "citation_accuracy", "answer_support_rate",
                "abstention_recall", "abstention_precision", "numeric_status_accuracy")


def check(report: dict, minimum: float) -> tuple[list[str], list[str]]:
    """Return ``(lines, failures)`` for one evaluation report."""
    lines: list[str] = []
    failures: list[str] = []
    for key in ("summary", "cases", "engine"):
        if key not in report:
            failures.append(f"report has no {key!r} section — wrong file or changed schema")
    if failures:
        return lines, failures

    summary, cases = report["summary"], report["cases"]
    n_total, n_pass = summary["cases"], summary["passed"]
    lines.append(f"cases              : {n_pass}/{n_total}")
    if n_pass != n_total:
        for case in cases:
            if not case["pass"]:
                failures.append(f"case {case['id']} ({case['kind']}) failed")
    for key in RATE_METRICS:
        if key not in summary:
            failures.append(f"summary is missing the {key!r} metric")
            continue
        value = float(summary[key])
        lines.append(f"{key:20s}: {value:.1%}")
        if value < minimum:
            failures.append(f"{key} = {value:.1%} is below {minimum:.0%}")
    if "median_latency_ms" in summary:
        lines.append(f"latency med/p95 ms  : {summary['median_latency_ms']:.0f} / "
                     f"{summary.get('p95_latency_ms', float('nan')):.0f}")
    engine = report["engine"]
    lines.append(f"engine             : {engine.get('embedding_backend')} · "
                 f"fts5={engine.get('fts5')} · {engine.get('mode')}")
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", default="data/eval_results.json")
    ap.add_argument("--min", type=float, default=0.99, dest="minimum",
                    help="required value for every rate metric (default 0.99)")
    args = ap.parse_args(argv)

    try:
        with open(args.path, encoding="utf-8") as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print(f"FAIL: {args.path} does not exist — run scripts/run_eval.py first", file=sys.stderr)
        return 2

    lines, failures = check(report, args.minimum)
    print("\n".join(lines))
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"\nOK: all cases passed, every rate metric >= {args.minimum:.0%}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
