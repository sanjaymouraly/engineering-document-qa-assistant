#!/usr/bin/env python3
"""Run the evaluation suite against the synthetic demo corpus.

Usage:
    python scripts/run_eval.py [--data-dir data] [--json-out data/eval_results.json]
Everything is measured live (offline, deterministic); the JSON report can be
committed to docs/ for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engrag.config import load_settings          # noqa: E402
from engrag.eval import run_evaluation            # noqa: E402
from engrag.service import Service                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="data dir with existing index (default: fresh temp DB)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    import os
    if args.data_dir:
        os.environ["ENGRAG_DATA_DIR"] = str(Path(args.data_dir))
        settings = load_settings()
        svc = Service(settings)
        from engrag.demo_corpus import generate_all
        generate_all(settings.data_dir / "demo_generated")   # ensure files exist
        tmp = settings.data_dir / "demo_generated"
        from engrag.eval import build_service
        need = not svc.db.list_documents()
        if need:
            svc = build_service(settings, tmp)
    else:
        with tempfile.TemporaryDirectory(prefix="engrag-eval-") as td:
            settings = load_settings({"ENGRAG_DATA_DIR": td})
            from engrag.eval import build_service
            svc = build_service(settings, Path(td) / "gen")
            report = _run(svc, args)
            _emit(report, args)
            return 0 if report["summary"]["passed"] == report["summary"]["cases"] \
                else 1
    report = _run(svc, args)
    _emit(report, args)
    return 0 if report["summary"]["passed"] == report["summary"]["cases"] else 1


def _run(svc: Service, args) -> dict:
    report = run_evaluation(svc, top_k=args.top_k)
    report["generated_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    report["engine"] = {"embedding_backend": svc.engine.embedder.status.backend,
                        "fts5": svc.db.fts5,
                        "mode": "offline extractive + deterministic verification",
                        "corpus": "synthetic demo manuals (engrag.demo_corpus)"}
    return report


def _emit(report: dict, args) -> None:
    s = report["summary"]
    print("=" * 72)
    print("Engineering RAG evaluation — synthetic demo corpus (offline)")
    print("=" * 72)
    print(f"cases passed          : {s['passed']}/{s['cases']}")
    print(f"Recall@k (evidence)   : {s['recall@k']:.1%}")
    print(f"Citation accuracy     : {s['citation_accuracy']:.1%}")
    print(f"Answer support rate   : {s['answer_support_rate']:.1%}")
    print(f"Abstention recall     : {s['abstention_recall']:.1%}")
    print(f"Abstention precision  : {s['abstention_precision']:.1%}")
    print(f"Numeric status acc.   : {s['numeric_status_accuracy']:.1%}")
    print(f"Latency median/p95 ms : {s['median_latency_ms']:.0f} / "
          f"{s['p95_latency_ms']:.0f}")
    print(f"per-kind pass         : {s['per_kind_pass']}")
    print("-" * 72)
    for c in report["cases"]:
        mark = "PASS" if c["pass"] else "FAIL"
        extra = ""
        for k, v in c["checks"].items():
            if v not in ("ok", None) and not isinstance(v, float):
                extra += f" [{k}:{v}]"
        print(f"{mark:4s} {c['id']:32s} {c.get('mode', ''):11s}"
              f"{('lat ' + str(c.get('latency_ms')) + 'ms') if c.get('latency_ms') else '':>12s}{extra}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.json_out}")


if __name__ == "__main__":
    raise SystemExit(main())
