"""CLI: `python -m engrag <command>` (installed as console script `engrag`)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def cmd_api(args) -> int:
    import uvicorn
    from .api.server import create_app
    from .config import load_settings
    s = load_settings()
    uvicorn.run(create_app(s), host=args.host or s.host, port=args.port or s.api_port,
                log_level="info", reload=args.reload)
    return 0


def cmd_web(args) -> int:
    from .config import load_settings
    s = load_settings()
    app_file = Path(__file__).resolve().parent / "frontend" / "app.py"
    port = args.port or s.web_port
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parent.parent))
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file),
           "--server.port", str(port), "--server.address", args.host or s.host,
           "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    return subprocess.call(cmd, env=env)


def cmd_build_demo(args) -> int:
    from .config import load_settings
    from .demo_corpus import generate_all
    from .service import Service
    s = load_settings()
    out = Path(args.out) if args.out else s.data_dir / "demo_generated"
    docs = generate_all(out)
    if args.no_ingest:
        for d in docs:
            print(f"generated {d.path}")
        return 0
    svc = Service(s)
    col = svc.db.get_or_create_collection("Synthetic demo manuals",
                                          "Invented demo data — not a real standard",
                                          is_demo=True)
    for d in docs:
        res = svc.upload_pdf(filename=d.path.name, data=d.path.read_bytes(),
                             collection_id=col, title=d.title, is_synthetic=True)
        print(f"{d.path.name}: document {res.document_id} — {res.status}")
    print(svc.stats())
    return 0


def cmd_ask(args) -> int:
    from .config import load_settings
    from .service import Service
    svc = Service(load_settings())
    res = svc.ask(args.question, top_k=args.top_k)
    print(res["answer"])
    print("\n--- verification:", res["verification"].get("overall_support"))
    for n in res["verification"].get("numeric_checks", [])[:10]:
        print(f"  [{n['status']}] {n['claim']} vs {n.get('source_value')} "
              f"({n['document_title']} p.{n['page_number']})")
    return 0


def cmd_search(args) -> int:
    from .config import load_settings
    from .service import Service
    svc = Service(load_settings())
    for h in svc.search(args.query, top_k=args.top_k):
        print(f"[{h['score']:.3f}] {h['document_title']} p.{h['matched_page']} "
              f"({h['kind']}) — {h['excerpt'][:90]!r}")
    return 0


def cmd_reset(args) -> int:
    from .config import load_settings
    s = load_settings()
    if not args.yes:
        print("Refusing to delete data without --yes. Data dir:", s.data_dir)
        return 1
    import shutil
    shutil.rmtree(s.data_dir, ignore_errors=True)
    print("Removed", s.data_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="engrag",
                                 description="Engineering Document Q&A "
                                             "Assistant (offline-first RAG)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("api", help="run the FastAPI server")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(fn=cmd_api)
    p = sub.add_parser("web", help="run the Streamlit frontend")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default=None)
    p.set_defaults(fn=cmd_web)
    p = sub.add_parser("build-demo", help="generate + ingest the synthetic corpus")
    p.add_argument("--out", default=None)
    p.add_argument("--no-ingest", action="store_true")
    p.set_defaults(fn=cmd_build_demo)
    p = sub.add_parser("ask", help="ask a question from the terminal")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=6)
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("search", help="keyword/semantic search from the terminal")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=8)
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("reset", help="delete the local data directory")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_reset)
    args = ap.parse_args(argv)
    from .logging_setup import setup_logging
    setup_logging()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
