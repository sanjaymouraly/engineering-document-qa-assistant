#!/usr/bin/env python3
"""Smoke test for the Streamlit frontend.

The pytest suite checks the UI *statically* (AST) because starting Streamlit in
a unit test is slow and flaky. This script is the live counterpart:

  1. importing ``engrag.frontend.app`` must NOT start the app (a module-level
     ``main()`` call would make the import-check impossible);
  2. a served Streamlit process must answer its health probe and serve the
     app's own HTML shell.

Usage (the ``--serve`` mode starts Streamlit itself):

    python scripts/ci_streamlit_smoke.py --serve --port 8501
    python scripts/ci_streamlit_smoke.py --url http://127.0.0.1:8501
"""
from __future__ import annotations

import argparse
import importlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def check_import_does_not_launch() -> str:
    """Import the UI module in a child interpreter; it must print IMPORT_OK."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import engrag.frontend.app\n"
        "print('IMPORT_OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120, cwd=str(SRC),
                         env={**os.environ, "PYTHONPATH": str(SRC)})
    assert res.returncode == 0, res.stderr[-2000:]
    assert "IMPORT_OK" in res.stdout, res.stdout + res.stderr
    assert "streamlit run" not in res.stderr, res.stderr[-1000:]
    return "importing engrag.frontend.app does not launch the app"


def free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    app_file = SRC / "engrag" / "frontend" / "app.py"
    data_dir = os.environ.get("ENGRAG_DATA_DIR", str(ROOT / "data"))
    env = {**os.environ, "PYTHONPATH": str(SRC), "ENGRAG_DATA_DIR": data_dir}
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app_file),
         "--server.port", str(port), "--server.address", "127.0.0.1",
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_healthy(url: str, proc: subprocess.Popen, timeout_s: int = 120) -> str:
    deadline = time.monotonic() + timeout_s
    last = "no attempt"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise SystemExit(f"streamlit exited early (rc={proc.returncode}):\n{out[-2000:]}")
        try:
            r = httpx.get(f"{url}/_stcore/health", timeout=5)
            if r.status_code == 200 and r.text.strip().lower() == "ok":
                r2 = httpx.get(url, timeout=10)
                assert r2.status_code == 200 and "<html" in r2.text.lower(), \
                    "the app shell did not serve HTML"
                return "served app answers /_stcore/health and serves the app shell"
        except Exception as exc:
            last = exc
        time.sleep(0.5)
    raise SystemExit(f"streamlit at {url} never became healthy ({last})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--serve", action="store_true",
                    help="start Streamlit yourself (otherwise --url is required)")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()

    checks = [check_import_does_not_launch()]

    if args.serve:
        importlib.import_module("engrag")            # fail fast if broken
        port = free_port(args.port)
        url = f"http://127.0.0.1:{port}"
        proc = start_server(port)
        try:
            checks.append(wait_healthy(url, proc))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    else:
        if not args.url:
            raise SystemExit("pass --serve or --url")
        checks.append(wait_healthy(args.url.rstrip("/"), _NullProc()))

    print("Streamlit smoke test passed:")
    for line in checks:
        print(f"  ✔ {line}")
    return 0


class _NullProc:
    def poll(self):
        return None


if __name__ == "__main__":
    sys.exit(main())
