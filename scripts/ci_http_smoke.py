#!/usr/bin/env python3
"""End-to-end smoke test of the running FastAPI service *over real HTTP*.

The pytest suite drives the app with ``TestClient`` (in-process). This script
is the network counterpart: it talks to a uvicorn process that is already
listening, so routing, serialization, multipart upload and response codes are
exercised the way a real client sees them.

    python scripts/ci_http_smoke.py --url http://127.0.0.1:8502

Exits non-zero on the first failed assertion and prints what it checked.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx


def wait_for_health(base: str, timeout_s: int = 60) -> dict:
    deadline = time.monotonic() + timeout_s
    last: Exception | str = "no attempt"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/api/v1/health", timeout=5)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:                       # not up yet
            last = exc
        time.sleep(0.5)
    raise SystemExit(f"service at {base} never became healthy ({last})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8502")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    checks: list[str] = []

    health = wait_for_health(base)
    assert health["status"] == "ok", health
    checks.append(f"health ok (fts5={health['fts5']}, version={health['version']})")

    with httpx.Client(base_url=base, timeout=120) as c:
        # Build + ingest the synthetic corpus through the API itself.
        r = c.post("/api/v1/demo-corpus")
        assert r.status_code == 200, (r.status_code, r.text[:300])
        created = r.json()["created"]
        ready = [d for d in created if d["status"] in ("ok", "duplicate")]
        assert len(ready) == 5, created
        checks.append(f"demo-corpus ingested {len(ready)} documents")

        r = c.get("/api/v1/documents")
        assert r.status_code == 200 and len(r.json()) == 5, r.text[:300]
        docs = r.json()
        checks.append(f"documents listed: {len(docs)}")

        doc = next(d for d in docs if "TR-01" in d["title"])
        doc_id = doc["id"]

        r = c.get(f"/api/v1/documents/{doc_id}")
        assert r.status_code == 200, r.text[:300]
        meta = r.json()
        assert meta["page_count"] >= 1 and meta["pages_meta"], meta
        checks.append(f"document metadata: {meta['page_count']} pages, "
                      f"{meta['chunk_count']} chunks")

        r = c.get(f"/api/v1/documents/{doc_id}/pages/1")
        assert r.status_code == 200, r.text[:300]
        page = r.json()
        assert page["text"].strip(), "page 1 extraction is empty"
        checks.append(f"page 1 text: {len(page['text'])} chars, "
                      f"printed label {page['printed_label']!r}")

        r = c.get(f"/api/v1/documents/{doc_id}/pages/1/render")
        assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n", \
            f"render returned {r.status_code}"
        checks.append(f"page render: {len(r.content)} byte PNG")

        r = c.post("/api/v1/ask", json={"question":
                   "What are the rated voltage and frequency of DEMO-TR-01?"})
        assert r.status_code == 200, (r.status_code, r.text[:300])
        ans = r.json()
        assert not ans["abstained"], ans
        assert ans["evidence"] and ans["verification"]["citations"], ans
        assert "11 kV" in ans["answer"] or "132/33 kV" in ans["answer"], \
            ans["answer"][:300]
        checks.append(f"ask: mode={ans['mode']} support="
                      f"{ans['verification']['overall_support']} "
                      f"citations={len(ans['verification']['citations'])}")

        r = c.post("/api/v1/ask", json={"question":
                   "What wind-speed rating does the DEMO hoverboard X-9 have?"})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["abstained"], "out-of-corpus question was answered"
        checks.append("ask: out-of-corpus question correctly abstained")

        r = c.get("/api/v1/history")
        assert r.status_code == 200 and len(r.json()) >= 2, r.text[:300]
        checks.append(f"history rows: {len(r.json())}")

    print("HTTP smoke test passed:")
    for line in checks:
        print(f"  ✔ {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
