"""API contract tests (in-process, no network) + LLM provider adapters."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from engrag.api.server import create_app
from engrag.config import Settings
from engrag.demo_corpus import generate_all
from engrag.llm.providers import (AnthropicProvider, ChatMessage,
                                   OpenAICompatibleProvider, build_provider,
                                   friendly_http_error)


@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", max_upload_mb=1)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------- basics
def test_health_and_stats(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "fts5" in body
    r = client.get("/api/v1/stats")
    assert r.status_code == 200
    assert r.json()["documents"] == 0


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/v1/documents",
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_too_large(client):
    big = b"%PDF-1.7\n" + b"0" * (1_400_000)      # limit is 1 MB in this fixture
    r = client.post("/api/v1/documents",
                    files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413


def test_upload_dupe_conflict_and_flow(client, tmp_path):
    docs = generate_all(tmp_path)
    path = docs[0].path
    data = path.read_bytes()
    r = client.post("/api/v1/documents",
                    files={"file": (path.name, data, "application/pdf")},
                    data={"title": docs[0].title})
    assert r.status_code == 200, r.text
    doc_id = r.json()["document_id"]
    # duplicate upload -> 409 with the existing document id
    r2 = client.post("/api/v1/documents",
                     files={"file": (path.name, data, "application/pdf")})
    assert r2.status_code == 409
    assert r2.json()["document_id"] == doc_id
    assert "already stored" in r2.json()["message"].lower()
    # document shows up with pages, chunks, values
    r3 = client.get(f"/api/v1/documents/{doc_id}")
    assert r3.status_code == 200
    info = r3.json()
    assert info["page_count"] >= 3 and info["chunk_count"] >= 3
    # the API never lets a caller self-label a file as synthetic demo data —
    # that flag is set by the bundled demo builder only
    assert info["is_synthetic"] == 0 and info["status"] == "ready"
    r4 = client.get(f"/api/v1/documents/{doc_id}/pages/1")
    assert r4.status_code == 200
    assert "rated" in r4.json()["text"].lower()
    assert r4.json()["page_number"] == 1
    # search reports the score kind honestly (composite, not a probability)
    sr = client.post("/api/v1/search", json={"query": "rated power"}).json()
    assert "not a calibrated probability" in sr["score_kind"]
    r5 = client.get(f"/api/v1/documents/{doc_id}/values")
    assert r5.status_code == 200
    assert any(str(v.get("unit_label", v.get("unit", ""))).startswith("kV")
               for v in r5.json())
    # search + ask work over HTTP
    r6 = client.post("/api/v1/search", json={"query": "rated voltage",
                                             "document_ids": [doc_id],
                                             "top_k": 5})
    assert r6.status_code == 200 and r6.json()["results"]
    r7 = client.post("/api/v1/ask",
                     json={"question": "What are the rated power and frequency "
                                       "of DEMO-TR-01?",
                           "document_ids": [doc_id]})
    assert r7.status_code == 200
    body = r7.json()
    assert body["mode"] == "extractive"
    assert not body["abstained"]
    assert body["evidence"] and body["verification"]["citations"]
    # every citation whose integrity passed must be findable on its physical
    # page (tables are part of the page: full_text = body + tables)
    norm = lambda t: " ".join(t.split()).lower()
    checked = 0
    for cite in body["verification"]["citations"]:
        if cite["integrity"] != "ok" or not cite["quote"]:
            continue
        pr = client.get(f"/api/v1/documents/{cite['document_id']}"
                        f"/pages/{cite['page_number']}").json()
        assert norm(cite["quote"]) in norm(pr["full_text"]), cite["quote"][:80]
        checked += 1
    assert checked >= 1
    pr1 = client.get(f"/api/v1/documents/{doc_id}/pages/1").json()
    assert "TABLE (page 1):" in pr1["tables_text"]
    # history recorded and retrievable
    r8 = client.get("/api/v1/history")
    assert r8.status_code == 200 and r8.json()
    qa_id = r8.json()[0]["id"]
    r9 = client.get(f"/api/v1/history/{qa_id}")
    assert r9.status_code == 200
    hist = r9.json()
    assert hist["citations"] and hist["mode"] == "extractive"
    assert any(c["integrity"] == "ok" for c in hist["citations"])
    assert all(c["page_number"] >= 1 for c in hist["citations"])
    # delete -> gone, 404 on fetch
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 200
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404


def test_verify_endpoint_flags_wrong_and_confirms_right(client, tmp_path):
    docs = generate_all(tmp_path)
    data = docs[0].path.read_bytes()
    r = client.post("/api/v1/documents",
                    files={"file": (docs[0].path.name, data, "application/pdf")})
    doc_id = r.json()["document_id"]
    # true statement (same unit family, exact numbers) -> verified
    ok = client.post("/api/v1/verify", json={
        "text": "The rated power of DEMO-TR-01 is 40 MVA, i.e. 40000 kVA.",
        "document_ids": [doc_id]})
    assert ok.status_code == 200
    st = {c["status"] for c in ok.json()["report"]["numeric_checks"]}
    assert st & {"verified", "unit_verified"}, ok.json()["report"]["numeric_checks"]
    # false statement -> mismatch (never silently accepted)
    bad = client.post("/api/v1/verify", json={
        "text": "The rated power of DEMO-TR-01 is 96 MVA.",
        "document_ids": [doc_id]})
    assert {c["status"] for c in bad.json()["report"]["numeric_checks"]} == {"mismatch"}


def test_ask_validation(client):
    # whitespace-only is rejected by the endpoint (400), too short by the
    # request model (422) — either way no pipeline run, no invented answer
    assert client.post("/api/v1/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/v1/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/v1/ask",
                       json={"question": "rated power",
                             "top_k": 999}).status_code == 422


def test_provider_settings_masks_key(client):
    r = client.post("/api/v1/providers", json={
        "kind": "openai", "base_url": "https://api.example.invalid/v1",
        "api_key": "sk-should-never-appear-123", "model": "gpt-4o-mini"})
    assert r.status_code == 200
    body = r.json()
    assert "sk-should-never-appear-123" not in json.dumps(body)
    r2 = client.get("/api/v1/providers")
    assert r2.status_code == 200
    assert "sk-should-never-appear-123" not in r2.text
    assert r2.json()["active"]["api_key"] == "set"


def test_render_page_png(client, tmp_path):
    docs = generate_all(tmp_path)
    r = client.post("/api/v1/documents",
                    files={"file": (docs[0].path.name,
                                    docs[0].path.read_bytes(),
                                    "application/pdf")})
    doc_id = r.json()["document_id"]
    r2 = client.get(f"/api/v1/documents/{doc_id}/pages/1/render?scale=1.0"
                    f"&terms=rated&terms=voltage")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/png"
    assert r2.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(r2.content) > 5000


# ----------------------------------------------------------- provider units
def _provider(cfg):
    return build_provider(cfg)


def test_offline_provider_never_calls_out():
    from engrag.llm.providers import ChatMessage
    p = _provider({"kind": "offline"})
    st = p.status()
    assert st.ok and st.kind == "offline" and not st.online
    r = p.chat([ChatMessage(role="system", content="s"),
                ChatMessage(role="user", content="evidence\nQ: what?")])
    # offline mode has no model: the caller must fall back to extraction
    assert not r.ok and "Offline" in (r.error or "")


def test_openai_adapter_protocol():  # noqa: D103
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant",
                                     "content": "Answer 11 kV [1]."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    p = OpenAICompatibleProvider(base_url="https://x.test/v1", api_key="sk-1",
                                 model="gpt-4o-mini",
                                 extra={"transport": httpx.MockTransport(handler)})
    r = p.chat([ChatMessage(role="user", content="hi")])
    assert r.ok and r.text.startswith("Answer")
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer sk-1"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert r.prompt_tokens == 10 and r.completion_tokens == 5


def test_anthropic_adapter_uses_native_protocol():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_1", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Grounded answer [2]."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 4}})
    p = AnthropicProvider(base_url="https://api.anthropic.com", api_key="sk-ant",
                          model="claude-sonnet-4-5",
                          extra={"transport": httpx.MockTransport(handler)})
    r = p.chat([ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="usr")])
    assert r.ok and r.text == "Grounded answer [2]."
    assert seen["url"].endswith("/v1/messages")
    assert seen["headers"]["x-api-key"] == "sk-ant"
    assert "anthropic-version" in seen["headers"]
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["messages"][0]["role"] == "user"


def test_provider_http_errors_are_friendly_not_leaky():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})
    p = OpenAICompatibleProvider(base_url="https://x.test/v1", api_key="sk-z",
                                 model="m",
                                 extra={"transport": httpx.MockTransport(handler)})
    r = p.chat([ChatMessage(role="user", content="hi")])
    assert not r.ok and r.error
    assert "sk-z" not in (r.error or "")
    msg = friendly_http_error(429, "rate limited")
    assert "429" in msg or "rate" in msg.lower()


def test_pick_default_model_never_invents():
    from engrag.llm.providers import pick_default_model
    assert pick_default_model("openai", []) in (None, "")
    got = pick_default_model("openai-compatible", ["gpt-4o-mini", "gpt-4o"])
    assert got == "gpt-4o-mini"          # cheap tier preferred when present
    # nothing safe to guess -> None, the UI must then ask the user
    assert pick_default_model("openai-compatible", ["gpt-5-turbo-ultra"]) is None
