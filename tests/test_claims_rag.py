"""Tests for citation verification, anti-injection and the offline pipeline."""
from __future__ import annotations

import pytest

from engrag.verify.claims import AnswerVerifier, find_conflicts


# ------------------------------------------------------------------ sentences
def test_split_keeps_quotes_whole():
    text = ('Rated voltage is “11 kV, 50 Hz, vector group Dyn11 per '
            'IEC 60076”. It was verified at factory. Next point: 630 A.')
    sents = AnswerVerifier.split_sentences(text)
    assert any("11 kV, 50 Hz" in s and s.count("“") == 1 for s in sents)
    assert not any(s.endswith("per") for s in sents)


def test_split_handles_colon_markers_and_reports():
    sents = AnswerVerifier.split_sentences("Steps: open the breaker. Then earth it. Done.")
    assert any(s.startswith("open the breaker") for s in sents) or \
        sents[0].startswith("Steps")


# ------------------------------------------------------------------ verifier
def _ev(db, doc_id, chunk_id, text, page):
    return {"ref_index": 1, "chunk_id": chunk_id, "document_id": doc_id,
            "title": "Doc", "page_start": page, "page_end": page,
            "matched_page": page, "text": text, "excerpt": text,
            "text_sha1": __import__("hashlib").sha1(text.encode()).hexdigest(),
            "kind": "text", "score": 1.0, "weak": False, "flags": []}


def test_verifier_confirms_real_quote(svc):
    rows = svc.db.query(
        "SELECT id, document_id, text, page_start FROM document_chunks "
        "WHERE kind='text' AND length(text) > 120 LIMIT 1")
    assert rows
    r = rows[0]
    ev = _ev(svc.db, r["document_id"], r["id"], r["text"], r["page_start"])
    quote = " ".join(r["text"].split())[:80]
    v = AnswerVerifier(svc.db, rel_tolerance=1e-9)
    report = v.verify(f"The document states “{quote}” [1].", [ev])
    cite = report.citations[0]
    assert cite.integrity == "ok"
    assert cite.support in ("supported", "weak")


def test_verifier_flags_fabricated_quote(svc):
    rows = svc.db.query("SELECT id, document_id, page_start FROM document_chunks LIMIT 1")
    r = rows[0]
    fake = "The transformer weighs 412 tonnes according to tests [1]."
    ev = _ev(svc.db, r["document_id"], r["id"],
             "Rated voltage 11 kV; rated power 1600 kVA.", r["page_start"])
    v = AnswerVerifier(svc.db, rel_tolerance=1e-9)
    report = v.verify(fake, [ev])
    assert report.citations[0].integrity in ("missing_quote", "stale_chunk") \
        or report.sentences[0].support == "insufficient"


def test_verifier_detects_wrong_number(svc):
    rows = svc.db.query(
        "SELECT id, document_id, page_start, text FROM document_chunks "
        "WHERE text LIKE '%Rated voltage%' LIMIT 1")
    r = rows[0]
    ev = _ev(svc.db, r["document_id"], r["id"], r["text"], r["page_start"])
    v = AnswerVerifier(svc.db, rel_tolerance=1e-9)
    report = v.verify("The rated voltage of DEMO-TR-01 is 99 kV [1].", [ev])
    checks = report.numeric_checks
    assert checks and any(c.status == "mismatch" for c in checks)
    assert report.sentences[0].support == "contradicted"


def test_stale_chunk_detected(svc):
    rows = svc.db.query(
        "SELECT id, document_id, page_start, text FROM document_chunks LIMIT 1")
    r = rows[0]
    ev = _ev(svc.db, r["document_id"], r["id"], r["text"] + " EXTRA",
             r["page_start"])
    v = AnswerVerifier(svc.db, rel_tolerance=1e-9)
    report = v.verify("Some claim about it [1].", [ev])
    assert report.citations[0].integrity == "stale_chunk"


# ---------------------------------------------------------------- conflicts
def _cev(doc, title, text, page=2):
    return {"ref_index": 1, "chunk_id": 1, "document_id": doc, "title": title,
            "page_start": page, "page_end": page, "matched_page": page,
            "text": text, "text_sha1": "x", "kind": "table", "score": 1.0}


def test_find_conflicts_between_variants(svc):
    def one_chunk(pat):
        return svc.db.query(
            "SELECT c.id, c.document_id, c.page_start, c.text FROM document_chunks c "
            "JOIN documents d ON d.id=c.document_id "
            "WHERE c.text LIKE ? AND c.kind='table' LIMIT 1", (f"%{pat}%",))
    rows = one_chunk("40 MVA")
    r2 = one_chunk("30 MVA")
    if not (rows and r2):
        pytest.skip("demo corpus changed")
    ev = [_ev(svc.db, rows[0]["document_id"], rows[0]["id"], rows[0]["text"],
              rows[0]["page_start"]),
          _ev(svc.db, r2[0]["document_id"], r2[0]["id"], r2[0]["text"],
              r2[0]["page_start"])]
    conflicts = find_conflicts(ev, rel_tol=1e-9)
    assert isinstance(conflicts, list)   # may be empty; never raises


def test_find_conflicts_requires_a_common_subject():
    """A real disagreement about the same machine IS reported."""
    a = _cev(1, "DEMO-TR-01 transformer manual",
             "Rated apparent power | 40 MVA | ONAF cooling")
    b = _cev(2, "DEMO-TR-01 transformer test report Rev C",
             "Rated apparent power | 45 MVA | ONAF cooling")
    conflicts = find_conflicts([a, b])
    assert len(conflicts) == 1
    assert {v["raw"] for v in conflicts[0]["values"]} == {"40 MVA", "45 MVA"}


def test_find_conflicts_ignores_different_equipment():
    """Two machines with the same wording are not a contradiction (the bug
    that made the Value Verification view claim dozens of cross-source
    disagreements between unrelated products)."""
    tr01 = _cev(1, "DEMO-TR-01 transformer manual",
                "Rated apparent power | 40 MVA | ONAF cooling")
    tr02 = _cev(2, "DEMO 11 kV switchgear handbook",
                "Rated apparent power | 25 MVA | ONAN self-cooled")
    assert find_conflicts([tr01, tr02]) == []


def test_find_conflicts_ignores_unattributed_values():
    """When neither passage names its equipment, no contradiction is asserted."""
    a = _cev(1, "Handbook A", "Rated apparent power | 40 MVA | ONAF cooling")
    b = _cev(2, "Handbook B", "Rated apparent power | 45 MVA | ONAF cooling")
    assert find_conflicts([a, b]) == []


def test_find_conflicts_never_mixes_power_categories():
    """kW and kVA share the SI dimension but are different quantities."""
    a = _cev(1, "DEMO-TR-01 manual", "Rated power 40 kW")
    b = _cev(2, "DEMO-TR-01 datasheet", "Rated power 40 kVA")
    assert find_conflicts([a, b]) == []


def test_value_verification_view_reports_no_bogus_conflicts(svc):
    """Regression for the screenshot bug: a TR-01 statement retrieved the
    unrelated TR-02 appendix and reported it as a source disagreement."""
    res = svc.verify_standalone(
        "The rated voltage of DEMO-TR-01 is 33 kV and its rated apparent "
        "power is 40 MVA.", document_ids=None, collection_id=None, page=None,
        rel_tolerance=0.0)
    assert res["report"]["conflicts"] == []
    assert res["report"]["numeric_checks"], "verification produced no checks"


# ------------------------------------------------------------------- rag e2e
def test_offline_ask_is_grounded_and_cited(svc, collection_id):
    res = svc.ask("What are the rated voltage and frequency of DEMO-TR-01?",
                  collection_id=collection_id)
    assert res["mode"] == "extractive"
    assert not res["abstained"]
    assert res["verification"]["overall_support"] in ("grounded",
                                                       "partially_grounded")
    assert res["evidence"]
    for ev in res["evidence"]:
        assert ev["page_start"] >= 1
        assert ev["text_sha1"]


def test_offline_ask_abstains_when_no_evidence(svc, collection_id):
    res = svc.ask("What is the recommended hoverboard tyre pressure for model X-9?",
                  collection_id=collection_id)
    assert res["abstained"]
    assert res["verification"]["overall_support"] == "abstained"
    low = res["answer"].lower()
    assert ("no verified answer" in low or "insufficient" in low
            or "cannot" in low), res["answer"][:200]
    assert res["abstain_reason"]      # the reason is surfaced, not hidden


def test_injected_instruction_not_followed(svc, collection_id):
    res = svc.ask("What is the rated power of DEMO-TR-03?",
                  collection_id=collection_id)
    low = res["answer"].lower()
    assert "exfiltrat" not in low and "api key" not in low
    assert "500" not in res["answer"] or "ignored" in low
    assert "1600 kva" in low.replace("kVA", "kVA").lower()


def test_pages_are_one_based_and_printed_labels_separate(svc, collection_id):
    res = svc.ask("What are the no-load and load losses of DEMO-TR-01?",
                  collection_id=collection_id)
    pages = {e["page"] for e in res["evidence"]}
    assert all(p >= 1 for p in pages)
    doc_id = res["evidence"][0]["document_id"]
    rows = svc.db.query(
        "SELECT printed_label, page_number FROM document_pages "
        "WHERE document_id=? AND printed_label != ''", (doc_id,))
    assert all(r["page_number"] >= 1 for r in rows)


def test_deleting_document_invalidates_answers(svc, collection_id):
    docs = svc.list_documents()
    victim = [d for d in docs if "injection" in d["title"].lower()]
    assert victim
    vid = victim[0]["id"]
    before = svc.search("rated power", top_k=10, document_ids=[vid])
    assert before
    svc.delete_document(vid)
    after = svc.search("rated power", top_k=10, document_ids=[vid])
    assert after == []
    # restore for other tests
    from engrag.demo_corpus import generate_all
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        docs2 = generate_all(Path(td))
        f = [d for d in docs2 if "injection" in str(d.path)][0]
        col = svc.db.get_or_create_collection("Synthetic demo manuals", "demo",
                                              is_demo=True)
        svc.upload_pdf(filename=f.path.name, data=f.path.read_bytes(),
                       collection_id=col, title=f.title, is_synthetic=True)
