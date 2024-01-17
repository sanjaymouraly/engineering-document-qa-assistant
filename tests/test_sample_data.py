"""The committed demo PDFs in ``sample_data/generated`` must match the generator.

They are in the repository so a reviewer can open a synthetic manual (and upload one
without running any code), while ``engrag build-demo`` regenerates the same files into the
data directory.  Two copies of anything drift, so this test compares them.  Byte equality is
not used because ReportLab stamps a creation timestamp into every PDF; the *content* is what
has to match — every page's text and the PDF page-label dictionary that carries the printed
page numbering.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader  # noqa: E402  (pypdf is a core dependency)

from engrag.demo_corpus import DOCUMENTS, generate_all  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "sample_data" / "generated"


def _fingerprint(path: Path) -> dict:
    reader = PdfReader(str(path))
    return {
        "pages": len(reader.pages),
        "text": [p.extract_text() or "" for p in reader.pages],
        "labels": list(reader.page_labels or []),
        "title": (reader.metadata.title or "").strip() if reader.metadata else "",
    }


def test_every_document_has_a_committed_copy():
    missing = [fname for _title, (fname, _fn) in DOCUMENTS.items()
               if not (SAMPLES / fname).exists()]
    assert not missing, f"sample_data/generated is missing {missing}"
    extra = sorted(p.name for p in SAMPLES.glob("*.pdf"))
    assert len(extra) == len(DOCUMENTS), f"unexpected files: {extra}"


def test_committed_copies_match_the_generator(tmp_path):
    fresh = {d.path.name: d for d in generate_all(tmp_path)}
    drift = []
    for name, doc in fresh.items():
        committed, generated = _fingerprint(SAMPLES / name), _fingerprint(doc.path)
        if committed != generated:
            which = [k for k in committed if committed[k] != generated[k]]
            drift.append(f"{name}: differs in {which}")
    assert not drift, "committed sample_data/generated PDFs are stale\n  " + "\n  ".join(drift)


def test_synthetic_watermark_is_on_every_page():
    """No demo file can be mistaken for a real manual: the watermark is per page."""
    for path in sorted(SAMPLES.glob("*.pdf")):
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            assert "SYNTHETIC DEMONSTRATION DOCUMENT" in text, f"{path.name} p{i}"
            assert "printed page" in text and "physical page" in text, f"{path.name} p{i}"
