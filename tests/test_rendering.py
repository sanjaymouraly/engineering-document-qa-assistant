"""Page rendering + citation highlighting.

These tests are deliberately pixel-level: a highlight that is *drawn* but at the
wrong coordinates looked fine in code review, so the check asserts where the
colour actually lands on the rendered bitmap.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from engrag.rendering import get_page_words, highlight_boxes, render_page_png

WORDS = [
    {"t": "Rated", "x0": 10.0, "x1": 40.0, "top": 100.0, "bottom": 112.0},
    {"t": "voltage", "x0": 45.0, "x1": 90.0, "top": 100.0, "bottom": 112.0},
    {"t": "132/33", "x0": 120.0, "x1": 160.0, "top": 100.0, "bottom": 112.0},
    {"t": "kV,", "x0": 162.0, "x1": 180.0, "top": 100.0, "bottom": 112.0},
    {"t": "Unrelated", "x0": 10.0, "x1": 70.0, "top": 130.0, "bottom": 142.0},
]


def test_multiword_terms_match_a_word_sequence():
    boxes = highlight_boxes(WORDS, ["Rated voltage", "132/33 kV"])
    assert len(boxes) == 2, boxes
    assert (round(boxes[0]["x0"]), round(boxes[0]["x1"])) == (9, 91)
    # the second term spans two words, so the box must join them
    assert (round(boxes[1]["x0"]), round(boxes[1]["x1"])) == (119, 181)


def test_absent_and_cross_line_terms_produce_no_box():
    assert highlight_boxes(WORDS, ["nope"]) == []
    # a phrase split over two lines must not become a band across the page
    assert highlight_boxes(WORDS, ["kV, Unrelated"]) == []


def test_punctuation_and_case_are_ignored():
    assert len(highlight_boxes(WORDS, ["132/33 kv"])) == 1


@pytest.fixture(scope="module")
def tiny_pdf(tmp_path_factory) -> Path:
    """A generated PDF (reportlab) with one known value on one page."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path_factory.mktemp("render") / "tiny.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(72, A4[1] - 100, "Rated power 40 MVA (three-phase)")
    c.drawString(72, A4[1] - 140, "Nothing relevant here")
    c.save()
    return path


def test_highlight_lands_on_the_right_pixels(tmp_path, tiny_pdf):
    """The rendered highlight must sit on the text, not off in a corner.

    pdfplumber reports points (origin top-left); the page image is rendered at
    ``scale`` pixels per point, so the draw step has to multiply by exactly
    that factor.  A wrong factor silently misplaces every citation highlight.
    """
    scale = 2.0
    words = get_page_words(tmp_path, tiny_pdf, 1, 1)
    target = next(w for w in words if w["t"] == "MVA")
    png = render_page_png(tiny_pdf, 1, 1, renders_dir=tmp_path, scale=scale,
                          highlight_terms=["40 MVA"], use_cache=False)
    assert png and len(png) > 1000
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")

    def yellow_fraction(w: dict) -> float:
        x0 = max(0, int(w["x0"] * scale) - 2)
        x1 = min(img.width - 1, int(w["x1"] * scale) + 2)
        y0 = max(0, int(w["top"] * scale) - 2)
        y1 = min(img.height - 1, int(w["bottom"] * scale) + 2)
        crop = img.crop((x0, y0, x1 + 1, y1 + 1))
        px = list(crop.getdata())
        if not px:
            return 0.0
        hit = sum(1 for (r, g, b) in px if r > 200 and g > 170 and b < 205)
        return hit / len(px)

    on = yellow_fraction(target)
    control = next(w for w in words if w["t"] == "relevant")
    off = yellow_fraction(control)
    assert on > 0.25, f"highlight covers only {on:.1%} of the cited word"
    assert off < 0.02, f"unrelated text is tinted ({off:.1%})"


def test_page_out_of_range_returns_none(tmp_path, tiny_pdf):
    assert render_page_png(tiny_pdf, 1, 9, renders_dir=tmp_path,
                           use_cache=False) is None
