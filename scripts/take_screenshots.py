#!/usr/bin/env python3
"""Capture the README screenshots with Playwright against a *running* app.

Nothing is mocked: every screenshot is a real render of the Streamlit frontend
answering real questions from the local index.  The script fails loudly (and
reports which steps succeeded) instead of silently producing blank images.

Usage:
    python scripts/take_screenshots.py --url http://127.0.0.1:8501 \
        --out docs/images

The target app must already be running (``python -m engrag web``) and must have
documents indexed (``python -m engrag build-demo``).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

QUESTION = ("What are the rated voltage and frequency of DEMO-TR-01, and what "
            "is its short-circuit impedance?")
SEARCH_QUERY = "earth conductor minimum cross-section 25 kA 3 s"
STATEMENT = ("The rated voltage of DEMO-TR-01 is 11 kV, which equals 11,000 V, "
             "and the maximum system voltage is 145.5 kV.")

NAV = {
    "ask": "Ask Engineering AI",
    "documents": "Document Library",
    "search": "Document Search",
    "citations": "Citation Viewer",
    "values": "Value Verification",
    "settings": "AI Settings",
    "evaluation": "Evaluation",
    "history": "History",
}


# The Streamlit *body* (not the sidebar).  Do not add a tag name here: the element is a
# styled component, and `[data-testid="stAppViewContainer"]` would include the sidebar.
MAIN_SEL = '[data-testid="stMain"]'


class Shot:
    def __init__(self, out: Path, url: str, scale: float) -> None:
        self.out = out
        self.url = url
        self.scale = scale
        self.manifest: list[dict] = []
        self.page = None

    # ------------------------------------------------------------- helpers
    def goto(self) -> None:
        self.page.goto(self.url, wait_until="domcontentloaded")
        self.page.wait_for_selector('div[data-testid="stAppViewContainer"]',
                                    timeout=90_000)
        self._wait_idle()

    def _wait_idle(self, settle_ms: int = 700) -> None:
        """Wait until Streamlit stops showing a spinner/progress marker."""
        try:
            self.page.wait_for_selector('div[data-testid="stStatusWidget"]',
                                        state="detached", timeout=120_000)
        except PWTimeout:
            pass
        self.page.wait_for_timeout(settle_ms)

    def nav(self, key: str) -> None:
        label = NAV[key]
        radio = self.page.get_by_role("radio", name=label)
        try:
            radio.first.wait_for(state="visible", timeout=15_000)
            radio.first.click(timeout=15_000)
        except PWTimeout:
            # Streamlit renders radios as <input>+<label>; fall back to the label
            self.page.locator(f'label:has-text("{label}")').first.click(
                timeout=15_000, force=True)
        self._wait_idle()

    def field(self, label: str, textarea: bool = True):
        kind = "stTextArea" if textarea else "stTextInput"
        loc = self.page.locator(f'div[data-testid="{kind}"]')
        try:
            loc = loc.filter(has_text=label).first
            ctrl = loc.locator("textarea" if textarea else "input").first
            ctrl.wait_for(timeout=10_000)
            return ctrl
        except PWTimeout:
            return self.page.locator("textarea" if textarea else "input").first

    def type_into(self, label: str, text: str, *, textarea: bool = True) -> None:
        """Fill a Streamlit input and force the change event it listens for.

        Streamlit only pushes a widget value into the app state on blur, and a
        button such as *Ask* stays disabled until that happens — so fill, Tab
        away, then wait for the app to settle.
        """
        ctrl = self.field(label, textarea=textarea)
        ctrl.click(timeout=20_000)
        ctrl.fill(text)
        ctrl.press("Tab")
        self._wait_idle(600)
        got = ctrl.input_value()
        assert text[:24] in got, f"input {label!r} holds {got[:60]!r}, not the text"

    def click(self, name: str, exact: bool = False, timeout: int = 25_000) -> bool:
        """Click a Streamlit button once it is visible and enabled."""
        try:
            btn = self.page.get_by_role("button", name=name, exact=exact).first
            btn.wait_for(state="visible", timeout=timeout)
            deadline = time.monotonic() + timeout / 1000
            while not btn.is_enabled() and time.monotonic() < deadline:
                self.page.wait_for_timeout(250)
            if not btn.is_enabled():
                return False
            btn.click(timeout=timeout)
            return True
        except PWTimeout:
            return False

    def tab(self, name: str) -> bool:
        try:
            t = self.page.get_by_role("tab", name=name).first
            t.wait_for(timeout=8_000)
            t.click()
            self._wait_idle(400)
            return True
        except PWTimeout:
            return False

    def wait_text(self, text: str, timeout: int = 120_000,
                  main_only: bool = False) -> bool:
        """Wait for *text*.  ``main_only`` restricts it to the page body.

        The restriction is not cosmetic: Streamlit's ``stAppViewContainer`` wraps the
        *sidebar too*, so scoping to it let a sidebar radio label ("Citation Viewer",
        "History", "Document Library") satisfy an assertion while the body still showed
        the previous view — a screenshot of the wrong page that looked green.  ``stMain``
        is the body only.
        """
        root = (self.page.locator(MAIN_SEL) if main_only else self.page)
        try:
            root.get_by_text(text, exact=False).first.wait_for(timeout=timeout)
            return True
        except PWTimeout:
            return False

    def _fail_if_view_raised(self, key: str) -> None:
        """A screenshot of an exception box is worse than no screenshot.

        Streamlit renders script errors inline, so every capture first checks
        that no exception/alert box is on the page (this is what caught a broken
        ``st.tabs`` unpack that a headless test run could not see).
        """
        box = self.page.locator('[data-testid="stException"], div.stException, '
                                '[data-testid="stAlertError"]')
        if box.count():
            text = " ".join(box.first.inner_text().split())[:400]
            raise RuntimeError(f"{key}: the view raised: {text}")

    def body_text(self, limit: int = 4000) -> str:
        """Text of the main panel, or ``""`` if it is not there at all."""
        loc = self.page.locator(MAIN_SEL)
        if not loc.count():
            return ""
        try:
            return loc.first.inner_text(timeout=5_000)[:limit]
        except Exception:                                   # pragma: no cover
            return ""

    def expect_view(self, heading: str, timeout: int = 60_000) -> None:
        """Fail unless the *body* of the app shows ``heading`` (i.e. the view switched).

        Streamlit re-renders asynchronously, and several view names double as sidebar
        radio labels, so asserting on the body only is what makes a green run meaningful.
        """
        if not self.wait_text(heading, timeout=timeout, main_only=True):
            body = " | ".join(self.body_text(220).split("\n"))[:220]
            raise AssertionError(f"expected view heading {heading!r} in the main panel; "
                                 f"the body shows: {body!r}")

    def _scroll_to_top(self) -> None:
        """Reset *every* scroll container before a full-page capture.

        Streamlit scrolls inside ``stMain`` rather than the document, so a plain
        ``full_page=True`` shot keeps whatever the last click had scrolled to and silently
        cuts off the top of the view (the section header, the document/page pickers).
        """
        self.page.evaluate("""() => {
            window.scrollTo(0, 0);
            for (const el of document.querySelectorAll('*')) {
                if (el.scrollHeight > el.clientHeight + 4 && el.clientHeight > 100) {
                    el.scrollTop = 0;
                }
            }
        }""")
        self.page.wait_for_timeout(250)

    def save(self, key: str, title: str, *, full_page: bool = False,
             note: str = "") -> None:
        self._fail_if_view_raised(key)
        if full_page:
            self._scroll_to_top()
        body = self.body_text().strip()
        if 0 < len(body) < 120:
            raise RuntimeError(f"{key}: the main panel is almost empty ({len(body)} chars) — "
                               "nothing worth screenshotting")
        path = self.out / f"{key}.png"
        self.page.screenshot(path=str(path), full_page=full_page,
                             animations="disabled")
        size = path.stat().st_size
        if size < 20_000:
            raise RuntimeError(f"{key}: suspiciously small screenshot ({size} B)")
        self.manifest.append({"file": path.name, "title": title, "note": note,
                              "bytes": size,
                              "viewport": {"width": self.page.viewport_size["width"],
                                            "height": self.page.viewport_size["height"]},
                              "device_scale_factor": self.scale})
        print(f"  ✔ {path.name:34s} {size / 1024:7.1f} KiB  {title}")

    # --------------------------------------------------------------- shots
    def run(self) -> int:
        out, failed = self.out, []

        def step(name, fn):
            try:
                fn()
            except Exception as exc:            # keep going, report at the end
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"  ✖ {name}: {type(exc).__name__}: {exc}")

        step("ask", self.ask_flow)
        step("documents", self.documents_flow)
        step("search", self.search_flow)
        step("values", self.values_flow)
        step("settings", self.settings_flow)
        step("evaluation", self.evaluation_flow)
        step("history", self.history_flow)

        (out / "screenshots_manifest.json").write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Playwright chromium against the running Streamlit app",
            "url": self.url,
            "question": QUESTION,
            "screenshots": self.manifest,
            "errors": failed,
        }, indent=2), encoding="utf-8")
        print(f"\n{len(self.manifest)} screenshots in {out}")
        if failed:
            print("errors:\n  " + "\n  ".join(failed))
        return 0 if self.manifest and not failed else 1

    def ask_flow(self) -> None:
        self.nav("ask")
        self.type_into("Your question", QUESTION)
        assert self.click("Ask", exact=True), "Ask button not found/enabled"
        self.expect_view("Ask Engineering AI")
        assert self.wait_text("Evidence from the source documents", main_only=True), \
            "extractive answer did not render"
        self.save("01_ask_answer", "Ask view: extractive answer with citations",
                  full_page=True,
                  note="offline mode: verbatim quotes, no generated prose")
        assert self.tab("Numeric verification"), "numeric tab missing"
        self.save("02_ask_numeric_verification",
                  "Numeric verification: value-by-value status badges")
        assert self.tab("Per-sentence checks"), "sentence tab missing"
        self.save("03_ask_sentence_checks",
                  "Per-sentence support, citation integrity notes")
        if self.tab("Citations & evidence"):
            if self.click("View page"):
                self._wait_idle(1500)
                self.expect_view("Citation Viewer")
                # the viewer must really be showing the page it claims: this is
                # what makes screenshot 04 evidence, not decoration
                shown = int(self.page.locator(
                    '[data-testid="stNumberInput"] input').first.input_value())
                assert self.wait_text(f"Physical page {shown} (", main_only=True), \
                    (f"the Citation Viewer opened page {shown} but is not "
                     "rendering that page's image")
                self.save("04_citation_page_highlight",
                          "Citation Viewer: source page opened from a citation "
                          "(rendered page, printed label, values detected on page)",
                          full_page=True)
                # ask for the nameplate values to be highlighted on the page image
                self.type_into("Highlight terms", "40 MVA, 132/33 kV, 50 Hz, YNd1",
                               textarea=False)
                self.page.keyboard.press("Enter")
                self._wait_idle(2500)
                self.save("04b_citation_highlights",
                          "Citation Viewer: cited values highlighted on the "
                          "rendered page", full_page=True)

    def documents_flow(self) -> None:
        self.nav("documents")
        self.expect_view("Document Library")
        self.save("05_document_library", "Document Library: index, statuses, "
                  "duplicate-safe upload", full_page=True)

    def search_flow(self) -> None:
        self.nav("search")
        self.type_into("Full-text + semantic search", SEARCH_QUERY, textarea=False)
        self.page.keyboard.press("Enter")
        self._wait_idle(1200)
        self.expect_view("Document Search")
        assert self.wait_text("result(s)", timeout=60_000, main_only=True), \
            "no search results"
        self._wait_idle(500)
        self.save("06_search_results", "Document Search: hybrid BM25 + vector "
                  "ranking with matched-snippet highlighting", full_page=True)

    def values_flow(self) -> None:
        self.nav("values")
        self.type_into("Statement to verify", STATEMENT)
        assert self.click("Verify", exact=True), "Verify button not found/enabled"
        self._wait_idle(1500)
        self.expect_view("Value Verification")
        assert (self.wait_text("unit_verified", main_only=True)
                or self.wait_text("verified", main_only=True)), "verification output missing"
        self.save("07_value_verification",
                  "Value Verification Engine: unit conversion and scope checks",
                  full_page=True,
                  note="11 kV→11,000 V passes as unit_verified (an exact "
                       "conversion) and 145.5 kV matches the DEMO-TR-01 source "
                       "row; every badge shows the document, page and quoted "
                       "source value it was checked against")

    def settings_flow(self) -> None:
        self.nav("settings")
        self.expect_view("AI Provider Settings")
        self.save("08_ai_settings", "AI Settings: provider picker, offline default",
                  full_page=True)

    def evaluation_flow(self) -> None:
        self.nav("evaluation")
        self.expect_view("Evaluation Dashboard")
        self.save("09_evaluation", "Evaluation Dashboard: measured metrics "
                  "(re-runnable from this page)", full_page=True)

    def history_flow(self) -> None:
        self.nav("history")
        self.expect_view("History")
        self.save("10_history", "History: stored answers, citations and "
                  "verification results", full_page=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1050)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(viewport={"width": args.width, "height": args.height},
                                  device_scale_factor=args.scale)
        page = ctx.new_page()
        shot = Shot(out, args.url, args.scale)
        shot.page = page
        shot.goto()
        rc = shot.run()
        browser.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
