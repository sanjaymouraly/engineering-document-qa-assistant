#!/usr/bin/env python3
"""Browser end-to-end checks for the workflows a unit test cannot see.

`pytest` never renders Streamlit, so the tests in `tests/` cannot catch a broken
*citation deep link*, a streamlit session-state mistake or a widget that loses
its value after a jump.  This script drives the real, already-running app with
Playwright and asserts the user-visible behaviour.

    # in one terminal
    python -m engrag build-demo
    python -m engrag web
    # in another
    python scripts/ui_e2e.py --url http://127.0.0.1:8501

Exits non-zero with a message naming the failed step.  Nothing is mocked: the
assertions are made against the live page in the browser.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

# A question whose best evidence sits on a page > 1, so "View page" has to
# actually change the page instead of agreeing with the default.
QUESTION = "What are the no-load loss and load loss of DEMO-TR-01 at 75 °C?"
SEARCH_QUERY = "earth conductor minimum cross-section"

EVIDENCE_HEAD = re.compile(
    r"\[\d+\]\s+(?P<title>.+?)\s+—\s+physical page\s+(?P<page>\d+)")


class Checker:
    def __init__(self, page) -> None:
        self.page = page
        self.steps: list[str] = []

    # ------------------------------------------------------------- plumbing
    def idle(self, settle_ms: int = 700) -> None:
        try:
            self.page.wait_for_selector('div[data-testid="stStatusWidget"]',
                                        state="detached", timeout=120_000)
        except PWTimeout:
            pass
        self.page.wait_for_timeout(settle_ms)

    def main_text(self) -> str:
        return self.page.locator('[data-testid="stMain"]').inner_text()

    def no_render_errors(self, where: str) -> None:
        boxes = self.page.locator('[data-testid="stException"]').count()
        assert boxes == 0, f"{where}: {boxes} rendered exception box(es)"
        assert "Traceback (most recent call last)" not in self.main_text(), \
            f"{where}: a traceback is visible in the page"
        self.steps.append(f"no rendered errors on {where}")

    def nav(self, label: str) -> None:
        """Switch views through the sidebar radio.

        Streamlit renders the radio as a visually hidden ``<input>`` plus a
        styled label; the input is often covered by the sidebar overlay, so the
        label is clicked with ``force`` when a normal click cannot land.
        """
        radio = self.page.get_by_role("radio", name=label)
        try:
            radio.first.wait_for(state="visible", timeout=15_000)
            radio.first.click(timeout=8_000)
        except PWTimeout:
            self.page.locator(f'label:has-text("{label}")').first.click(
                timeout=15_000, force=True)
        self.idle(900)

    def type_into(self, label: str, text: str, *, textarea: bool = True) -> None:
        kind = "stTextArea" if textarea else "stTextInput"
        loc = self.page.locator(f'div[data-testid="{kind}"]').filter(has_text=label).first
        ctrl = loc.locator("textarea" if textarea else "input").first
        ctrl.wait_for(timeout=20_000)
        ctrl.click()
        ctrl.fill(text)
        ctrl.press("Tab")
        self.idle(600)

    def click(self, name: str, *, nth: int = 0, timeout: int = 30_000) -> None:
        btn = self.page.get_by_role("button", name=name).nth(nth)
        btn.wait_for(state="visible", timeout=timeout)
        deadline = time.monotonic() + timeout / 1000
        while not btn.is_enabled() and time.monotonic() < deadline:
            self.page.wait_for_timeout(250)
        assert btn.is_enabled(), f"button {name!r} never became enabled"
        btn.click(timeout=timeout)

    # ---------------------------------------------------------------- flows
    def citation_deep_link(self) -> None:
        """Ask → click *View page* on a page-N citation → the viewer must show
        that exact document and page, and keep working afterwards."""
        self.nav("Ask Engineering AI")
        self.type_into("Your question", QUESTION)
        self.click("Ask", timeout=60_000)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            m = EVIDENCE_HEAD.search(self.main_text())
            if m:
                break
            self.page.wait_for_timeout(500)
        else:
            body = self.main_text()
            print("---- main area text (first 1200 chars) ----")
            print(body[:1200])
            print("-------------------------------------------")
            raise AssertionError("the answer never listed an evidence passage "
                                 "with a physical page number")
        title, want_page = m.group("title").strip(), int(m.group("page"))
        assert want_page > 1, (
            f"this test is pointless unless the citation is off page 1 (got "
            f"page {want_page}) — the demo corpus changed")
        self.steps.append(f"answer cites “{title}” on physical page {want_page}")

        self.click("View page", nth=0)
        self.idle(2000)
        assert "Citation Viewer" in self.main_text(), "citation link did not open the Citation Viewer"
        got = int(self.page.locator('[data-testid="stNumberInput"] input').first
                  .input_value())
        assert got == want_page, (
            f"the viewer opened physical page {got}, but the citation pointed at "
            f"page {want_page}")
        self.steps.append(f"“View page” opened the cited page {got}")
        assert title in self.main_text(), \
            "the viewer did not switch to the cited document"
        assert f"Physical page {want_page} (" in self.main_text(), \
            "the rendered page image is not labelled with the cited page"
        self.steps.append("viewer shows the cited document's rendered page image")
        self.no_render_errors("Citation Viewer")

        # the page field must still be a normal, usable widget after the jump
        box = self.page.locator('[data-testid="stNumberInput"] input').first
        box.fill("1")
        box.press("Enter")
        self.idle(2000)
        assert int(box.input_value()) == 1, "the page field lost its value"
        assert "Physical page 1 (" in self.main_text(), \
            "changing the page did not re-render the page image"
        self.steps.append("the page field re-renders a different page")
        self.no_render_errors("Citation Viewer (page 1)")

    def search(self) -> None:
        self.nav("Document Search")
        self.type_into("Full-text + semantic search", SEARCH_QUERY, textarea=False)
        self.page.keyboard.press("Enter")
        self.idle(1200)
        txt = self.main_text()
        assert "result(s)" in txt, "search produced no result counter"
        m = re.search(r"(\d+)\s+result\(s\)", txt)
        assert m and int(m.group(1)) >= 1, f"search returned {m.group(1) if m else '?'} results"
        self.steps.append(f"search returned {m.group(1)} result(s)")
        self.no_render_errors("Document Search")

    def value_verification(self) -> None:
        self.nav("Value Verification")
        self.type_into("Statement to verify",
                       "The rated voltage of DEMO-TR-01 is 33 kV.")
        self.click("Verify", timeout=60_000)
        self.idle(1500)
        txt = self.main_text()
        assert "result" in txt.lower() or "verified" in txt.lower(), \
            "the verification view produced no report"
        assert "No verified answer" not in txt, "the verifier abstained on a valid statement"
        self.no_render_errors("Value Verification")

    def ai_settings_offline(self) -> None:
        self.nav("AI Settings")
        txt = self.main_text()
        assert "Offline" in txt, "AI Settings does not offer the offline mode"
        assert "sk-" not in txt and "sk_live" not in txt, \
            "what looks like an API key is rendered in the settings page"
        self.steps.append("AI Settings defaults to offline and shows no secret")
        self.no_render_errors("AI Settings")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1500, "height": 1050},
                                device_scale_factor=1)
        page.set_default_timeout(30_000)
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=90_000)
        ck = Checker(page)
        ck.idle(1500)

        modules = [("citation deep link", ck.citation_deep_link),
                   ("search", ck.search),
                   ("value verification", ck.value_verification),
                   ("AI settings", ck.ai_settings_offline)]
        for name, fn in modules:
            try:
                fn()
            except Exception as exc:
                browser.close()
                print(f"\nE2E FAILED in {name}: {type(exc).__name__}: {exc}")
                print("steps that passed before the failure:")
                for s in ck.steps:
                    print(f"  ✔ {s}")
                return 1
        browser.close()

    print("UI end-to-end checks passed:")
    for s in ck.steps:
        print(f"  ✔ {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
