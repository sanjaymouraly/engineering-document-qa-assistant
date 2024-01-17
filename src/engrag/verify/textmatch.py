"""Text matching helpers shared by citation verification and highlighting.

All comparisons run on *normalized* text so that whitespace and hyphenation
differences between the answer and the source do not break matching, while
the returned spans index into the ORIGINAL text for display/highlighting.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_DASH = re.compile(r"[\u2010-\u2015]|\u00ad")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_NUMCOMMA = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def normalize_for_match(text: str) -> str:
    t = _DASH.sub("-", text)
    t = t.replace("\u00a0", " ").replace("\u202f", " ")
    t = _NUMCOMMA.sub("", t)            # 1,500 -> 1500 (thousands separator)
    t = _WS.sub(" ", t)
    return t.strip()


@dataclass
class Span:
    start: int          # index into the original (non-normalized) text
    end: int
    matched: str        # exact original substring


class NormalizedIndex:
    """Maps normalized matches back to original-text spans."""

    def __init__(self, original: str):
        self.original = original
        chars: list[str] = []
        self._map: list[int] = []
        for i, ch in enumerate(original):
            ch2 = _DASH.sub("-", ch) if _DASH.match(ch) else ch
            if ch2 == "\u00a0":
                ch2 = " "
            if ch2.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    self._map.append(i)
                continue
            chars.append(ch2.lower())
            self._map.append(i)
        self.norm = "".join(chars)

    def find(self, needle: str) -> Span | None:
        n = normalize_for_match(needle).lower()
        if not n:
            return None
        # thousands separators: also strip from target for numeric equality
        n2 = _NUMCOMMA.sub("", n)
        for candidate in (n, n2):
            pos = self.norm.find(candidate)
            if pos >= 0:
                start = self._map[pos]
                end = self._map[pos + len(candidate) - 1] + 1
                return Span(start, end, self.original[start:end])
        return None


_INDEX_CACHE: "OrderedDict[str, NormalizedIndex]" = OrderedDict()
_INDEX_CACHE_MAX = 64


def index_for(text: str) -> NormalizedIndex:
    """Memoize normalized indexes.

    Citation verification probes a sentence against the same passage many
    times (once per candidate fragment); building the per-character index is
    by far the dominant cost, so the exact same string is indexed once.
    Keyed by the text itself (not ``hash``) so a collision can never return a
    wrong span — a verification tool must not trade correctness for speed.
    """
    idx = _INDEX_CACHE.get(text)
    if idx is None:
        idx = NormalizedIndex(text)
        _INDEX_CACHE[text] = idx
        while len(_INDEX_CACHE) > _INDEX_CACHE_MAX:
            _INDEX_CACHE.popitem(last=False)
    else:
        _INDEX_CACHE.move_to_end(text)
    return idx


def phrase_in_text(quote: str, text: str) -> Span | None:
    """Locate a (possibly multi-fragment, ``…``-joined) quote in a text."""
    idx = index_for(text)
    fragments = [f for f in _ELLIPSIS.split(quote) if f.strip()]
    if not fragments:
        return None
    first = idx.find(fragments[0])
    if first is None:
        return None
    if len(fragments) == 1:
        return first
    last_span = first
    cursor = first.end
    for frag in fragments[1:]:
        tail_idx = NormalizedIndex(text[cursor:])
        s = tail_idx.find(frag)
        if s is None:
            return None
        last_span = Span(first.start, cursor + s.end, text[first.start:cursor + s.end])
        cursor = cursor + s.end
    return last_span


def token_set(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-']{1,}|\d+(?:[.,]\d+)?", text)}
