"""Prompt-injection defenses for untrusted document text.

Uploaded manuals are attacker-controlled input.  Before any passage is sent
to an LLM we:
  * neutralise bracket tokens like ``[D1 p3]`` so a document cannot forge the
    citation syntax our verifier relies on (the app adds citations itself);
  * mask classic instruction-override phrases and tool/system markup so they
    are visible to the user but inert for the model;
  * record every hit so the UI can warn "document text contained
    injection-like content (treated as data only)".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Deliberately broad: the cost of over-masking a manual line is a weird token,
# the cost of missing an injection is a fabricated answer.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?"
        r"(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|"
        r"messages?|rules?|directives?)", re.I)),
    ("system_prompt_try", re.compile(r"\bsystem\s*(?:prompt|message)\b", re.I)),
    ("role_marker", re.compile(r"^\s*(?:system|assistant|developer|tool)\s*:",
                               re.M | re.I)),
    ("chatml_markup", re.compile(r"</?\s*\|?im_start\|?|</?\s*\|?im_end\|?"
                                 r"|<\s*/?\s*(?:system|instruction|prompt)\s*>", re.I)),
    ("secret_ask", re.compile(r"\b(?:reveal|print|repeat|show|output|leak)\b[^.\n]{0,40}"
                              r"\b(?:api[_ ]?key|secret|token|password|credential)s?\b",
                              re.I)),
    ("tool_trigger", re.compile(r"\b(?:execute|run|call|invoke)\s+(?:this\s+)?(?:tool|"
                                r"function|command|shell|bash)\b", re.I)),
    ("persona_hijack", re.compile(r"\byou\s+are\s+now\b|\bnew\s+instructions?\s*:",
                                  re.I)),
]
_CITE_TOKEN = re.compile(r"\[[^\]\n]{0,48}\]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class SanitizedText:
    text: str
    flags: list[str] = field(default_factory=list)
    replaced: int = 0

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def sanitize_document_text(text: str) -> SanitizedText:
    """Return model-safe text + flags describing what was neutralised."""
    flags: list[str] = []
    text = _CONTROL.sub(" ", text)
    for name, pat in _PATTERNS:
        text, n = pat.subn("[redacted-instruction-like-text]", text)
        if n:
            flags.append(name)
    # citation-syntax neutralisation (after pattern scan so [..] stays intact)
    text, n = _CITE_TOKEN.subn("[bracket-masked]", text)
    replaced = n
    return SanitizedText(text=text, flags=flags, replaced=replaced)


def combined_flags(results: list[SanitizedText]) -> list[str]:
    out: list[str] = []
    for r in results:
        for f in r.flags:
            if f not in out:
                out.append(f)
    return out
