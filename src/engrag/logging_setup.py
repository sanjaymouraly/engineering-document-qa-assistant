"""Structured logging with credential masking."""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timezone

_SENSITIVE = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.I)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]{6,}", re.I)
_KEYISH = re.compile(r"((?:sk|pk|key|token|secret|authorization)\s*[:=]\s*)(\S+)", re.I)
_KEYY = re.compile(r"\b((?:sk|pk|rk|ghp|xoxb|api)[-_][A-Za-z0-9\-_]{8,})\b")


class MaskingFormatter(logging.Formatter):
    """Never let API keys reach stdout, files, or tracebacks in logs."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        msg = _BEARER.sub(r"\1***", msg)
        msg = _KEYISH.sub(lambda m: m.group(1) + "***", msg)
        return msg

    @staticmethod
    def mask(value: str | None) -> str:
        """Mask a secret for safe display: keeps a 4-char fingerprint."""
        if not value:
            return ""
        v = str(value)
        if len(v) <= 8:
            return "****"
        return f"{v[:4]}…{v[-2:]} ({len(v)} chars, hidden)"


def mask_key_for_display(value: str | None) -> str:
    return MaskingFormatter.mask(value)


def scrub(text: str) -> str:
    """Redact anything key-like from arbitrary text (errors from providers)."""
    if not text:
        return text
    text = _BEARER.sub(r"\1***", text)
    text = _KEYISH.sub(lambda m: m.group(1) + "***", text)
    text = _KEYY.sub(lambda m: m.group(1)[:4] + "***", text)
    return text


def setup_logging(level: str | None = None, json_format: bool = False) -> None:
    lvl = (level or os.environ.get("ENGRAG_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger("engrag")
    if root.handlers:
        root.setLevel(lvl)
        return
    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",' \
              '"msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
    handler.setFormatter(MaskingFormatter(fmt))
    root.addHandler(handler)
    root.setLevel(lvl)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"engrag.{name}")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
