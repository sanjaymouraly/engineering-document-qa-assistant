"""LLM provider adapters.

Protocols differ and are implemented natively — the Anthropic Messages API is
*not* treated as OpenAI-compatible.  Supported kinds:

* ``openai-compatible`` — OpenAI, OpenRouter, LM Studio, llama.cpp servers,
  vLLM, any gateway exposing ``/chat/completions``.
* ``anthropic``         — ``/v1/messages`` with ``x-api-key``.
* ``ollama``            — native ``/api/chat`` (no key needed).
* ``offline``           — no provider at all (extractive mode).

Security rules enforced here:
* secrets are never echoed in errors/status (masked);
* private/loopback endpoints are allowed (this is a local tool) but public
  endpoints with an *empty* base URL are refused when kind demands a key;
* nothing is called until the user saves + tests a provider explicitly.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from ..logging_setup import get_logger

log = get_logger("llm.providers")


# ----------------------------------------------------------------- models
@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    text: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None
    raw_finish: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ProviderStatus:
    kind: str
    label: str
    configured: bool
    online: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)
    latency_ms: int = 0
    ok: bool = False


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token + per-message
    overhead).  Marked as an estimate everywhere it is displayed."""
    return max(1, len(text) // 4 + 4)


# USD per 1M tokens — *estimates* surfaced in the UI, user-editable via the
# pricing table below; unknown models report cost as "not estimable" instead
# of guessing.  Keep tiny; deliberately not exhaustive.
_PRICING: list[tuple[re.Pattern, float, float, str]] = [
    (re.compile(r"gpt-4o-mini", re.I), 0.15, 0.60, "estimate"),
    (re.compile(r"gpt-4o$", re.I), 2.50, 10.0, "estimate"),
    (re.compile(r"gpt-4\.1-mini", re.I), 0.40, 1.60, "estimate"),
    (re.compile(r"o4-mini", re.I), 1.10, 4.40, "estimate"),
    (re.compile(r"claude-3-5-haiku|claude-haiku", re.I), 0.80, 4.00, "estimate"),
    (re.compile(r"claude-sonnet", re.I), 3.00, 15.00, "estimate"),
    (re.compile(r"llama3|mistral|qwen2|gemma|phi", re.I), 0.0, 0.0, "local-free"),
]


def price_for(model: str) -> tuple[float, float, str] | None:
    for pat, pin, pout, note in _PRICING:
        if pat.search(model or ""):
            return pin, pout, note
    return None


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    p = price_for(model)
    if not p:
        return None
    pin, pout, _ = p
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


# ------------------------------------------------------------------ errors
def friendly_http_error(status: int, body: str) -> str:
    b = (body or "")[:400]
    b = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***", b)
    if status in (401, 403):
        return ("Authentication failed (HTTP %d). Check the API key; note a "
                "ChatGPT/Claude.ai *subscription* is NOT an API key — API "
                "access is billed separately at platform.openai.com / "
                "console.anthropic.com." % status)
    if status == 404:
        return ("Endpoint or model not found (HTTP 404). Verify the base URL "
                "and model name; run 'List models' to see what this provider "
                "offers.")
    if status == 429:
        return "Rate limited (HTTP 429). The request was retried with backoff; try again shortly."
    if status >= 500:
        return f"Provider server error (HTTP {status})."
    return f"Provider error (HTTP {status}): {b}"


# --------------------------------------------------------------- providers
class BaseProvider:
    kind = "base"

    def __init__(self, *, base_url: str = "", api_key: str = "", model: str = "",
                 temperature: float = 0.1, max_tokens: int = 900, timeout_s: int = 90,
                 retries: int = 2, label: str = "", extra: dict | None = None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retries = max(0, min(4, int(retries)))
        self.label = label or self.kind
        self.extra = extra or {}

    # -------------------------------------------------------- url safety
    def _resolve_base(self, default_public: str | None) -> str:
        if self.base_url:
            u = urlparse(self.base_url)
            if u.scheme not in ("http", "https") or not u.netloc:
                raise ValueError("Base URL must start with http:// or https://")
            return self.base_url
        if default_public:
            return default_public
        raise ValueError("This provider kind needs a base URL "
                         "(e.g. http://localhost:11434 for Ollama).")

    @staticmethod
    def _is_local(url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(
            (".local", ".internal"))

    # ------------------------------------------------------------ plumbing
    def _client(self, timeout: float) -> httpx.Client:
        """Single construction point for every HTTP client in this module.

        ``extra["transport"]`` lets tests inject ``httpx.MockTransport`` so the
        suite can never reach the network — there are no other ``httpx.Client``
        constructions to miss.
        """
        return httpx.Client(timeout=timeout,
                            transport=self.extra.get("transport"))

    def _request_json(self, client: httpx.Client, method: str, url: str, *,
                      headers: dict, json_body: dict | None = None) -> tuple[int, dict, str]:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = client.request(method, url, headers=headers, json=json_body)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    ra = r.headers.get("retry-after")
                    try:
                        delay = float(ra) if ra else min(8.0, 0.8 * 2 ** attempt)
                    except ValueError:
                        delay = 1.0
                    time.sleep(delay)
                    continue
                try:
                    data = r.json()
                except (json.JSONDecodeError, ValueError):
                    data = {}
                return r.status_code, data, r.text
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(min(8.0, 0.8 * 2 ** attempt))
                    continue
                break
            except httpx.HTTPError as exc:
                last_exc = exc
                break
        raise ConnectionError(str(last_exc) if last_exc else "network failure")

    def _effective_model(self) -> str:
        if self.model:
            return self.model
        raise ValueError("No model configured. Click 'List models' to pick one.")

    # ----------------------------------------------------------- interface
    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        raise NotImplementedError

    def list_models(self) -> tuple[list[str], str | None]:
        raise NotImplementedError

    def status(self) -> ProviderStatus:
        t0 = time.perf_counter()
        try:
            models, err = self.list_models()
            latency = int((time.perf_counter() - t0) * 1000)
            if err:
                return ProviderStatus(self.kind, self.label, True, False, err,
                                      models, latency, ok=False)
            return ProviderStatus(self.kind, self.label, True, True,
                                  f"reachable; {len(models)} model(s) listed",
                                  models, latency, ok=True)
        except Exception as exc:
            return ProviderStatus(self.kind, self.label, True, False,
                                  re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***", str(exc))[:300],
                                  [], int((time.perf_counter() - t0) * 1000), ok=False)


class OpenAICompatibleProvider(BaseProvider):
    kind = "openai-compatible"

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        t0 = time.perf_counter()
        try:
            base = self._resolve_base(self.extra.get("default_base_url"))
            if self.extra.get("requires_key", True) and not self.api_key \
                    and not self._is_local(base):
                raise ValueError("API key is required for this endpoint "
                                 "(local endpoints may run without one).")
            body = {
                "model": self._effective_model(),
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            with self._client(self.timeout_s) as client:
                code, data, raw = self._request_json(
                    client, "POST", f"{base}/chat/completions",
                    headers=headers, json_body=body)
            if code != 200:
                return ChatResponse(error=friendly_http_error(code, raw),
                                    latency_ms=int((time.perf_counter() - t0) * 1000))
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            return ChatResponse(
                text=(choice.get("message") or {}).get("content") or "",
                model=data.get("model") or self.model,
                prompt_tokens=int(usage.get("prompt_tokens")
                                  or estimate_tokens("".join(m.content for m in messages))),
                completion_tokens=int(usage.get("completion_tokens")
                                      or estimate_tokens(json.dumps(choice))),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                raw_finish=choice.get("finish_reason") or "")
        except Exception as exc:
            return ChatResponse(error=_clean_error(exc),
                                latency_ms=int((time.perf_counter() - t0) * 1000))

    def list_models(self) -> tuple[list[str], str | None]:
        base = self._resolve_base(self.extra.get("default_base_url"))
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with self._client(min(20, self.timeout_s)) as client:
            code, data, raw = self._request_json(client, "GET", f"{base}/models",
                                                 headers=headers)
        if code != 200:
            return [], friendly_http_error(code, raw)
        ids = sorted({m.get("id", "") for m in data.get("data", []) if m.get("id")})
        return ids, None


class AnthropicProvider(BaseProvider):
    kind = "anthropic"

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        t0 = time.perf_counter()
        try:
            base = self._resolve_base("https://api.anthropic.com")
            if not self.api_key:
                raise ValueError("Anthropic requires an API key "
                                 "(console.anthropic.com — separate from a "
                                 "Claude subscription).")
            system = "\n\n".join(m.content for m in messages if m.role == "system")
            conv = [m for m in messages if m.role != "system"]
            body = {
                "model": self._effective_model(),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "assistant" if m.role == "assistant" else "user",
                              "content": m.content} for m in conv],
            }
            if system:
                body["system"] = system
            headers = {"x-api-key": self.api_key,
                       "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
            with self._client(self.timeout_s) as client:
                code, data, raw = self._request_json(
                    client, "POST", f"{base}/v1/messages", headers=headers,
                    json_body=body)
            if code != 200:
                return ChatResponse(error=friendly_http_error(code, raw),
                                    latency_ms=int((time.perf_counter() - t0) * 1000))
            text = "".join(bl.get("text", "") for bl in data.get("content", [])
                           if bl.get("type") == "text")
            usage = data.get("usage") or {}
            return ChatResponse(
                text=text, model=data.get("model") or self.model,
                prompt_tokens=int(usage.get("input_tokens") or 0),
                completion_tokens=int(usage.get("output_tokens") or 0),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                raw_finish=data.get("stop_reason") or "")
        except Exception as exc:
            return ChatResponse(error=_clean_error(exc),
                                latency_ms=int((time.perf_counter() - t0) * 1000))

    def list_models(self) -> tuple[list[str], str | None]:
        base = self._resolve_base("https://api.anthropic.com")
        if not self.api_key:
            return [], "API key required before listing models."
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        try:
            with self._client(min(20, self.timeout_s)) as client:
                code, data, raw = self._request_json(client, "GET", f"{base}/v1/models",
                                                      headers=headers)
        except Exception as exc:
            return [], _clean_error(exc)
        if code != 200:
            # Some deployments do not expose /v1/models — not fatal.
            return [], ("This endpoint did not provide a model list "
                        f"({friendly_http_error(code, raw)}); enter a model name manually.")
        ids = sorted({m.get("id", "") for m in data.get("data", []) if m.get("id")})
        return ids, None


class OllamaProvider(BaseProvider):
    kind = "ollama"

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        t0 = time.perf_counter()
        try:
            base = self._resolve_base("http://localhost:11434")
            body = {
                "model": self._effective_model(),
                "stream": False,
                "options": {"temperature": self.temperature,
                            "num_predict": self.max_tokens},
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            }
            with self._client(self.timeout_s) as client:
                code, data, raw = self._request_json(
                    client, "POST", f"{base}/api/chat",
                    headers={"Content-Type": "application/json"}, json_body=body)
            if code != 200:
                msg = (data.get("error") if isinstance(data, dict) else None) or raw
                return ChatResponse(
                    error=f"Ollama error (HTTP {code}): {str(msg)[:200]}. If the model "
                          "is missing, run: ollama pull <model>",
                    latency_ms=int((time.perf_counter() - t0) * 1000))
            return ChatResponse(
                text=(data.get("message") or {}).get("content") or "",
                model=data.get("model") or self.model,
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                raw_finish=data.get("done_reason") or "")
        except Exception as exc:
            return ChatResponse(error=_clean_error(exc) + " — is Ollama running "
                                "(`ollama serve`)?",
                                latency_ms=int((time.perf_counter() - t0) * 1000))

    def list_models(self) -> tuple[list[str], str | None]:
        base = self._resolve_base("http://localhost:11434")
        try:
            with self._client(min(15, self.timeout_s)) as client:
                code, data, raw = self._request_json(client, "GET", f"{base}/api/tags",
                                                      headers={})
        except Exception as exc:
            return [], _clean_error(exc) + " — start Ollama (`ollama serve`) first."
        if code != 200:
            return [], friendly_http_error(code, raw)
        ids = sorted({m.get("name", "") for m in data.get("models", []) if m.get("name")})
        return ids, None


class OfflineProvider(BaseProvider):
    kind = "offline"

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        return ChatResponse(error="Offline mode: no LLM configured. Answers are "
                                  "built extractively from the documents themselves.")

    def list_models(self) -> tuple[list[str], str | None]:
        return [], None

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.kind, "Offline (no API)", True, False,
                              "extractive answers; no network calls", ok=True)


# ------------------------------------------------------------ model choice
_SAFE_MODEL_HINTS = {
    "openai-compatible": [r"^gpt-4o-mini$", r"^gpt-4\.1-mini$"],
    "anthropic": [r"claude-[a-z0-9.\-]*haiku", r"claude-[a-z0-9.\-]*sonnet"],
}


def pick_default_model(kind: str, available: list[str]) -> str | None:
    """Choose a verified model ONLY from what the provider actually reports.

    We never invent model ids and never silently pick an expensive model:
    mini/haiku tiers are preferred when present, otherwise the caller must
    ask the user to choose.
    """
    hints = _SAFE_MODEL_HINTS.get(kind, [])
    for pat in hints:
        for m in available:
            if re.search(pat, m, re.I):
                return m
    return None


def _clean_error(exc: Exception) -> str:
    s = str(exc)
    s = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***", s)
    s = re.sub(r"(Bearer\s+)[A-Za-z0-9\-._~+/]{6,}", r"\1***", s, flags=re.I)
    return f"{type(exc).__name__}: {s[:280]}"


# --------------------------------------------------------------- registry
PROVIDER_KINDS = {
    "offline": {"label": "Offline (no API)", "needs_base": False, "needs_key": False},
    "ollama": {"label": "Ollama (local)", "needs_base": False, "needs_key": False},
    "openai": {"label": "OpenAI API", "needs_base": False, "needs_key": True,
               "base": "https://api.openai.com/v1"},
    "anthropic": {"label": "Anthropic Claude API", "needs_base": False,
                  "needs_key": True},
    "openrouter": {"label": "OpenRouter", "needs_base": False, "needs_key": True,
                   "base": "https://openrouter.ai/api/v1"},
    "custom": {"label": "Custom OpenAI-compatible endpoint", "needs_base": True,
               "needs_key": False},
}


def build_provider(config: dict) -> BaseProvider:
    kind = config.get("kind", "offline")
    kwargs = dict(
        base_url=config.get("base_url", "") or PROVIDER_KINDS.get(kind, {}).get("base", ""),
        api_key=config.get("api_key", "") or "",
        model=config.get("model", "") or "",
        temperature=float(config.get("temperature", 0.1) or 0.1),
        max_tokens=int(config.get("max_tokens", 900) or 900),
        timeout_s=int(config.get("timeout_s", 90) or 90),
        retries=int(config.get("retries", 2) if config.get("retries") is not None else 2),
        label=config.get("label", "") or PROVIDER_KINDS.get(kind, {}).get("label", kind),
        extra={"default_base_url": PROVIDER_KINDS.get(kind, {}).get("base", ""),
               "requires_key": PROVIDER_KINDS.get(kind, {}).get("needs_key", False)})
    if kind == "offline":
        return OfflineProvider(**kwargs)
    if kind == "ollama":
        return OllamaProvider(**kwargs)
    if kind == "anthropic":
        return AnthropicProvider(**kwargs)
    if kind in ("openai", "openrouter", "custom"):
        return OpenAICompatibleProvider(**kwargs)
    raise ValueError(f"Unknown provider kind: {kind!r}")
