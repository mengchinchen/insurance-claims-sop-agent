"""Minimal LLM client (Anthropic Messages API or any OpenAI-compatible Chat Completions API) over httpx."""
from __future__ import annotations

import json
import time

import httpx

from .config import LLMSettings


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: LLMSettings):
        self.cfg = cfg
        self.calls: list[dict] = []   # lightweight telemetry for the UI

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    # ---------------------------------------------------------------- public API
    def extract(self, system: str, prompt: str, schema: dict) -> dict:
        """Structured extraction via forced tool/function call."""
        if self.cfg.provider == "anthropic":
            body = {
                "model": self.cfg.resolved_model(), "max_tokens": 800, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [{"name": "record_understanding", "description": "Record the structured understanding.",
                           "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": "record_understanding"},
            }
            data = self._post_anthropic(body, "extract")
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    return block.get("input") or {}
            raise LLMError("no tool_use block in response")
        body = {
            "model": self.cfg.resolved_model(), "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "tools": [{"type": "function", "function": {"name": "record_understanding",
                                                        "description": "Record the structured understanding.",
                                                        "parameters": schema}}],
            "tool_choice": {"type": "function", "function": {"name": "record_understanding"}},
        }
        data = self._post_openai(body, "extract")
        calls = data["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            raise LLMError("no tool call in response")
        return json.loads(calls[0]["function"]["arguments"] or "{}")

    def generate(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        if self.cfg.provider == "anthropic":
            body = {"model": self.cfg.resolved_model(), "max_tokens": max_tokens, "temperature": 0.3,
                    "system": system, "messages": [{"role": "user", "content": prompt}]}
            data = self._post_anthropic(body, "generate")
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        body = {"model": self.cfg.resolved_model(), "temperature": 0.3, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        data = self._post_openai(body, "generate")
        return (data["choices"][0]["message"].get("content") or "").strip()

    # ---------------------------------------------------------------- transport
    def _post_anthropic(self, body: dict, kind: str) -> dict:
        headers = {"x-api-key": self.cfg.api_key or "", "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        return self._post(f"{self.cfg.resolved_base_url()}/v1/messages", headers, body, kind)

    def _post_openai(self, body: dict, kind: str) -> dict:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "content-type": "application/json"}
        return self._post(f"{self.cfg.resolved_base_url()}/chat/completions", headers, body, kind)

    def _post(self, url: str, headers: dict, body: dict, kind: str) -> dict:
        t0 = time.time()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=self.cfg.timeout_s)
        except httpx.HTTPError as exc:
            self.calls.append({"kind": kind, "ok": False, "ms": int((time.time() - t0) * 1000), "error": str(exc)[:120]})
            raise LLMError(f"network error: {exc}") from exc
        ms = int((time.time() - t0) * 1000)
        if resp.status_code >= 400:
            self.calls.append({"kind": kind, "ok": False, "ms": ms, "error": f"HTTP {resp.status_code}"})
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        self.calls.append({"kind": kind, "ok": True, "ms": ms})
        return resp.json()


def probe(cfg: LLMSettings) -> tuple[bool, str]:
    """Cheap connectivity/auth check used by the UI's 'Test key' button."""
    try:
        out = LLMClient(cfg).generate("Reply with the single word: ok", "ping", max_tokens=5)
        return True, out or "ok"
    except Exception as exc:
        return False, str(exc)[:200]
