"""FastAPI app: JSON API + the single-page test UI."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .agent import SOPAgent
from .config import LLMSettings, settings
from .data import Repository
from .llm import probe

app = FastAPI(title="Insurance Claims SOP Agent", version="1.0.0")
repo = Repository(settings.fixtures_dir)
STATIC = Path(__file__).parent / "static"

MAX_SESSIONS = 200
_sessions: "OrderedDict[str, tuple[SOPAgent, threading.Lock, float]]" = OrderedDict()
_lock = threading.Lock()


class NewSession(BaseModel):
    provider: str | None = None        # anthropic | openai | none
    api_key: str | None = None         # optional per-session token (falls back to env)
    model: str | None = None
    base_url: str | None = None
    consent_scenario: str | None = None


class UserMessage(BaseModel):
    text: str


def _llm_settings(req: NewSession) -> LLMSettings:
    base = settings.default_llm()
    if req.provider == "none":
        return LLMSettings(provider="none")
    provider = (req.provider or base.provider).lower()
    key = req.api_key or (base.api_key if provider == base.provider else None)
    return LLMSettings(provider=provider, api_key=key, model=req.model or (base.model if provider == base.provider
                                                                            else None),
                       base_url=req.base_url or (base.base_url if provider == base.provider else None),
                       timeout_s=base.timeout_s)


def _get(session_id: str):
    with _lock:
        item = _sessions.get(session_id)
        if not item:
            raise HTTPException(404, "session not found — start a new conversation")
        _sessions.move_to_end(session_id)
        return item


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    llm = settings.default_llm()
    return {"llm": {"provider": llm.provider, "model": llm.resolved_model(), "has_server_key": bool(llm.api_key)},
            "consent_scenarios": sorted(repo.consent_scenarios), "default_consent_scenario": settings.consent_scenario,
            "demo_today": settings.today().isoformat(), "min_identity_fields": settings.min_identity_fields}


@app.post("/api/llm/test")
def llm_test(req: NewSession):
    cfg = _llm_settings(req)
    if not cfg.enabled:
        return {"ok": False, "detail": "No API key provided (template mode will be used)."}
    ok, detail = probe(cfg)
    return {"ok": ok, "detail": detail, "model": cfg.resolved_model()}


@app.post("/api/sessions")
def new_session(req: NewSession):
    agent = SOPAgent(repo, settings, _llm_settings(req), consent_scenario=req.consent_scenario)
    first = agent.start()
    with _lock:
        _sessions[agent.state.session_id] = (agent, threading.Lock(), time.time())
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    return {"session_id": agent.state.session_id, **first}


@app.post("/api/sessions/{session_id}/messages")
def send(session_id: str, msg: UserMessage):
    agent, lock, _ = _get(session_id)
    text = msg.text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    if len(text) > 2000:
        raise HTTPException(400, "message too long")
    with lock:
        return agent.handle(text)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    agent, _, _ = _get(session_id)
    return {"state": agent.snapshot(),
            "transcript": [m.model_dump() for m in agent.state.transcript]}


@app.exception_handler(Exception)
def on_error(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"{exc.__class__.__name__}: {exc}"})


def main():
    import os

    import uvicorn
    uvicorn.run("sop_agent.server:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
