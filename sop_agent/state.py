"""Session state: SOP phase, cross-phase memory, counters and an audit trace."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    COMPLETE = "COMPLETE"      # terminal: conversation wrapped up
    HANDOFF = "HANDOFF"        # terminal: transferred to a human representative


WORKFLOW = [Phase.VERIFY_ID, Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS]

# The SOP graph. Anything not listed here is an illegal transition and raises.
ALLOWED_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.VERIFY_ID: {Phase.RESOLVE_INTENT, Phase.HANDOFF},
    Phase.RESOLVE_INTENT: {Phase.PROCESS_CASE, Phase.POST_PROCESS, Phase.HANDOFF},
    Phase.PROCESS_CASE: {Phase.RESOLVE_INTENT, Phase.POST_PROCESS, Phase.HANDOFF},
    Phase.POST_PROCESS: {Phase.PROCESS_CASE, Phase.RESOLVE_INTENT, Phase.COMPLETE, Phase.HANDOFF},
    Phase.COMPLETE: set(),
    Phase.HANDOFF: set(),
}

# How much freedom the LLM gets in each phase (used by the responder).
PHASE_FREEDOM = {
    Phase.VERIFY_ID: "strict",          # LLM may only extract + rephrase; never sees claim data
    Phase.RESOLVE_INTENT: "guided",     # LLM interprets messy language, picks among bounded options
    Phase.PROCESS_CASE: "grounded",     # LLM reasons/answers freely but only from tool facts
    Phase.POST_PROCESS: "strict",       # consent-driven email offer
    Phase.COMPLETE: "strict",
    Phase.HANDOFF: "strict",
}


class Slot(BaseModel):
    """A remembered fact with provenance: when (turn) and in which phase it was captured."""
    value: Any
    turn: int
    phase: Phase
    source: str = "rules"   # rules | llm


class Memory(BaseModel):
    identity: dict[str, Slot] = Field(default_factory=dict)       # full_name, dob, phone, email, id_last4, policy_number
    caller: dict[str, Slot] = Field(default_factory=dict)         # is_representative, caller_name, relationship
    case_hints: dict[str, Slot] = Field(default_factory=dict)     # case_id, case_type, status, month, year
    intent: Optional[Slot] = None
    open_questions: list[Slot] = Field(default_factory=list)      # questions asked before we could answer
    declined_fields: list[str] = Field(default_factory=list)
    unavailable_fields: list[str] = Field(default_factory=list)

    def remember(self, bucket: str, key: str, value: Any, turn: int, phase: Phase, source: str = "rules") -> bool:
        store: dict[str, Slot] = getattr(self, bucket)
        old = store.get(key)
        if old is not None and old.value == value:
            return False
        store[key] = Slot(value=value, turn=turn, phase=phase, source=source)
        return True

    def get(self, bucket: str, key: str, default: Any = None) -> Any:
        slot = getattr(self, bucket).get(key)
        return slot.value if slot else default


class Counters(BaseModel):
    out_of_scope_consecutive: int = 0
    out_of_scope_total: int = 0
    persuasion_attempts: int = 0
    verification_failures: int = 0
    emotional_turns: int = 0


class Verification(BaseModel):
    verified: bool = False
    party_id: Optional[str] = None
    matched_fields: list[str] = Field(default_factory=list)
    verified_turn: Optional[int] = None


class Consent(BaseModel):
    required: bool = False
    scenario: str = "default"
    status: str = "not_requested"   # not_requested | pending | approved | timeout
    polls: int = 0
    request_id: Optional[str] = None
    rep_name: Optional[str] = None
    relationship: Optional[str] = None


class TraceEvent(BaseModel):
    turn: int
    phase: str
    kind: str          # transition | tool | gate | guard | nlu | llm | memory
    detail: dict[str, Any]
    ts: float = Field(default_factory=time.time)


class Message(BaseModel):
    role: str          # user | agent
    text: str
    phase: str
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.VERIFY_ID
    turn: int = 0
    memory: Memory = Field(default_factory=Memory)
    verification: Verification = Field(default_factory=Verification)
    consent: Consent = Field(default_factory=Consent)
    counters: Counters = Field(default_factory=Counters)
    active_case_id: Optional[str] = None
    case_candidates: list[str] = Field(default_factory=list)
    discussed_cases: list[str] = Field(default_factory=list)
    topics: list[dict[str, str]] = Field(default_factory=list)      # [{case_id, aspect}]
    pending_prompt: Optional[str] = None   # what yes/no answers refer to
    email: dict[str, Any] = Field(default_factory=dict)
    handoff: dict[str, Any] = Field(default_factory=dict)
    transcript: list[Message] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)

    def log(self, kind: str, **detail: Any) -> None:
        self.trace.append(TraceEvent(turn=self.turn, phase=self.phase.value, kind=kind, detail=detail))

    def add_topic(self, case_id: str, aspect: str) -> None:
        item = {"case_id": case_id, "aspect": aspect}
        if item not in self.topics:
            self.topics.append(item)
