"""SOP orchestrator.

Division of labour
------------------
* Code (this file) owns the workflow: phase order, gates, which tools may run, what must be said.
* The LLM (optional) owns language: understanding messy input (nlu.py) and phrasing replies (responder.py),
  with freedom that varies per phase and with validators that fall back to safe templates.
"""
from __future__ import annotations

import re
from typing import Any

from .config import LLMSettings, Settings
from .data import Repository
from .llm import LLMClient
from .nlu import UNSUPPORTED_INTENTS, NLUResult, _aspects, understand
from .normalize import (IDENTITY_FIELDS, MONTH_NAMES, human_date, join_words, mask_value, money)
from .responder import TurnPlan, render
from .state import ALLOWED_TRANSITIONS, WORKFLOW, Message, Phase, SessionState, Slot
from .tools import ASPECT_LABELS, Toolbox, claim_brief

INTENT_DEFAULT_ASPECTS = {
    "denial_question": ["status", "denial_reason", "documents_needed", "appeal_deadline"],
    "status_inquiry": ["status"],
    "document_submission": ["documents_needed", "submission_method", "document_requirements"],
    "next_steps": ["documents_needed", "submission_method", "submission_timing", "appeal_deadline"],
    "payment_question": ["status", "amounts"],
    "status_inquiry_closed": ["status", "amounts"],
    "general_claim_question": ["status"],
    "list_claims": ["status"],
}
ASPECT_TO_TOPIC = {
    "submission_method": "submission_method",
    "submission_timing": "submission_timing",
    "processing_time": "processing_time_after_submission",
    "receipt_confirmation": "receipt_confirmation",
    "document_alternatives": "missing_required_material_alternatives",
}
DOC_ASPECTS = {"documents_needed", "document_requirements", "document_alternatives", "submission_method",
               "submission_timing", "processing_time", "receipt_confirmation"}
PRONOUN = {"mother": "her", "father": "his", "son": "his", "daughter": "her"}


class SOPViolation(RuntimeError):
    pass


class SOPAgent:
    def __init__(self, repo: Repository, settings: Settings, llm_settings: LLMSettings | None = None,
                 consent_scenario: str | None = None):
        self.repo, self.settings = repo, settings
        self.state = SessionState()
        self.state.consent.scenario = consent_scenario or settings.consent_scenario
        self.llm = LLMClient(llm_settings) if llm_settings and llm_settings.enabled else None
        self.tools = Toolbox(repo, settings, self.state)
        self.expecting: list[str] = list(IDENTITY_FIELDS)
        self.noted: set[str] = set()

    # ================================================================== public API
    def start(self) -> dict:
        plan = TurnPlan(phase=self.state.phase)
        plan.add("greeting")
        return self._finish(plan, user_text="")

    def handle(self, text: str) -> dict:
        st = self.state
        text = (text or "").strip()
        st.turn += 1
        st.transcript.append(Message(role="user", text=text, phase=st.phase.value))
        plan = TurnPlan(phase=st.phase)

        if st.phase in (Phase.COMPLETE, Phase.HANDOFF):
            plan.add("complete_already") if st.phase == Phase.COMPLETE else plan.add(
                "text", text="You've been transferred to a human representative, who will continue from here.")
            return self._finish(plan, text)

        nlu = understand(text, self._nlu_ctx(), self.llm)
        st.log("nlu", source=nlu.source, act=nlu.act, scope=nlu.scope, intent=nlu.intent, aspects=nlu.aspects,
               emotions=nlu.emotions, identity_fields=sorted(nlu.identity), hints=nlu.case_hints,
               caller={k: v for k, v in nlu.caller.items()}, llm_error=nlu.llm_error)
        new_hints = self._remember(nlu)
        plan.emotions = list(nlu.emotions)
        if nlu.emotions:
            st.counters.emotional_turns += 1

        # ---- global rule 1: the caller can always ask for a human
        if nlu.act == "request_human" or (st.pending_prompt == "transfer_offer" and nlu.act == "affirm"):
            self._handoff(plan, "caller requested a human representative")
            return self._finish(plan, text)
        if st.pending_prompt == "transfer_offer" and nlu.act == "deny":
            st.pending_prompt = None
            plan.add("text", text="No problem, let's keep going.")

        # ---- global rule 2: scope guard
        if nlu.scope == "out_of_scope":
            st.counters.out_of_scope_consecutive += 1
            st.counters.out_of_scope_total += 1
            n, mx = st.counters.out_of_scope_consecutive, self.settings.max_out_of_scope
            st.log("guard", check="scope", ok=False, consecutive=n)
            if n > mx:
                self._handoff(plan, "caller repeatedly asked out-of-scope questions")
            else:
                plan.add("oos", count=n, max=mx, redirect=self._redirect_hint())
                if n >= mx:
                    st.pending_prompt = "transfer_offer"
            return self._finish(plan, text)
        st.counters.out_of_scope_consecutive = 0

        if nlu.scope == "chitchat" and not nlu.has_slots() and nlu.act not in ("affirm", "deny", "done",
                                                                               "choose_option"):
            plan.add("chitchat", text=self._chitchat(text))

        # ---- phase handler (code decides; may auto-advance through several phases in one turn)
        handler = {Phase.VERIFY_ID: self._verify, Phase.RESOLVE_INTENT: self._resolve,
                   Phase.PROCESS_CASE: self._process, Phase.POST_PROCESS: self._post}[st.phase]
        handler(nlu, plan, text, new_hints)
        return self._finish(plan, text)

    # ================================================================== memory
    def _remember(self, nlu: NLUResult) -> dict:
        """Store everything useful, whatever phase it belongs to. Returns hints that are new this turn."""
        st, mem = self.state, self.state.memory
        new: dict[str, Any] = {}
        src = "llm" if "llm" in nlu.source else "rules"
        # A representative revealed later: a name stored earlier as the policyholder's was actually the caller's.
        if nlu.caller.get("is_representative") and not mem.get("caller", "is_representative"):
            prev = mem.get("identity", "full_name")
            if prev and prev != nlu.identity.get("full_name") and "caller_name" not in nlu.caller:
                mem.remember("caller", "caller_name", prev, st.turn, st.phase, src)
                if "full_name" not in nlu.identity:
                    mem.identity.pop("full_name", None)
            new["is_representative"] = True
        for k, v in nlu.caller.items():
            mem.remember("caller", k, v, st.turn, st.phase, src)
        if not st.verification.verified:
            for k, v in nlu.identity.items():
                if mem.remember("identity", k, v, st.turn, st.phase, src):
                    new.setdefault("identity", []).append(k)
                if k in mem.declined_fields:
                    mem.declined_fields.remove(k)
                if k in mem.unavailable_fields:
                    mem.unavailable_fields.remove(k)
        for k, v in nlu.case_hints.items():
            if mem.remember("case_hints", k, v, st.turn, st.phase, src):
                new.setdefault("hints", []).append(k)
        if nlu.intent:
            if not mem.intent or mem.intent.value != nlu.intent:
                new["intent"] = nlu.intent
            mem.intent = Slot(value=nlu.intent, turn=st.turn, phase=st.phase, source=src)
        for f in nlu.declined_fields:
            if f not in mem.declined_fields and f not in nlu.identity:
                mem.declined_fields.append(f)
        for f in nlu.unavailable_fields:
            if f not in mem.unavailable_fields and f not in nlu.identity:
                mem.unavailable_fields.append(f)
        if nlu.is_question and not st.verification.verified and (nlu.intent or nlu.aspects):
            mem.open_questions.append(Slot(value=st.transcript[-1].text, turn=st.turn, phase=st.phase))
        if new:
            st.log("memory", stored=new)
        return new

    def _hint_summary(self) -> str:
        h = {k: v.value for k, v in self.state.memory.case_hints.items()}
        if h.get("case_id"):
            return f"claim {h['case_id']}"
        words = ["your"]
        if h.get("status"):
            words.append(h["status"])
        if h.get("case_type"):
            words.append(h["case_type"])
        words.append("claim")
        when = []
        if h.get("month"):
            when.append(MONTH_NAMES[int(h["month"])])
        if h.get("year"):
            when.append(str(h["year"]))
        if when:
            words.append("from " + " ".join(when))
        return " ".join(words)

    # ================================================================== transitions
    def _goto(self, target: Phase, plan: TurnPlan, nlu: NLUResult | None, text: str = "") -> None:
        st = self.state
        if target not in ALLOWED_TRANSITIONS[st.phase]:
            raise SOPViolation(f"illegal transition {st.phase.value} -> {target.value}")
        gate_ok, why = self._gate(target)
        st.log("gate", target=target.value, allowed=gate_ok, reason=why)
        if not gate_ok:
            raise SOPViolation(f"gate refused {st.phase.value} -> {target.value}: {why}")
        st.log("transition", frm=st.phase.value, to=target.value)
        st.phase = target
        plan.phase = target
        st.pending_prompt = None
        enter = {Phase.RESOLVE_INTENT: self._enter_resolve, Phase.PROCESS_CASE: self._enter_process,
                 Phase.POST_PROCESS: self._enter_post}.get(target)
        if enter:
            enter(plan, nlu, text)

    def _gate(self, target: Phase) -> tuple[bool, str]:
        st = self.state
        verified_ok = st.verification.verified and (not st.consent.required or st.consent.status == "approved")
        if target in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS) and not verified_ok:
            return False, "identity not verified (or consent missing)"
        if target == Phase.PROCESS_CASE:
            owned = {c["case_id"] for c in self.repo.claims_for(st.verification.party_id or "")}
            if st.active_case_id not in owned:
                return False, "no resolved case owned by the verified party"
        if target == Phase.COMPLETE and "decision" not in st.email:
            return False, "email summary decision not recorded"
        return True, "ok"

    # ================================================================== VERIFY_ID (strict)
    def _verify(self, nlu: NLUResult, plan: TurnPlan, text: str, new: dict) -> None:
        st, mem = self.state, self.state.memory
        is_rep = bool(mem.get("caller", "is_representative"))
        plan.facts = {"verification_rule": f"at least {self.settings.min_identity_fields} of: full name, DOB, phone, "
                                           f"email, SSN last 4",
                      "caller_is_representative": is_rep}

        # Waiting on the policyholder's consent (representative calls).
        if st.consent.status == "pending":
            res = self.tools.call("check_consent")
            holder = self._holder_first()
            if res["status"] == "approved":
                plan.add("consent_approved", holder_first=holder)
                self._goto(Phase.RESOLVE_INTENT, plan, nlu, text)
            elif res["status"] == "timeout":
                plan.add("consent_timeout")
                self._handoff(plan, "policyholder consent not received", announce=False)
            else:
                if set(nlu.emotions) & {"frustration", "anger", "confusion"}:
                    plan.add("explain_consent", holder_first=holder, pronoun=self._pronoun())
                seq = self.repo.consent_scenarios[st.consent.scenario]["status_sequence"]
                plan.add("consent_pending", holder_first=holder, polls=res["polls"], max_polls=len(seq))
            return

        if new.get("is_representative"):
            holder = mem.get("identity", "full_name") or "the policyholder"
            plan.add("rep_detected", relationship=mem.get("caller", "relationship") or "family member",
                     holder=holder.split(" ")[0] if holder != "the policyholder" else holder)
        if new.get("hints") or (new.get("intent") and new["intent"] not in UNSUPPORTED_INTENTS):
            if mem.case_hints:
                plan.add("noted_hint", summary=self._hint_summary())
                plan.facts["caller_reason_in_own_words"] = self._hint_summary()
            elif "question" not in self.noted:
                plan.add("noted_question")
            self.noted.add("question")
        elif nlu.is_question and (nlu.intent or nlu.aspects) and "question" not in self.noted:
            plan.add("noted_question")
            self.noted.add("question")
        declined_now = [f for f in nlu.declined_fields + nlu.unavailable_fields if f in IDENTITY_FIELDS]
        if declined_now:
            plan.add("field_declined_ack", fields=declined_now)

        have = [f for f in IDENTITY_FIELDS if f in mem.identity]
        need = self.settings.min_identity_fields
        if is_rep and len(have) >= need and not mem.get("caller", "caller_name"):
            plan.add("rep_need_name")
            self.expecting = ["caller_name"]
            return

        if len(have) >= need:
            fields = {f: mem.get("identity", f) for f in have}
            if mem.get("identity", "policy_number"):
                fields["policy_number"] = mem.get("identity", "policy_number")
            res = self.tools.call("verify_identity", fields=fields)
            if res["status"] == "verified":
                self._on_verified(res, plan, nlu, text, is_rep)
                return
            st.counters.verification_failures += 1
            left = self.settings.max_verification_failures - st.counters.verification_failures
            st.log("memory", cleared="identity slots after failed match")
            mem.identity.clear()
            plan.add("verify_mismatch", attempts_left=max(left, 0))
            if left <= 0:
                self._handoff(plan, "identity could not be verified after repeated attempts")
            self.expecting = list(IDENTITY_FIELDS)
            return

        # Not enough details yet: acknowledge, explain (if needed), persuade, or stop persuading and escalate.
        options = [f for f in IDENTITY_FIELDS if f not in have and f not in mem.declined_fields
                   and f not in mem.unavailable_fields]
        still = need - len(have)
        pushback = bool(set(nlu.emotions) & {"frustration", "anger", "refusal", "confusion"}) or (
            nlu.is_question and bool(nlu.intent or nlu.aspects))
        if pushback:
            st.counters.persuasion_attempts += 1
            plan.add("explain_verification")
            if is_rep:
                plan.add("explain_consent", holder_first=self._holder_first(), pronoun=self._pronoun())
        if len(options) < still:
            plan.add("no_options_left")
            plan.add("offer_human", text="Would you like me to transfer you to a human representative?")
            st.pending_prompt = "transfer_offer"
            return
        if st.counters.persuasion_attempts > self.settings.max_persuasion_attempts and pushback:
            # Know when to stop persuading: no more explanations, escalate with context.
            plan.blocks = [b for b in plan.blocks if b[0] not in ("explain_verification", "explain_consent",
                                                                  "noted_question", "noted_hint")]
            plan.add("text", text="I don't want to keep you going in circles. Since I can't share claim details "
                                  "without verification, the best next step is a colleague who can help further.")
            self._handoff(plan, "caller declined verification after repeated explanations")
            return
        newly = [f for f in new.get("identity", []) if f in IDENTITY_FIELDS]
        if newly:
            plan.add("ack_fields", have=newly, for_rep=is_rep)
        plan.add("ask_fields", needed=still, options=options, have_any=bool(have), for_rep=is_rep)
        if pushback and st.counters.persuasion_attempts >= self.settings.max_persuasion_attempts:
            plan.add("offer_human")           # stop persuading: make the human route the explicit next step
            st.pending_prompt = "transfer_offer"
        elif pushback and st.counters.persuasion_attempts >= 2:
            plan.add("soft_human_option")
        self.expecting = options
        plan.facts.update({"details_received": [f for f in have], "still_needed": still,
                           "acceptable_remaining_options": options})

    def _on_verified(self, res: dict, plan: TurnPlan, nlu: NLUResult, text: str, is_rep: bool) -> None:
        st, mem = self.state, self.state.memory
        if is_rep:
            rep = self.tools.call("lookup_representative", rep_name=mem.get("caller", "caller_name"),
                                  party_id=res["party_id"])
            if not rep["registered"]:
                plan.add("rep_not_authorized")
                st.pending_prompt = "transfer_offer"
                return
            st.verification.verified = True
            st.verification.party_id = res["party_id"]
            st.verification.matched_fields = res["matched_fields"]
            st.verification.verified_turn = st.turn
            st.consent.required = True
            st.consent.rep_name = rep["rep_name"]
            st.consent.relationship = rep["relationship"]
            req = self.tools.call("request_consent", party_id=res["party_id"])
            plan.add("consent_requested", holder_first=self._holder_first(), request_id=req["request_id"])
            plan.required.append(req["request_id"])
            self.expecting = []
            return
        st.verification.verified = True
        st.verification.party_id = res["party_id"]
        st.verification.matched_fields = res["matched_fields"]
        st.verification.verified_turn = st.turn
        # Verified in the same turn: "I'll pull that up after verification" would be stale.
        plan.blocks = [b for b in plan.blocks if b[0] not in ("noted_hint", "noted_question")]
        holder = self.repo.policyholder(res["party_id"])
        plan.add("verified", name=" ".join(holder["name"].split(" ")[:-1]) or holder["name"])
        self.expecting = []
        self._goto(Phase.RESOLVE_INTENT, plan, nlu, text)

    # ================================================================== RESOLVE_INTENT (guided)
    def _enter_resolve(self, plan: TurnPlan, nlu: NLUResult | None, text: str) -> None:
        st, mem = self.state, self.state.memory
        plan.allow_claim_data = True
        intent = mem.intent.value if mem.intent else None
        if intent in UNSUPPORTED_INTENTS:
            plan.add("unsupported_intent", intent=intent)
            st.pending_prompt = "transfer_offer"
            return
        hints = {k: v.value for k, v in mem.case_hints.items()}
        self._select_case(plan, hints, from_memory=self._hints_from_earlier_turn(), nlu=nlu, text=text)

    def _hints_from_earlier_turn(self) -> bool:
        st = self.state
        return any(s.turn < st.turn or s.phase == Phase.VERIFY_ID for s in st.memory.case_hints.values())

    def _select_case(self, plan: TurnPlan, hints: dict, from_memory: bool, nlu: NLUResult | None, text: str,
                     within: list[str] | None = None) -> None:
        st = self.state
        all_claims = self.tools.call("list_claims")
        if not all_claims:
            plan.add("no_claims")
            st.case_candidates = []
            st.pending_prompt = "anything_else"
            return
        if hints.get("case_id") and hints["case_id"] not in {c["case_id"] for c in all_claims}:
            plan.add("cross_party", case_id=hints["case_id"])
            hints = {k: v for k, v in hints.items() if k != "case_id"}
            st.memory.case_hints.pop("case_id", None)
        cands = self.tools.call("find_claims", hints=hints, within=within) if hints else []
        if not hints and len(all_claims) == 1:
            cands = all_claims
        if len(cands) == 1:
            st.active_case_id = cands[0]["case_id"]
            st.case_candidates = []
            plan.add("case_resolved", owner=self._owner(), claim=cands[0], created_human=human_date(cands[0]["created_at"]),
                     from_memory=from_memory)
            plan.required.append(cands[0]["case_id"])
            self._goto(Phase.PROCESS_CASE, plan, nlu, text)
            return
        if len(cands) > 1:
            options = cands
            lead = f"I found {len(cands)} claims that could match {self._hint_summary()}"
        elif hints:
            # Relax the date constraint before giving up, then show everything.
            relaxed = {k: v for k, v in hints.items() if k not in ("month", "year")}
            options = self.tools.call("find_claims", hints=relaxed) if relaxed and relaxed != hints else []
            if len(options) == 1:
                st.active_case_id = options[0]["case_id"]
                plan.add("text", text=f"I didn't see an exact match for {self._hint_summary()}, but this looks like "
                                      f"the closest one.")
                plan.add("case_resolved", owner=self._owner(), claim=options[0], created_human=human_date(options[0]["created_at"]))
                plan.required.append(options[0]["case_id"])
                self._goto(Phase.PROCESS_CASE, plan, nlu, text)
                return
            options = options or all_claims
            if "cross_party" not in plan.kinds():
                plan.add("no_case_match", summary=self._hint_summary(), options=[o["label"] for o in options])
                st.case_candidates = [o["case_id"] for o in options]
                st.pending_prompt = "choose_case"
                plan.facts = {"claims_on_policy": [o["label"] for o in options]}
                return
            lead = "Here are the claims on this policy"
        else:
            options = all_claims
            lead = f"I see {len(all_claims)} claims on this policy"
        st.case_candidates = [o["case_id"] for o in options]
        st.pending_prompt = "choose_case"
        plan.add("choose_case", lead=lead, options=[o["label"] for o in options])
        plan.required += [o["case_id"] for o in options]
        plan.facts = {"claims_on_policy": [o["label"] for o in options]}

    def _resolve(self, nlu: NLUResult, plan: TurnPlan, text: str, new: dict) -> None:
        st = self.state
        plan.allow_claim_data = True
        if nlu.intent in UNSUPPORTED_INTENTS:
            plan.add("unsupported_intent", intent=nlu.intent)
            st.pending_prompt = "transfer_offer"
            return
        if nlu.act == "done":
            self._goto(Phase.POST_PROCESS, plan, nlu, text)
            return
        cands = st.case_candidates
        if nlu.choice_index is not None and cands:
            idx = nlu.choice_index
            if -len(cands) <= idx < len(cands):
                st.active_case_id = cands[idx]
                brief = claim_brief(next(c for c in self.repo.claims_for(st.verification.party_id)
                                         if c["case_id"] == cands[idx]))
                plan.add("case_resolved", owner=self._owner(), claim=brief, created_human=human_date(brief["created_at"]))
                plan.required.append(brief["case_id"])
                self._goto(Phase.PROCESS_CASE, plan, nlu, text)
                return
        if nlu.case_hints:
            narrowed = self.tools.call("find_claims", hints=nlu.case_hints, within=cands or None) if cands else []
            if narrowed:
                self._select_case(plan, nlu.case_hints, from_memory=False, nlu=nlu, text=text, within=cands)
                return
            # New hints that don't fit the offered options: the caller changed their mind -> search afresh.
            st.memory.case_hints.clear()
            for k, v in nlu.case_hints.items():
                st.memory.remember("case_hints", k, v, st.turn, st.phase)
            self._select_case(plan, dict(nlu.case_hints), from_memory=False, nlu=nlu, text=text)
            return
        if not cands and nlu.act in ("deny",):
            self._goto(Phase.POST_PROCESS, plan, nlu, text)
            return
        hints = {k: v.value for k, v in st.memory.case_hints.items()}
        self._select_case(plan, hints, from_memory=False, nlu=nlu, text=text)

    # ================================================================== PROCESS_CASE (grounded)
    def _enter_process(self, plan: TurnPlan, nlu: NLUResult | None, text: str) -> None:
        st = self.state
        if st.active_case_id not in st.discussed_cases:
            st.discussed_cases.append(st.active_case_id)
        intent = st.memory.intent.value if st.memory.intent else None
        aspects = list(INTENT_DEFAULT_ASPECTS.get(intent or "", ["status"]))
        # Questions asked earlier (e.g. during verification) are answered now.
        texts = [q.value for q in st.memory.open_questions] + ([text] if text else [])
        for t in texts:
            aspects += [a for a in _aspects(t.lower()) if a not in aspects]
        if nlu:
            aspects += [a for a in nlu.aspects if a not in aspects]
        claim = next(c for c in self.repo.claims_for(st.verification.party_id) if c["case_id"] == st.active_case_id)
        if claim["status"] == "denied" and intent in (None, "general_claim_question", "status_inquiry", "list_claims"):
            aspects += [a for a in ("denial_reason", "documents_needed") if a not in aspects]
        st.memory.open_questions = []
        self._answer(plan, aspects, " ".join(texts))

    def _process(self, nlu: NLUResult, plan: TurnPlan, text: str, new: dict) -> None:
        st = self.state
        plan.allow_claim_data = True
        if nlu.act == "done" or (st.pending_prompt == "anything_else" and nlu.act == "deny"
                                 and not nlu.aspects and not nlu.intent):
            self._goto(Phase.POST_PROCESS, plan, nlu, text)
            return
        if nlu.intent in UNSUPPORTED_INTENTS:
            plan.add("unsupported_intent", intent=nlu.intent)
            st.pending_prompt = "transfer_offer"
            return
        active = self.tools.call("get_claim", case_id=st.active_case_id)
        h = nlu.case_hints
        if h.get("case_id") and h["case_id"] != st.active_case_id and \
                self.tools.call("get_claim", case_id=h["case_id"]) is None:
            # Claim belongs to someone else (or doesn't exist): refuse without confirming either way; stay put.
            st.memory.case_hints.pop("case_id", None)
            plan.add("cross_party", case_id=h["case_id"])
            plan.add("anything_else")
            st.pending_prompt = "anything_else"
            return
        switching = (h.get("case_id") and h["case_id"] != st.active_case_id) or \
                    (h.get("case_type") and h["case_type"] != active["case_type"]) or \
                    (h.get("status") and h["status"] != active["status"] and re.search(r"\bother\b|another|\bmy\b", text,
                                                                                       re.I)) or \
                    (nlu.choice_index is not None and st.case_candidates)
        if switching:
            st.memory.case_hints.clear()
            for k, v in h.items():
                st.memory.remember("case_hints", k, v, st.turn, st.phase)
            self._goto(Phase.RESOLVE_INTENT, plan, nlu, text)
            if st.phase == Phase.RESOLVE_INTENT and nlu.choice_index is not None:
                self._resolve(nlu, plan, text, new)
            return
        if st.pending_prompt == "anything_else" and nlu.act == "affirm" and not nlu.aspects and not nlu.intent:
            plan.add("text", text="Sure — what would you like to know?")
            st.pending_prompt = None
            return
        aspects = list(nlu.aspects)
        if not aspects and nlu.intent:
            aspects = list(INTENT_DEFAULT_ASPECTS.get(nlu.intent, []))
        if not aspects:
            fu = self.tools.call("get_followup_guidance", case_id=st.active_case_id, text=text)
            if fu:
                aspects = ["followup_only"]
            elif nlu.is_question:
                plan.add("no_info")
                st.pending_prompt = "transfer_offer"
                plan.facts = self._claim_facts(active)
                return
            else:
                plan.add("text", text="Could you tell me a bit more about what you'd like to know about this claim?")
                return
        self._answer(plan, aspects, text)

    def _answer(self, plan: TurnPlan, aspects: list[str], text: str) -> None:
        st = self.state
        plan.allow_claim_data = True
        c = self.tools.call("get_claim", case_id=st.active_case_id)
        docs = c.get("documents_needed", [])
        guidance = self.tools.call("get_document_guidance", case_id=c["case_id"]) if docs else {}
        topics = [ASPECT_TO_TOPIC[a] for a in aspects if a in ASPECT_TO_TOPIC]
        followups = self.tools.call("get_followup_guidance", case_id=c["case_id"], text=text, topics=topics)
        fu_by_topic = {f["topic"]: f["text"] for f in followups}
        low = text.lower()
        sections: list[str] = []
        seen = set()
        for a in aspects:
            if a in seen:
                continue
            seen.add(a)
            s = self._section(a, c, docs, guidance, fu_by_topic, low)
            if s:
                sections.append(s)
                st.add_topic(c["case_id"], a if a in ASPECT_LABELS else "general")
        # Follow-up rules matched by the caller's wording but not covered by an aspect.
        covered = {ASPECT_TO_TOPIC.get(a) for a in aspects}
        for topic, msg in fu_by_topic.items():
            if topic not in covered and msg not in sections:
                sections.append(msg)
                st.add_topic(c["case_id"], next((a for a, t in ASPECT_TO_TOPIC.items() if t == topic), "general"))
        if not sections:
            sections.append(self.repo.guidelines["claim_followup_fallback"]["en"] if docs else
                            self._section("status", c, docs, guidance, fu_by_topic, low))
        plan.add("answer", sections=sections)
        plan.add("anything_else")
        st.pending_prompt = "anything_else"
        plan.facts = self._claim_facts(c, guidance, followups)
        plan.facts["caller_question"] = text

    def _section(self, a: str, c: dict, docs: list[str], g: dict, fu: dict, low: str) -> str:
        cid = c["case_id"]
        if a == "status":
            s = {"denied": f"Claim {cid} ({c['case_type']}, filed {c['created_at_human']}) is currently denied.",
                 "open": f"Claim {cid} ({c['case_type']}, filed {c['created_at_human']}) is open and still in progress "
                         f"— nothing has been paid yet.",
                 "closed": f"Claim {cid} ({c['case_type']}, filed {c['created_at_human']}) is closed — "
                           f"{c['summary'].lower()}."}.get(c["status"], f"Claim {cid} is {c['status']}.")
            return s
        if a == "denial_reason":
            if c["status"] != "denied":
                return f"This claim wasn't denied — its status is {c['status']}."
            return f"It was denied because {c.get('denial_reason', 'of missing information')}."
        if a == "documents_needed":
            if not docs:
                return "There are no outstanding documents requested on this claim."
            return f"To have it reconsidered, we need the {join_words(docs)}."
        if a == "appeal_deadline":
            if not c.get("appeal_deadline"):
                return ""
            if c["appeal_deadline_passed"]:
                return (f"The appeal deadline on file was {c['appeal_deadline_human']}, which has already passed "
                        f"(as of {human_date(c['as_of'])}). A human claims representative would need to review what "
                        f"options remain.")
            return (f"The appeal deadline is {c['appeal_deadline_human']} — that's {c['appeal_days_remaining']} days "
                    f"from today.")
        if a == "document_requirements":
            if not docs:
                return "There are no outstanding documents requested on this claim."
            parts = []
            mentioned = [d for d in g.get("documents", []) if any(w in low for w in d["document"].lower().split()
                                                                   if w not in ("report", "note"))]
            for d in (mentioned or g.get("documents", [])):
                parts.append(f"For the {d['document']}: {d['requirements']}" if d["requirements"]
                             else f"For the {d['document']}: {g['default_guidance']}")
            if g.get("case_type_guidance"):
                parts.append(g["case_type_guidance"])
            return "\n".join(parts)
        if a == "document_alternatives":
            if not docs:
                return "There are no outstanding documents requested on this claim."
            mentioned = [d for d in g.get("documents", []) if any(w in low for w in d["document"].lower().split()
                                                                   if w not in ("report", "note"))]
            chosen = mentioned or g.get("documents", [])
            parts = [f"About the {d['document']}: {d['alternatives']}" for d in chosen]
            return "\n".join(parts)
        if a in ASPECT_TO_TOPIC:
            return fu.get(ASPECT_TO_TOPIC[a], "")
        if a == "amounts":
            if c["status"] == "closed":
                return (f"The allowed maximum was {money(c['allowed_max_amount'])}, the expected reimbursement was "
                        f"{money(c['expected_reimbursement_amount'])}, and {money(c['net_pay'])} was paid.")
            return (f"On file: allowed maximum {money(c['allowed_max_amount'])}, expected reimbursement "
                    f"{money(c['expected_reimbursement_amount'])}, and {money(c['net_pay'])} paid so far.")
        return ""

    def _claim_facts(self, c: dict, guidance: dict | None = None, followups: list | None = None) -> dict:
        claim = {k: v for k, v in c.items() if k not in ("party_id",)}
        for k in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee"):
            if k in claim:
                claim[k + "_formatted"] = money(claim[k])
        facts = {"claim": claim, "today": human_date(c.get("as_of"))}
        if guidance:
            facts["document_guidance"] = guidance
        if followups:
            facts["followup_guidance"] = followups
        facts["other_claims_on_policy"] = [b["label"] for b in self.tools.call("list_claims")
                                           if b["case_id"] != c["case_id"]]
        return facts

    # ================================================================== POST_PROCESS (strict, consent-based)
    def _enter_post(self, plan: TurnPlan, nlu: NLUResult | None, text: str) -> None:
        st = self.state
        summary = self.tools.call("compose_summary")
        st.email["draft"] = summary
        plan.add("offer_email", to_masked=summary["to_masked"])
        plan.required.append(summary["to_masked"])
        plan.facts = {"email_to_masked": summary["to_masked"], "summary_sections": ["what we discussed",
                                                                                   "claim status/outcome",
                                                                                   "follow-up items / next steps"]}
        st.pending_prompt = "email_offer"

    def _post(self, nlu: NLUResult, plan: TurnPlan, text: str, new: dict) -> None:
        st = self.state
        summary = st.email.get("draft") or self.tools.call("compose_summary")
        on_file = summary["to"]
        if nlu.identity.get("email") and nlu.identity["email"] != on_file:
            plan.add("email_other_address")
            plan.facts = {"email_to_masked": summary["to_masked"]}
            return
        wants_send = nlu.act == "affirm" or bool(re.search(r"\b(send|email) (it|me|the summary)\b", text, re.I))
        if wants_send and nlu.act != "deny":
            res = self.tools.call("send_email_summary", summary=summary)
            st.email.update(decision="sent", sent=res)
            plan.add("email_sent", to_masked=res["to_masked"])
            plan.required.append(res["to_masked"])
            plan.add("closing")
            self._goto(Phase.COMPLETE, plan, nlu, text)
            return
        if nlu.act in ("deny", "done"):
            st.email.update(decision="skipped")
            plan.add("email_skipped")
            plan.add("closing")
            self._goto(Phase.COMPLETE, plan, nlu, text)
            return
        if nlu.aspects or nlu.intent or nlu.case_hints:
            # A new claim question before the email decision: go back (allowed by the SOP graph).
            target = Phase.PROCESS_CASE if not nlu.case_hints else Phase.RESOLVE_INTENT
            if target == Phase.RESOLVE_INTENT:
                st.memory.case_hints.clear()
                for k, v in nlu.case_hints.items():
                    st.memory.remember("case_hints", k, v, st.turn, st.phase)
                self._goto(target, plan, nlu, text)
            else:
                self._goto(Phase.PROCESS_CASE, plan, None, "")
                plan.blocks = [b for b in plan.blocks if b[0] != "answer"]
                self._process(nlu, plan, text, new)
            return
        plan.add("offer_email", to_masked=summary["to_masked"])
        plan.required.append(summary["to_masked"])

    # ================================================================== shared helpers
    def _handoff(self, plan: TurnPlan, reason: str, announce: bool = True) -> None:
        st = self.state
        res = self.tools.call("transfer_to_human", reason=reason)
        st.handoff = res
        plan.add("handoff", ticket=res["ticket"])
        plan.required.append(res["ticket"])
        st.log("transition", frm=st.phase.value, to=Phase.HANDOFF.value)
        st.phase = Phase.HANDOFF
        plan.phase = Phase.HANDOFF
        st.pending_prompt = None

    def _owner(self) -> str:
        return f"{self._holder_first()}'s" if self.state.consent.required else "your"

    def _holder_first(self) -> str:
        pid = self.state.verification.party_id
        if pid:
            return self.repo.policyholder(pid)["name"].split(" ")[0]
        name = self.state.memory.get("identity", "full_name")
        return name.split(" ")[0] if name else "the policyholder"

    def _pronoun(self) -> str:
        rel = (self.state.memory.get("caller", "relationship") or "").lower()
        return {"mother": "her", "father": "his"}.get(rel, "their")

    def _redirect_hint(self) -> str:
        st = self.state
        if st.phase == Phase.VERIFY_ID:
            if st.consent.status == "pending":
                return "We're still waiting on the policyholder's consent — just say \"check\" when they've approved."
            return "I'm happy to help with your claim once we've verified your identity."
        if st.phase == Phase.RESOLVE_INTENT:
            return "Which of your claims can I help you with?"
        if st.phase == Phase.PROCESS_CASE:
            return f"Is there anything else about claim {st.active_case_id} I can help with?"
        if st.phase == Phase.POST_PROCESS:
            return "Would you like me to email the summary of today's conversation?"
        return ""

    def _chitchat(self, text: str) -> str:
        low = text.lower()
        if re.search(r"how are you", low):
            return "I'm doing well, thanks for asking!"
        if re.search(r"who are you|what can you do|what do you do", low):
            return ("I'm a virtual claims assistant — I can check claim status, explain denials, help with required "
                    "documents and next steps, and email you a summary.")
        if re.search(r"thank", low):
            return "You're welcome!"
        if re.search(r"^\s*(hi|hello|hey|good)", low):
            return "Hi there!"
        return ""

    def _nlu_ctx(self) -> dict:
        st = self.state
        last_agent = next((m.text for m in reversed(st.transcript) if m.role == "agent"), "")
        options = []
        if st.verification.verified and st.case_candidates:
            options = [claim_brief(c)["label"] for cid in st.case_candidates
                       for c in self.repo.claims_for(st.verification.party_id) if c["case_id"] == cid]
        return {"phase": st.phase.value, "pending_prompt": st.pending_prompt, "expecting": self.expecting,
                "last_agent": last_agent, "options": options, "today": self.settings.today()}

    def _finish(self, plan: TurnPlan, user_text: str) -> dict:
        st = self.state
        history = [{"role": m.role, "text": m.text} for m in st.transcript[:-1]] if user_text else []
        reply, meta = render(plan, llm=self.llm, user_text=user_text, history=history, repo=self.repo,
                             verified=st.verification.verified and (not st.consent.required
                                                                    or st.consent.status == "approved"),
                             log=st.log)
        meta["plan"] = plan.kinds()
        meta["emotions"] = plan.emotions
        st.transcript.append(Message(role="agent", text=reply, phase=st.phase.value, meta=meta))
        return {"reply": reply, "meta": meta, "state": self.snapshot()}

    # ================================================================== UI snapshot
    def snapshot(self) -> dict:
        st = self.state
        mem = st.memory
        ident = {f: {"value": mask_value(f, s.value), "turn": s.turn, "phase": s.phase.value, "source": s.source}
                 for f, s in mem.identity.items()}
        hints = {k: {"value": s.value, "turn": s.turn, "phase": s.phase.value, "source": s.source}
                 for k, s in mem.case_hints.items()}
        progress = []
        for p in WORKFLOW:
            if st.phase == p:
                status = "active"
            elif st.phase in (Phase.COMPLETE,) or (WORKFLOW.index(p) < WORKFLOW.index(st.phase)
                                                    if st.phase in WORKFLOW else False):
                status = "done"
            else:
                status = "todo"
            progress.append({"phase": p.value, "status": status})
        email = dict(st.email)
        if "draft" in email:
            email["draft"] = {k: v for k, v in email["draft"].items() if k != "to"}
        return {
            "session_id": st.session_id, "phase": st.phase.value, "turn": st.turn, "progress": progress,
            "llm": {"enabled": bool(self.llm), "provider": self.llm.cfg.provider if self.llm else None,
                    "model": self.llm.cfg.resolved_model() if self.llm else None,
                    "calls": self.llm.calls[-10:] if self.llm else []},
            "verification": {"verified": st.verification.verified, "party_id": st.verification.party_id,
                             "matched_fields": st.verification.matched_fields,
                             "required": self.settings.min_identity_fields,
                             "collected": ident, "declined": mem.declined_fields,
                             "unavailable": mem.unavailable_fields,
                             "failures": st.counters.verification_failures},
            "consent": st.consent.model_dump(),
            "caller": {k: s.value for k, s in mem.caller.items()},
            "memory": {"case_hints": hints,
                       "intent": ({"value": mem.intent.value, "turn": mem.intent.turn, "phase": mem.intent.phase.value}
                                  if mem.intent else None),
                       "open_questions": [q.value for q in mem.open_questions]},
            "active_case_id": st.active_case_id, "case_candidates": st.case_candidates,
            "discussed_cases": st.discussed_cases, "topics": st.topics,
            "pending_prompt": st.pending_prompt, "counters": st.counters.model_dump(),
            "email": email, "handoff": st.handoff, "demo_today": self.settings.today().isoformat(),
            "trace": [e.model_dump() for e in st.trace[-60:]],
        }
