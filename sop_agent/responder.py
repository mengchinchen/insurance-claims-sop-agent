"""Turns a TurnPlan (decided by the SOP engine) into the caller-facing reply.

1. A deterministic template draft is always rendered. It is correct by construction and is the fallback.
2. If an LLM is configured, it rewrites/answers within phase-specific freedom:
     strict   -> rephrase the draft only (VERIFY_ID / POST_PROCESS)
     guided   -> rephrase + respond to the caller's wording (RESOLVE_INTENT)
     grounded -> answer the caller's question using only the FACTS packet (PROCESS_CASE)
3. The LLM output is validated (claim-data leak before verification, PII leak, ungrounded numbers/claim IDs,
   required content). Any violation -> the safe draft is used instead.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .normalize import FIELD_LABELS, join_words
from .state import PHASE_FREEDOM, Phase

EMOTION_PRIORITY = ["anger", "frustration", "anxiety", "refusal", "confusion"]
EMPATHY = {
    "anger": ["I'm really sorry — I can hear how frustrating this has been, and I do want to get you answers.",
              "I completely understand why you're upset, and I'm sorry for the hassle."],
    "frustration": ["I hear you, and I'm sorry this is taking longer than you'd like.",
                    "I understand — repeating yourself is frustrating, and I appreciate your patience."],
    "anxiety": ["I understand this is stressful, especially when it involves a claim you're counting on. "
                "Let's take it one step at a time — I'm here to help.",
                "That sounds worrying, and it's completely understandable. We'll sort through it together."],
    "refusal": ["That's completely okay — you don't have to share anything you're not comfortable with.",
                "No problem at all, that's your choice."],
    "confusion": ["Good question — let me explain.", "Sorry for the confusion — let me clarify."],
}


@dataclass
class TurnPlan:
    blocks: list[tuple[str, dict]] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)        # grounded data the LLM may use
    required: list[str] = field(default_factory=list)          # substrings the final reply must contain
    phase: Phase = Phase.VERIFY_ID
    allow_claim_data: bool = False

    def add(self, kind: str, **params: Any) -> None:
        self.blocks.append((kind, params))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.blocks]


# --------------------------------------------------------------------------- templates
def render_draft(plan: TurnPlan, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(0)
    parts: list[str] = []
    emos = [e for e in EMOTION_PRIORITY if e in plan.emotions]
    if emos:
        parts.append(rng.choice(EMPATHY[emos[0]]))
    for kind, p in plan.blocks:
        text = _block(kind, p)
        if text:
            parts.append(text)
    return "\n\n".join(_merge_short(parts))


def _merge_short(parts: list[str]) -> list[str]:
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < 140 and "\n" not in out[-1] and "\n" not in p and len(p) < 220:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _labels(fields: list[str]) -> list[str]:
    return [FIELD_LABELS.get(f, f) for f in fields]


def _block(kind: str, p: dict) -> str:
    if kind == "greeting":
        return ("Thank you for calling Claims Support — I'm your virtual claims assistant. Before I can look at any "
                "claim, I need to verify identity. Please share at least 3 of the following: your full name, date of "
                "birth, phone number on file, email on file, or the last 4 digits of your SSN. If you're calling on "
                "behalf of a family member, just let me know.")
    if kind == "explain_verification":
        return ("Claim details include protected health and financial information, so I'm required to confirm "
                "identity with at least 3 details before I can discuss them. It's the same check that keeps anyone "
                "else from accessing your claim.")
    if kind == "explain_consent":
        return (f"Because you're calling on {p['holder_first']}'s behalf, {p['holder_first']} also needs to approve "
                f"sharing claim information with you — that protects {p['pronoun']} privacy.")
    if kind == "noted_hint":
        return (f"I've noted that you're calling about {p['summary']}. I'll pull that up as soon as we've verified "
                f"identity.")
    if kind == "noted_question":
        return "I've noted your question and I'll answer it right after verification."
    if kind == "rep_detected":
        return (f"Thanks for helping your {p['relationship']}. I'll need to verify {p['holder']}'s identity, and "
                f"then {p['holder']} will need to approve you discussing the claim.")
    if kind == "ack_fields":
        own = {"full_name": "name", "dob": "date of birth", "phone": "phone number", "email": "email",
               "id_last4": "SSN last 4"}
        who = "their" if p.get("for_rep") else "your"
        return f"Thank you — I have {who} {join_words([own.get(f, f) for f in p['have']])}."
    if kind == "field_declined_ack":
        return f"No problem, we can skip the {join_words(_labels(p['fields']), 'and')}."
    if kind == "ask_fields":
        n, opts = p["needed"], _labels(p["options"])
        whose = " for the policyholder" if p.get("for_rep") else ""
        if p.get("have_any"):
            ask = f"I just need {n} more detail{'s' if n > 1 else ''}{whose}"
        else:
            ask = f"To get started, I need at least {n} details{whose}"
        if opts:
            return f"{ask} — any of: {join_words(opts, 'or')}."
        return f"{ask}."
    if kind == "soft_human_option":
        return ("And if you'd rather not continue with me, I can transfer you to a human representative at any time "
                "(they'll need to verify identity too).")
    if kind == "no_options_left":
        return ("It looks like we don't have enough verification options left that you can provide, so the best "
                "path is a human representative who can verify you another way.")
    if kind == "verify_mismatch":
        left = p["attempts_left"]
        base = ("I'm sorry, I wasn't able to match those details to our records. For security I can't say which "
                "detail didn't match.")
        if left <= 0:
            return base
        return (base + " Could you double-check and share at least 3 details again? "
                f"(You have {left} more attempt{'s' if left != 1 else ''} before I connect you with a human "
                f"representative.)")
    if kind == "rep_need_name":
        return "And could I get your own full name, since you're calling on their behalf?"
    if kind == "rep_not_authorized":
        return ("Thank you. I don't see an authorization on file for you to discuss this policy, so I'm not able to "
                "share any claim information. The policyholder can call us directly, or I can connect you with a "
                "human representative who can help set up an authorization. Would you like me to transfer you?")
    if kind == "consent_requested":
        return (f"Thank you, those details check out. Since you're calling on {p['holder_first']}'s behalf, I've just "
                f"sent a consent request to {p['holder_first']}'s phone on file (reference {p['request_id']}). Once "
                f"{p['holder_first']} approves it, I can go over the claim with you. Let me know when it's been "
                f"approved, or just say \"check\" and I'll look again.")
    if kind == "consent_pending":
        return (f"I've checked again — the consent request is still pending. As soon as {p['holder_first']} taps "
                f"approve, we can continue. (Check {p['polls']} of {p['max_polls']}.)")
    if kind == "consent_approved":
        return f"Great news — {p['holder_first']} has approved the request, so I can now discuss the claim with you."
    if kind == "consent_timeout":
        return ("Unfortunately the consent request still hasn't been approved, so I can't share claim information on "
                "this call. I'm transferring you to a human representative who can help with other options.")
    if kind == "verified":
        return f"Thank you, {p['name']} — you're verified."
    if kind == "offer_human":
        return p.get("text") or ("If you'd prefer, I can transfer you to a human representative — they'll also need to "
                                 "verify identity, but they may have more options. Would you like me to do that?")
    if kind == "handoff":
        return (f"I'm transferring you to a human representative now (ticket {p['ticket']}). I've passed along the "
                f"context of our conversation so you won't have to start over. Thank you for your patience.")
    if kind == "oos":
        if p["count"] >= p["max"]:
            return ("I'm only able to help with insurance claims and policy support on this line, and it seems what "
                    "you need is outside that. Would you like me to connect you with a human representative?")
        return ("I'm sorry, I can only help with insurance claims and related support, so I can't answer that one. "
                + p.get("redirect", ""))
    if kind == "chitchat":
        return p.get("text", "")
    if kind == "unclear":
        return p.get("text", "Sorry, I didn't quite catch that.")
    if kind == "case_resolved":
        c = p["claim"]
        lead = "Based on what you mentioned earlier, I found" if p.get("from_memory") else "I found"
        return f"{lead} {p.get('owner', 'your')} {c['case_type']} claim {c['case_id']} from {p['created_human']}."
    if kind == "choose_case":
        opts = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(p["options"]))
        return f"{p.get('lead', 'I found more than one claim that could match')}. Which one are you calling about?\n{opts}"
    if kind == "no_case_match":
        opts = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(p["options"]))
        return (f"I couldn't find a claim matching {p['summary']} on this policy. Here are the claims I do see — which "
                f"one would you like to talk about?\n{opts}")
    if kind == "no_claims":
        return ("I don't see any claims on file for this policy yet. If you need to file a new claim, a human "
                "representative can help with that. Is there anything else I can help you with?")
    if kind == "ask_intent":
        return "How can I help you today — is this about one of your claims?"
    if kind == "unsupported_intent":
        what = {"new_claim": "filing a new claim", "policy_change": "policy, billing, or coverage changes"}.get(
            p["intent"], "that request")
        return (f"I'm not able to handle {what} on this line — that's done by our human representatives. I can "
                f"transfer you, or help with an existing claim. Which would you prefer?")
    if kind == "answer":
        out: list[str] = []
        for sec in p["sections"]:
            if out and "\n" not in sec and "\n" not in out[-1] and len(out[-1]) + len(sec) < 320:
                out[-1] += " " + sec
            else:
                out.append(sec)
        return "\n\n".join(out)
    if kind == "cross_party":
        return (f"I'm not able to find claim {p['case_id']} on this policy, so I can't share details about it. I can "
                f"only discuss claims on the verified policy.")
    if kind == "no_info":
        return ("I don't have that information in the claim record, so I don't want to guess. A human claims "
                "representative can look into it — would you like me to transfer you, or is there something else about "
                "this claim I can help with?")
    if kind == "anything_else":
        return "Is there anything else I can help you with on this claim or another one?"
    if kind == "offer_email":
        return (f"Before we wrap up — would you like me to email a summary of today's conversation to the email on "
                f"file ({p['to_masked']})? It will include what we discussed, the claim status, and your next steps. "
                f"Just say yes to send it or no to skip.")
    if kind == "email_other_address":
        return ("For security, I can only send the summary to the email address already on file. "
                "Would you like me to send it there?")
    if kind == "email_sent":
        return f"Done — I've sent the summary to {p['to_masked']}."
    if kind == "email_skipped":
        return "No problem, I won't send an email."
    if kind == "closing":
        return "Thank you for calling Claims Support. Take care!"
    if kind == "complete_already":
        return ("This conversation has been wrapped up. If you need anything else, please start a new conversation "
                "and I'll be glad to help.")
    if kind == "text":
        return p["text"]
    return ""


# --------------------------------------------------------------------------- LLM phrasing + validation
BASE_RULES = """You are the voice of "Claims Support", a virtual insurance claims assistant, in a text chat.
A deterministic SOP engine has already decided WHAT happens this turn (phase, gates, actions). You only decide HOW
to say it. Style: warm, professional, concise, natural (no bullet spam, no markdown headings). Mirror the caller's
emotion briefly before moving on. Never mention "SOP", "phase", "draft", "facts", or internal tools.
Hard rules:
- Never state any number, date, amount, claim ID, document name, or policy rule that is not in FACTS or DRAFT.
- Never reveal personal data from records (phone, email, DOB, SSN); use masked forms only if provided.
- Keep every request/question the DRAFT asks for (e.g. which identity details are needed, yes/no to email).
- Do not promise outcomes (approval, payment) and do not give legal or medical advice.
- Output only the reply text."""

PHASE_RULES = {
    "strict": """Freedom: STRICT. Rephrase the DRAFT naturally and respond briefly to what the caller said, but do not add
any new information or steps. The caller is NOT verified yet (unless DRAFT says so): you must not discuss or hint at
any claim details, statuses, reasons, or amounts beyond words the caller used themselves.""",
    "guided": """Freedom: GUIDED. Rephrase the DRAFT naturally, acknowledging the caller's wording. You may reorder for
flow, but keep all listed options exactly (IDs, dates) and keep the closing question.""",
    "grounded": """Freedom: GROUNDED. Answer the caller's actual question using ONLY the FACTS (claim record + retrieved
guidance). You may select, summarize and explain in plain language, and you may leave out irrelevant draft content,
but every fact you state must come from FACTS. If FACTS don't answer the question, say you don't have that
information and offer a human representative. End with the DRAFT's closing question if it has one.""",
}


def phrase_with_llm(llm, plan: TurnPlan, draft: str, user_text: str, history: list[dict]) -> str:
    freedom = PHASE_FREEDOM[plan.phase]
    system = BASE_RULES + "\n\n" + PHASE_RULES[freedom]
    hist = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in history[-6:])
    prompt = (f"CONVERSATION SO FAR:\n{hist}\n\nCALLER'S LATEST MESSAGE:\n{user_text}\n\n"
              f"DETECTED CALLER EMOTION: {', '.join(plan.emotions) or 'neutral'}\n\n"
              f"FACTS (the only data you may use):\n{json.dumps(plan.facts, indent=1, default=str)[:6000]}\n\n"
              f"DRAFT (correct content, robotic wording):\n{draft}\n\nWrite the reply now.")
    return llm.generate(system, prompt)


NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CASE_RE = re.compile(r"\bCL-\d+\b", re.I)


def validate_reply(reply: str, *, plan: TurnPlan, draft: str, user_texts: str, repo, verified: bool) -> list[str]:
    problems = []
    if not reply or len(reply) > 2500:
        return ["empty_or_too_long"]
    low, ulow, dlow = reply.lower(), user_texts.lower(), draft.lower()
    facts_txt = json.dumps(plan.facts, default=str).lower()
    allowed_src = " ".join([facts_txt, dlow, ulow])
    if not plan.allow_claim_data or not verified:
        for marker in repo.sensitive_claim_markers:
            if marker in low and marker not in ulow and marker not in dlow:
                problems.append(f"claim_leak:{marker[:30]}")
    reply_digits = re.sub(r"\D", "", reply)
    for marker in repo.sensitive_pii_markers:
        m = marker.lower()
        if m.isdigit() or re.fullmatch(r"[\d-]+", m):
            hit = re.sub(r"\D", "", m) in reply_digits and re.sub(r"\D", "", m) not in re.sub(r"\D", "", user_texts)
            if hit and len(re.sub(r"\D", "", m)) >= 8:
                problems.append("pii_leak")
        elif m in low and m not in ulow:
            problems.append("pii_leak")
    allowed_nums = {n.replace(",", "") for n in NUM_RE.findall(allowed_src)}
    allowed_nums |= {n.lstrip("0") for n in allowed_nums}
    for n in NUM_RE.findall(reply):
        v = n.replace(",", "").rstrip(".")
        if v in allowed_nums or v.lstrip("0") in allowed_nums:
            continue
        try:
            if float(v) <= 10:
                continue
        except ValueError:
            pass
        problems.append(f"ungrounded_number:{v}")
    for cid in CASE_RE.findall(reply):
        if cid.lower() not in allowed_src:
            problems.append(f"ungrounded_case:{cid}")
    for req in plan.required:
        if req.lower() not in low:
            problems.append(f"missing_required:{req}")
    return problems


def render(plan: TurnPlan, *, llm, user_text: str, history: list[dict], repo, verified: bool, log) -> tuple[str, dict]:
    draft = render_draft(plan, random.Random(len(history)))
    meta: dict[str, Any] = {"mode": "template", "freedom": PHASE_FREEDOM[plan.phase]}
    if llm is None or not llm.enabled or not plan.blocks:
        return draft, meta
    try:
        reply = phrase_with_llm(llm, plan, draft, user_text, history)
    except Exception as exc:
        meta.update(mode="template (llm error)", error=str(exc)[:160])
        log("llm", stage="respond", ok=False, error=str(exc)[:160])
        return draft, meta
    user_texts = " ".join(m["text"] for m in history if m["role"] == "user") + " " + user_text
    problems = validate_reply(reply, plan=plan, draft=draft, user_texts=user_texts, repo=repo, verified=verified)
    if problems:
        meta.update(mode="template (llm reply rejected)", problems=problems[:6], rejected=reply[:400])
        log("guard", check="response_validator", ok=False, problems=problems[:6])
        return draft, meta
    meta["mode"] = "llm"
    log("guard", check="response_validator", ok=True)
    return reply, meta
