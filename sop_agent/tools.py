"""Phase-gated tools over the fixtures.

Every tool call goes through Toolbox.call(), which enforces the SOP:
  * the tool must be allowed in the current phase, and
  * claim/PII tools require a verified (and, for representatives, consented) caller,
  * claim lookups are scoped to the verified party — a caller can never read another party's claim.
The LLM never calls tools directly; the orchestrator does, so the LLM cannot bypass these gates.
"""
from __future__ import annotations

import json
import re
import smtplib
import time
import uuid
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from .config import Settings
from .data import Repository
from .normalize import (MONTH_NAMES, human_date, join_words, mask_email, money, norm_email, norm_last4,
                        norm_name, norm_phone, norm_policy)
from .state import Phase, SessionState


class ToolGateError(PermissionError):
    pass


PHASE_TOOLS: dict[Phase, set[str]] = {
    Phase.VERIFY_ID: {"verify_identity", "lookup_representative", "request_consent", "check_consent", "transfer_to_human"},
    Phase.RESOLVE_INTENT: {"list_claims", "find_claims", "transfer_to_human"},
    Phase.PROCESS_CASE: {"get_claim", "get_document_guidance", "get_followup_guidance", "list_claims", "find_claims",
                         "transfer_to_human"},
    Phase.POST_PROCESS: {"compose_summary", "send_email_summary", "transfer_to_human"},
    Phase.COMPLETE: set(),
    Phase.HANDOFF: set(),
}
REQUIRES_VERIFIED = {"list_claims", "find_claims", "get_claim", "get_document_guidance", "get_followup_guidance",
                     "compose_summary", "send_email_summary"}

ASPECT_LABELS = {
    "status": "Current claim status",
    "denial_reason": "Why the claim was denied",
    "documents_needed": "Documents needed to move the claim forward",
    "document_requirements": "How to prepare the requested documents",
    "document_alternatives": "What to do if a requested document is unavailable",
    "submission_method": "How to submit documents",
    "submission_timing": "When to submit documents",
    "processing_time": "Processing time after submission",
    "receipt_confirmation": "How to confirm the documents were received",
    "amounts": "Payment and reimbursement amounts",
    "appeal_deadline": "Appeal deadline",
    "general": "General claim questions",
}


class Toolbox:
    def __init__(self, repo: Repository, settings: Settings, state: SessionState):
        self.repo, self.settings, self.state = repo, settings, state

    # ------------------------------------------------------------------ gatekeeping
    def call(self, name: str, **kwargs: Any) -> Any:
        phase = self.state.phase
        if name not in PHASE_TOOLS.get(phase, set()):
            self.state.log("gate", tool=name, allowed=False, reason=f"tool not allowed in {phase.value}")
            raise ToolGateError(f"Tool '{name}' is not allowed in phase {phase.value}")
        if name in REQUIRES_VERIFIED:
            if not self.state.verification.verified:
                self.state.log("gate", tool=name, allowed=False, reason="caller not verified")
                raise ToolGateError(f"Tool '{name}' requires a verified caller")
            if self.state.consent.required and self.state.consent.status != "approved":
                self.state.log("gate", tool=name, allowed=False, reason="policyholder consent missing")
                raise ToolGateError(f"Tool '{name}' requires policyholder consent")
        result = getattr(self, f"_t_{name}")(**kwargs)
        self.state.log("tool", tool=name, args=_redact(kwargs), result=_summarize(result))
        return result

    # ------------------------------------------------------------------ VERIFY_ID
    def _t_verify_identity(self, fields: dict[str, str]) -> dict:
        provided = {k: v for k, v in fields.items() if v and k in ("full_name", "dob", "phone", "email", "id_last4")}
        policy = fields.get("policy_number")
        need = self.settings.min_identity_fields
        if len(provided) < need:
            return {"status": "insufficient", "provided": sorted(provided), "needed_more": need - len(provided)}
        matches = []
        for rec in self.repo.policyholders:
            matched, mismatched = _compare(rec, provided)
            if policy and norm_policy(policy) != rec["policy_number"]:
                mismatched.append("policy_number")
            if len(matched) >= need and not mismatched:
                matches.append((rec, matched))
        if len(matches) == 1:
            rec, matched = matches[0]
            return {"status": "verified", "party_id": rec["party_id"], "matched_fields": matched}
        # Do not reveal which field was wrong (prevents enumeration of PII).
        return {"status": "mismatch", "provided": sorted(provided)}

    def _t_lookup_representative(self, rep_name: str | None, party_id: str) -> dict:
        for r in self.repo.representatives:
            if r["buyer_party_id"] == party_id and rep_name and norm_name(r["rep_name"]) == norm_name(rep_name):
                return {"registered": True, "relationship": r["relationship"], "rep_name": r["rep_name"]}
        return {"registered": False}

    def _t_request_consent(self, party_id: str) -> dict:
        c = self.state.consent
        scenarios = self.repo.consent_scenarios
        c.scenario = c.scenario if c.scenario in scenarios else "default"
        c.request_id = "CNS-" + uuid.uuid4().hex[:6].upper()
        c.polls = 0
        c.status = scenarios[c.scenario]["status_sequence"][0]
        return {"request_id": c.request_id, "status": c.status, "sent_to": "policyholder's phone on file"}

    def _t_check_consent(self) -> dict:
        c = self.state.consent
        seq = self.repo.consent_scenarios[c.scenario]["status_sequence"]
        c.polls += 1
        if c.polls < len(seq):
            c.status = seq[c.polls]
        elif seq[-1] == "pending":
            c.status = "timeout"
        return {"request_id": c.request_id, "status": c.status, "polls": c.polls}

    # ------------------------------------------------------------------ claims
    def _party(self) -> str:
        return self.state.verification.party_id  # type: ignore[return-value]

    def _t_list_claims(self) -> list[dict]:
        return [claim_brief(c) for c in sorted(self.repo.claims_for(self._party()), key=lambda c: c["created_at"],
                                               reverse=True)]

    def _t_find_claims(self, hints: dict[str, Any], within: list[str] | None = None) -> list[dict]:
        out = []
        for c in self.repo.claims_for(self._party()):
            if within and c["case_id"] not in within:
                continue
            if hints.get("case_id") and c["case_id"] != hints["case_id"]:
                continue
            if hints.get("case_type") and c["case_type"] != hints["case_type"]:
                continue
            if hints.get("status") and c["status"] != hints["status"]:
                continue
            created = date.fromisoformat(c["created_at"])
            if hints.get("month") and created.month != int(hints["month"]):
                continue
            if hints.get("year") and created.year != int(hints["year"]):
                continue
            out.append(claim_brief(c))
        return sorted(out, key=lambda c: c["created_at"], reverse=True)

    def _t_get_claim(self, case_id: str) -> dict | None:
        c = next((c for c in self.repo.claims_for(self._party()) if c["case_id"] == case_id), None)
        if not c:
            return None
        rec = dict(c)
        rec["created_at_human"] = human_date(c["created_at"])
        if c.get("appeal_deadline"):
            dl = date.fromisoformat(c["appeal_deadline"])
            days = (dl - self.settings.today()).days
            rec["appeal_deadline_human"] = human_date(c["appeal_deadline"])
            rec["appeal_days_remaining"] = days
            rec["appeal_deadline_passed"] = days < 0
        rec["as_of"] = self.settings.today().isoformat()
        rec["field_meanings"] = {k: v["description"] for k, v in self.repo.claim_schema["field_descriptions"].items()}
        return rec

    def _t_get_document_guidance(self, case_id: str) -> dict:
        claim = self._t_get_claim(case_id)
        if not claim:
            return {}
        g = self.repo.guidelines
        docs = []
        for doc in claim.get("documents_needed", []):
            key = _match_doc_key(doc, g["document_guidance"])
            alt_key = _match_doc_key(doc, g["document_alternative_guidance"])
            docs.append({
                "document": doc,
                "requirements": g["document_guidance"][key]["en"] if key else None,
                "alternatives": (g["document_alternative_guidance"][alt_key]["en"] if alt_key
                                 else g["document_alternative_guidance"]["default"]["en"]),
            })
        ct = g["case_type_guidance"].get(claim["case_type"], {}).get("en")
        return {"case_id": case_id, "default_guidance": g["default_guidance"]["en"], "case_type_guidance": ct,
                "documents": docs,
                "human_review_rule": g["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]["en"]}

    def _t_get_followup_guidance(self, case_id: str, text: str, topics: list[str] | None = None) -> list[dict]:
        """Deterministic retrieval of follow-up rules: phrase match on the caller's words, or explicit topics."""
        claim = self._t_get_claim(case_id)
        if not claim:
            return []
        g = self.repo.guidelines
        docs = claim.get("documents_needed", [])
        low = " " + re.sub(r"\s+", " ", text.lower()) + " "
        out = []
        for entry in g["claim_followup_guidance"]:
            if entry.get("requires_documents") and not docs:
                continue
            hit = entry["topic"] in (topics or []) or any(p in low for p in entry.get("match_any", []))
            if not hit:
                continue
            msg = entry["en"].format(
                case_id=case_id, documents=join_words(docs) or "the requested documents",
                average_processing_time_after_submission=g["claim_followup_settings"][
                    "average_processing_time_after_submission"]["en"])
            out.append({"topic": entry["topic"], "text": msg})
        return out

    # ------------------------------------------------------------------ POST_PROCESS
    def _t_compose_summary(self) -> dict:
        st = self.state
        rec = self.repo.policyholder(self._party()) or {}
        first = rec.get("name", "").split(" ")[0] or "there"
        if st.consent.required and st.consent.rep_name:
            greeting_name = st.consent.rep_name.split(" ")[0]
        else:
            greeting_name = first
        discussed = []
        for t in st.topics:
            label = ASPECT_LABELS.get(t["aspect"], t["aspect"])
            discussed.append(f"{label} ({t['case_id']})")
        if not discussed:
            discussed = ["Identity verification and a review of the claims on your policy"]
        outcomes, next_steps = [], []
        for cid in st.discussed_cases:
            c = self._t_get_claim(cid)
            if not c:
                continue
            line = f"{cid} — {c['case_type']} claim filed {c['created_at_human']}: status {c['status'].upper()}."
            if c["status"] == "denied" and c.get("denial_reason"):
                line += f" Reason: {c['denial_reason']}."
            if c["status"] == "closed":
                line += f" Paid {money(c['net_pay'])} (allowed max {money(c['allowed_max_amount'])})."
            if c["status"] == "open":
                line += (f" In progress; expected reimbursement {money(c['expected_reimbursement_amount'])}, "
                         f"nothing paid yet.")
            outcomes.append(line)
            if c.get("documents_needed"):
                next_steps.append(f"Submit the {join_words(c['documents_needed'])} for {cid} through the member "
                                  f"portal or claim upload link (support can arrange fax or mail if needed).")
                ct = self.repo.guidelines["case_type_guidance"].get(c["case_type"], {}).get("en")
                next_steps.append("Make sure each file is legible and complete." + (f" {ct}" if ct else ""))
                next_steps.append("If a document is unavailable, request a replacement copy from the provider first; "
                                  "a human claims representative can review options if nothing can be obtained.")
                next_steps.append("After submission, review usually takes less than a week (another review cycle "
                                  "may be needed).")
            if c.get("appeal_deadline"):
                if c.get("appeal_deadline_passed"):
                    next_steps.append(f"The appeal deadline on file for {cid} ({c['appeal_deadline_human']}) has "
                                      f"passed — contact a human representative to review options.")
                else:
                    next_steps.append(f"Appeal deadline for {cid}: {c['appeal_deadline_human']}.")
        if not next_steps:
            next_steps = ["No action is required from you right now. Contact us anytime with questions."]
        to = rec.get("email", "")
        subject = "Summary of your claims support conversation"
        if st.discussed_cases:
            subject += " — " + ", ".join(st.discussed_cases)
        body = "\n".join([
            f"Hi {greeting_name},",
            "",
            "Thank you for contacting Claims Support. Here is a summary of today's conversation.",
            "",
            "WHAT WE DISCUSSED",
            *[f"  • {d}" for d in discussed],
            "",
            "CLAIM STATUS / OUTCOME",
            *([f"  • {o}" for o in outcomes] or ["  • No specific claim was reviewed."]),
            "",
            "FOLLOW-UP ITEMS / NEXT STEPS",
            *[f"  • {n}" for n in next_steps],
            "",
            "If anything here looks wrong, reply to this email or call us and a representative will help.",
            "",
            "— Claims Support (virtual assistant)",
        ])
        return {"to": to, "to_masked": mask_email(to) if to else "", "subject": subject, "body": body,
                "discussed": discussed, "outcomes": outcomes, "next_steps": next_steps}

    def _t_send_email_summary(self, summary: dict) -> dict:
        out_dir = Path(self.settings.data_dir) / "outbox"
        out_dir.mkdir(parents=True, exist_ok=True)
        msg_id = "EM-" + uuid.uuid4().hex[:8].upper()
        record = {"message_id": msg_id, "session_id": self.state.session_id, "sent_at": time.time(),
                  "to": summary["to"], "subject": summary["subject"], "body": summary["body"], "transport": "outbox"}
        smtp_host = __import__("os").getenv("SMTP_HOST")
        if smtp_host:  # optional real delivery
            try:
                import os
                em = EmailMessage()
                em["From"] = os.getenv("SMTP_FROM", "claims-support@example.com")
                em["To"], em["Subject"] = summary["to"], summary["subject"]
                em.set_content(summary["body"])
                with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", "587")), timeout=10) as s:
                    if os.getenv("SMTP_STARTTLS", "1") == "1":
                        s.starttls()
                    if os.getenv("SMTP_USER"):
                        s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
                    s.send_message(em)
                record["transport"] = "smtp"
            except Exception as exc:  # fall back to outbox, never crash the conversation
                record["transport"] = f"outbox (smtp failed: {exc.__class__.__name__})"
        (out_dir / f"{msg_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"message_id": msg_id, "to_masked": summary["to_masked"], "transport": record["transport"]}

    # ------------------------------------------------------------------ any phase
    def _t_transfer_to_human(self, reason: str) -> dict:
        st = self.state
        ticket = "HT-" + uuid.uuid4().hex[:6].upper()
        context = {
            "ticket": ticket,
            "reason": reason,
            "phase_at_transfer": st.phase.value,
            "verified": st.verification.verified,
            "party_id": st.verification.party_id if st.verification.verified else None,
            "active_case_id": st.active_case_id if st.verification.verified else None,
            "caller_intent": st.memory.intent.value if st.memory.intent else None,
            "case_hints": {k: v.value for k, v in st.memory.case_hints.items()},
            "is_representative": st.memory.get("caller", "is_representative", False),
        }
        out_dir = Path(self.settings.data_dir) / "handoffs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ticket}.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
        return context


# ---------------------------------------------------------------------- helpers
def claim_brief(c: dict) -> dict:
    d = date.fromisoformat(c["created_at"])
    return {"case_id": c["case_id"], "case_type": c["case_type"], "status": c["status"], "created_at": c["created_at"],
            "label": f"{c['case_id']}: {c['case_type']} claim from {MONTH_NAMES[d.month]} {d.day}, {d.year} ({c['status']})"}


def _compare(rec: dict, provided: dict[str, str]) -> tuple[list[str], list[str]]:
    matched, mismatched = [], []
    for k, v in provided.items():
        if k == "full_name":
            names = {norm_name(rec["name"])} | {norm_name(a) for a in rec.get("name_aliases", [])}
            ok = norm_name(v) in names
        elif k == "dob":
            ok = v == rec["dob"]
        elif k == "phone":
            phones = {norm_phone(rec["phone"])} | {norm_phone(a) for a in rec.get("phone_aliases", [])}
            ok = norm_phone(v) in phones
        elif k == "email":
            emails = {norm_email(rec["email"])} | {norm_email(a) for a in rec.get("email_aliases", [])}
            ok = norm_email(v) in emails
        elif k == "id_last4":
            ok = norm_last4(v) == rec["id_last4"]
        else:
            continue
        (matched if ok else mismatched).append(k)
    return matched, mismatched


def _match_doc_key(doc: str, table: dict) -> str | None:
    tokens = set(doc.lower().split())
    best = None
    for key in table:
        if key == "default":
            continue
        if tokens <= set(key.lower().split()):
            if best is None or len(key) < len(best):
                best = key
    return best


def _redact(kwargs: dict) -> dict:
    out = {}
    for k, v in kwargs.items():
        if k == "fields" and isinstance(v, dict):
            out[k] = {f: ("<redacted>" if f != "full_name" else val) for f, val in v.items() if val}
        elif k == "summary":
            out[k] = "<email body>"
        else:
            out[k] = v
    return out


def _summarize(result: Any) -> Any:
    if isinstance(result, list):
        return [r.get("case_id", r.get("topic", "?")) if isinstance(r, dict) else r for r in result]
    if isinstance(result, dict):
        keep = {k: v for k, v in result.items() if k in
                ("status", "party_id", "matched_fields", "registered", "request_id", "case_id", "message_id",
                 "ticket", "to_masked", "transport", "polls")}
        return keep or {"keys": sorted(result)[:8]}
    return result
