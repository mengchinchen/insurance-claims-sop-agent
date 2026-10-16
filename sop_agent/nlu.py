"""Natural-language understanding: deterministic rules + optional LLM extraction, merged safely.

The rules give a dependable baseline (and make the demo work with no API key). When an LLM is configured it
handles messy phrasing; its output is schema-validated and identity values are only accepted if they are
traceable to the caller's actual words (anti-hallucination guard).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field

from .normalize import (MONTHS, norm_case_id, norm_email, norm_last4, norm_phone, norm_policy, parse_date)

INTENTS = ["denial_question", "status_inquiry", "document_submission", "next_steps", "payment_question",
           "general_claim_question", "list_claims", "new_claim", "policy_change"]
UNSUPPORTED_INTENTS = {"new_claim", "policy_change"}
ASPECTS = ["status", "denial_reason", "documents_needed", "document_requirements", "document_alternatives",
           "submission_method", "submission_timing", "processing_time", "receipt_confirmation", "amounts",
           "appeal_deadline"]
ACTS = ["provide_info", "ask_question", "affirm", "deny", "done", "request_human", "choose_option", "chitchat", "other"]
EMOTIONS = ["frustration", "anger", "anxiety", "confusion", "refusal"]


class NLUResult(BaseModel):
    identity: dict[str, str] = Field(default_factory=dict)
    caller: dict[str, Any] = Field(default_factory=dict)
    case_hints: dict[str, Any] = Field(default_factory=dict)
    intent: Optional[str] = None
    aspects: list[str] = Field(default_factory=list)
    act: str = "other"
    choice_index: Optional[int] = None
    declined_fields: list[str] = Field(default_factory=list)
    unavailable_fields: list[str] = Field(default_factory=list)
    scope: str = "in_scope"        # in_scope | chitchat | out_of_scope
    emotions: list[str] = Field(default_factory=list)
    is_question: bool = False
    source: str = "rules"
    llm_error: Optional[str] = None

    def has_slots(self) -> bool:
        return bool(self.identity or self.case_hints or self.caller.get("is_representative") or self.intent
                    or self.declined_fields or self.unavailable_fields)


# --------------------------------------------------------------------------- regexes
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_SPOKEN_EMAIL = re.compile(r"\b([\w.+-]+)\s+at\s+([\w-]+)\s+dot\s+(com|net|org|edu|io)\b", re.I)
RE_PHONE = re.compile(r"(?<![\d-])(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?![\d-])")
RE_POLICY = re.compile(r"\bPOL[-\s]?(\d{3,6})\b", re.I)
RE_CASE = re.compile(r"\b(?:CL|claim\s*(?:#|number|no\.?)?)\s*[-\s]?(\d{4})\b", re.I)
RE_LAST4 = re.compile(r"(?:ssn|social(?:\s+security)?(?:\s+number)?|last\s*(?:four|4)(?:\s+digits)?|national\s+id"
                      r"|id\s+number|ending\s+in)[^0-9\n]{0,30}?(\d{4})(?![\d/\-])", re.I)
MONTH_ALT = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|" \
            "apr|jun|jul|aug|sept|sep|oct|nov|dec"
RE_DATE_ANY = re.compile(
    rf"\b(?:(?:19|20)\d{{2}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.](?:19|20)\d{{2}}"
    rf"|(?:{MONTH_ALT})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTH_ALT})\.?,?\s+(?:19|20)\d{{2}})\b", re.I)
RE_DOB_CUE = re.compile(r"\b(dob|d\.o\.b|birth|born|birthday|bday)\b", re.I)

NAME_STOP = {"and", "my", "i", "policy", "dob", "calling", "the", "from", "born", "ssn", "phone", "email", "is",
             "about", "with", "on", "for", "here", "im", "i'm", "date", "birth", "social", "last", "number", "but",
             "please", "pol", "calling.", "insurance", "claim"}
WEAK_NAME_BLOCK = {"Calling", "The", "Here", "Trying", "Not", "So", "Just", "Sorry", "Really", "Very", "Still",
                   "Looking", "Worried", "Frustrated", "Confused", "Done", "Fine", "Good", "Okay", "Ok", "Happy",
                   "Policyholder", "Her", "His", "A", "An", "Your", "Glad", "Sure", "Upset", "Angry", "Afraid"}
RELATIONS = {"mother": "mother", "mom": "mother", "mum": "mother", "father": "father", "dad": "father",
             "wife": "spouse", "husband": "spouse", "spouse": "spouse", "son": "son", "daughter": "daughter",
             "parent": "parent", "sister": "sibling", "brother": "sibling", "grandmother": "grandparent",
             "grandfather": "grandparent", "aunt": "relative", "uncle": "relative", "caregiver": "caregiver",
             "client": "client", "friend": "friend"}
REL_ALT = "|".join(RELATIONS)

FIELD_WORDS = {
    "id_last4": r"ssn|social security|social|national id|last four|last 4",
    "dob": r"date of birth|dob|birthday|birth date|birthdate",
    "phone": r"phone|cell|mobile",
    "email": r"e-?mail",
}

CASE_TYPE_WORDS = {
    "healthcare": r"health\s*care|healthcare|health|medical|hospital|doctor|surgery|clinic|biopsy|pathology",
    "dental": r"dental|dentist|teeth|tooth|orthodont",
    "auto": r"\bauto\b|\bcar\b|vehicle|accident|collision|crash|body shop",
}
STATUS_WORDS = {
    "denied": r"\bdenied\b|\bdeny\b|\bdenial\b|\brejected\b|\bturned down\b",
    "open": r"\bopen\b|\bin progress\b|\bpending\b|\bstill processing\b|\bongoing\b",
    "closed": r"\bclosed\b|\bsettled\b|\bcompleted\b|\bfinished\b",
}

DOMAIN = re.compile(
    r"claim|insur|polic(y|ies)|coverage|\bcover(ed)?\b|denied|\bdeny|denial|appeal|document|\breports?\b|paperwork|"
    r"\bpa(y|id|yment)|reimburs|deductible|premium|verif|identity|\bssn\b|social security|\bdob\b|birth|e-?mail|"
    r"\bphone|\bname\b|health|dental|\bauto\b|\bcar\b|medical|hospital|doctor|\bstatus\b|submit|upload|\bfax\b|"
    r"portal|deadline|pathology|office note|estimate|\bphotos?\b|\bagent\b|human|representative|\bperson\b|consent|"
    r"summary|benefit|\bbill|money|amount|refund|\bcase\b|\bfiles?\b|\bforms?\b|\bpdf\b|\bscan|\bcl-|\bpol-|"
    r"mother|father|\bmom\b|\bdad\b|behalf|accident|repair|dentist|treatment|provider|\blab\b|clinic|diagnos|"
    r"\bhelp\b|\bverify|\bwhy do you need|approv|\bcheck\b|\byet\b", re.I)
CHITCHAT = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|thx|ok(ay)?|cool|great|got it|"
    r"sounds good|alright|yes|yeah|yep|no|nope|sure|bye|goodbye|how are you|who are you|what can you do)\b", re.I)
OOS_CUES = re.compile(r"^\s*(what|who|how|why|when|where|can you|could you|tell me|explain|write|give me|define|"
                      r"is it|are you|do you|which|recommend|translate|solve|calculate)\b|\?\s*$", re.I)

AFFIRM = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ok(ay)?|please( do)?|go ahead|send( it)?|do it|sounds good|"
                    r"correct|that'?s (right|correct|it|the one)|right|absolutely|of course|definitely|y)\b", re.I)
DENY = re.compile(r"^\s*(no|nope|nah|don'?t|do not|skip|not now|no need|no thanks|no thank you|not really|n)\b", re.I)
DONE = re.compile(r"\b(that'?s all|that is all|that'?s it|nothing else|no more questions|no other questions|"
                  r"i'?m (good|all set|done)|all set|that'?s everything|goodbye|bye|that will be all|"
                  r"no,? (thanks|thank you)|that helps,? thanks)\b", re.I)
HUMAN = re.compile(r"\b(human|real person|live (agent|person|rep)|operator|supervisor|manager|"
                   r"(talk|speak|connect)( me)? (to|with) (a |an |someone|somebody|your )?"
                   r"(agent|representative|rep|person|someone|somebody|human|supervisor|manager)|"
                   r"transfer me|escalate)\b", re.I)

EMOTION_RULES = {
    "anger": r"ridiculous|absurd|stupid|useless|terrible|furious|angry|pissed|unacceptable|outrageous|"
             r"this is crazy|insane|wtf|damn|hate this|!!+|sick of",
    "frustration": r"already told you|told you already|i just told you|again\?|how many times|waste of time|"
                   r"frustrat|annoy|just tell me|come on|seriously|for the last time|why is this so|"
                   r"taking (so|too) long|keep asking|stop asking|enough with",
    "anxiety": r"worried|anxious|scared|nervous|stress|afraid|can'?t afford|panic|urgent|desperate|"
               r"freaking out|really need|what if|losing sleep",
    "confusion": r"confus|don'?t understand|do not understand|what do you mean|not sure what|why do you need|"
                 r"why do you want|why does it matter|huh\b|\?\?|makes no sense|lost here",
    "refusal": r"\bi won'?t\b|\bwill not\b|not going to (give|share|tell|do)|refuse|none of your business|"
               r"not giving|rather not|no way|not comfortable|not sharing|i'?m not telling",
}

ORDINALS = {"first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1, "third": 2, "3rd": 2, "three": 2,
            "fourth": 3, "4th": 3, "four": 3, "last": -1}


# --------------------------------------------------------------------------- rule extraction
def _clean_name(raw: str) -> str | None:
    words = []
    for w in re.split(r"\s+", raw.strip()):
        w = w.strip(",.;:!?\"'()")
        if not w or w.lower() in NAME_STOP or not re.match(r"^[A-Za-z][A-Za-z'\-]*$", w):
            break
        words.append(w[0].upper() + w[1:])
        if len(words) == 4:
            break
    return " ".join(words) if len(words) >= 2 else None


def _strong_name(text: str, cue: str) -> str | None:
    m = re.search(cue + r"\s+(.{2,60})", text, re.I)
    return _clean_name(re.split(r"[,.;!?\n]| and ", m.group(1))[0]) if m else None


def _weak_name(text: str) -> str | None:
    for m in re.finditer(r"\b(?:I'm|I am|[Tt]his is|[Ii]t's|[Ii]t is)\s+([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,3})",
                         text):
        first = m.group(1).split()[0]
        if first not in WEAK_NAME_BLOCK:
            return _clean_name(m.group(1))
    return None


def _extract_names(text: str, ctx: dict) -> tuple[dict, dict]:
    identity, caller = {}, {}
    low = text.lower()
    self_policyholder = bool(re.search(r"\b(i'?m|i am) the (policy ?holder|insured|member)\b", low))
    rep = None
    if not self_policyholder:
        m = (re.search(rf"\bon behalf of (?:my )?({REL_ALT})?", low)
             or re.search(rf"\b(?:calling|call|here) (?:for|about) my ({REL_ALT})\b", low)
             or re.search(rf"\b(?:i'?m|i am) (?:her|his|their) ({REL_ALT}|representative|authorized representative)\b",
                          low)
             or re.search(rf"\bmy ({REL_ALT})'?s (?:claim|policy|insurance)", low)
             or re.search(rf"\bfor my ({REL_ALT})\b", low))
        if m:
            rep = m.group(1) or "representative"
    if rep:
        caller["is_representative"] = True
        caller["relationship"] = RELATIONS.get(rep, rep)
        ph = (_strong_name(text, rf"\b(?:my (?:{REL_ALT}),?|on behalf of(?: my (?:{REL_ALT}))?,?)")
              or _strong_name(text, r"\b(?:her|his|their) (?:full )?name is")
              or _strong_name(text, r"\bpolicy ?holder(?:'s name)? is"))
        if ph:
            identity["full_name"] = ph
        own = _strong_name(text, r"\b(?:my name is|my name's)") or _weak_name(text)
        if own and own != ph:
            caller["caller_name"] = own
    else:
        if self_policyholder:
            caller["is_representative"] = False
        name = (_strong_name(text, r"\b(?:my (?:full )?name is|my name's|full name is|name is|name:|name)")
                or _weak_name(text))
        if not name:
            m = re.match(r"^\s*(?:hi|hello|hey)?[,!.]?\s*([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,3})\s*(?:,|\bhere\b|$)",
                         text)
            if m and m.group(1).split()[0] not in WEAK_NAME_BLOCK:
                name = _clean_name(m.group(1))
        exp = ctx.get("expecting") or []
        if not name and (not exp or "full_name" in exp or "caller_name" in exp):
            stripped = text.strip().strip(".!")
            if re.fullmatch(r"[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){1,3}", stripped) and \
                    not DOMAIN.search(stripped) and not CHITCHAT.match(stripped) and \
                    not any(w.lower() in NAME_STOP or w.lower() in MONTHS for w in stripped.split()):
                name = _clean_name(stripped)
        if name:
            if "caller_name" in exp:
                caller["caller_name"] = name
            else:
                identity["full_name"] = name
    return identity, caller


def _dates(text: str, today: date) -> tuple[str | None, dict]:
    dob, hints = None, {}
    for m in RE_DATE_ANY.finditer(text):
        d = parse_date(m.group(0))
        if not d:
            continue
        before = text[max(0, m.start() - 30):m.start()]
        if RE_DOB_CUE.search(before) or d.year <= today.year - 16:
            dob = dob or d.isoformat()
        else:
            hints.update({"month": d.month, "year": d.year})
    # Partial claim-date hints like "from January", "back in March", "January 12", "in 2025".
    low = text.lower()
    m = re.search(rf"\b(?:in|from|on|since|during|filed in|back in|around|early|late|mid)\s+(?:early |late |mid-)?"
                  rf"({MONTH_ALT})\b", low) or re.search(rf"\b({MONTH_ALT})\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!,?\s*\d{{4}})",
                                                         low) \
        or re.search(r"\b(january|february|april|june|july|august|september|october|november|december)\b", low)
    if m and "month" not in hints:
        window = low[max(0, m.start() - 30):m.start()]
        if not RE_DOB_CUE.search(window):
            hints["month"] = MONTHS[m.group(1)]
    if "year" not in hints:
        for y in re.finditer(r"(?<![-\d])\b(20\d{2})\b(?![-/]\d)", low):
            yr = int(y.group(1))
            if 2000 <= yr <= today.year and not RE_DOB_CUE.search(low[max(0, y.start() - 30):y.start()]):
                hints["year"] = yr
                break
    return dob, hints


def _fields_near(text: str, pattern: str) -> list[str]:
    out = []
    for m in re.finditer(pattern, text, re.I):
        window = re.split(r"[.!?;]", text[m.end(): m.end() + 45].lower())[0]
        for f, words in FIELD_WORDS.items():
            if re.search(words, window) and f not in out:
                out.append(f)
    return out


def rule_extract(text: str, ctx: dict) -> NLUResult:
    today: date = ctx.get("today") or date.today()
    expecting = ctx.get("expecting") or []
    r = NLUResult()
    low = text.lower().strip()

    # ---- identity
    ident, caller = _extract_names(text, ctx)
    r.identity.update(ident)
    r.caller.update(caller)
    m = RE_EMAIL.search(text)
    if m:
        r.identity["email"] = norm_email(m.group(0))
    else:
        m = RE_SPOKEN_EMAIL.search(text)
        if m:
            r.identity["email"] = f"{m.group(1)}@{m.group(2)}.{m.group(3)}".lower()
    for m in RE_PHONE.finditer(text):
        if len(norm_phone(m.group(0))) == 10:
            r.identity["phone"] = norm_phone(m.group(0))
            break
    m = RE_LAST4.search(text)
    if m:
        r.identity["id_last4"] = m.group(1)
    elif "id_last4" in expecting and re.fullmatch(r"\D{0,20}\d{4}\D{0,5}", low):
        r.identity["id_last4"] = norm_last4(low)
    m = RE_POLICY.search(text)
    if m:
        r.identity["policy_number"] = norm_policy(m.group(0))
    dob, date_hints = _dates(text, today)
    if dob:
        r.identity["dob"] = dob

    # ---- case hints (remembered in any phase)
    r.case_hints.update(date_hints)
    m = RE_CASE.search(text)
    if m:
        r.case_hints["case_id"] = norm_case_id("CL" + m.group(1))
    for ct, pat in CASE_TYPE_WORDS.items():
        if re.search(pat, low):
            r.case_hints["case_type"] = ct
            break
    for st, pat in STATUS_WORDS.items():
        if re.search(pat, low):
            r.case_hints["status"] = st
            break

    # ---- refusals / unavailable identity fields
    r.declined_fields = _fields_near(text, r"(?:don'?t|do not|won'?t|will not|not going to|rather not|refuse to|"
                                           r"not comfortable|(?:i'?m|i am) not|never)\s+(?:want to\s+|going to\s+)?"
                                           r"(?:give|giving|share|sharing|provide|providing|tell|telling|say)")
    r.unavailable_fields = _fields_near(text, r"(?:don'?t|do not|can'?t|cannot|no longer)\s+(?:have|know|remember|"
                                              r"recall|find|use)")
    if re.search(r"\bno (e-?mail|email address)\b", low):
        r.unavailable_fields.append("email")

    # ---- intent + aspects
    r.is_question = "?" in text or bool(re.match(r"^\s*(what|why|how|when|where|who|which|can|could|do|does|is|are|"
                                                 r"will|should)\b", low))
    r.intent = _intent(low, r)
    r.aspects = _aspects(low)

    # ---- dialogue act
    if HUMAN.search(text) and not re.search(r"\b(?:i'?m|i am) (?:her|his|their|an?|the) (?:authorized )?representative",
                                            low):
        r.act = "request_human"
    elif DONE.search(low):
        r.act = "done"
    elif AFFIRM.match(low) and len(low.split()) <= 8:
        r.act = "affirm"
    elif DENY.match(low) and len(low.split()) <= 8 and not r.declined_fields and not r.unavailable_fields:
        r.act = "deny"
    elif r.identity or r.caller:
        r.act = "provide_info"
    elif r.is_question:
        r.act = "ask_question"
    m = re.search(r"\b(?:the\s+)?(first|second|third|fourth|1st|2nd|3rd|4th)\b(?:\s+one)?|\b(?:the\s+)?(last)\s+one\b|"
                  r"\b(?:number|option|#)\s*(\d)\b", low)
    if m and m.group(1) in ("one",):
        m = None
    if m:
        r.choice_index = ORDINALS[m.group(1) or m.group(2)] if (m.group(1) or m.group(2)) else int(m.group(3)) - 1
        if r.act in ("other", "ask_question", "affirm"):
            r.act = "choose_option"

    # ---- emotions
    for emo, pat in EMOTION_RULES.items():
        if re.search(pat, text, re.I):
            r.emotions.append(emo)
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.7 and "anger" not in r.emotions:
        r.emotions.append("anger")
    if r.declined_fields and "refusal" not in r.emotions:
        r.emotions.append("refusal")

    # ---- scope
    r.scope = _scope(text, r)
    return r


def _intent(low: str, r: NLUResult) -> str | None:
    if re.search(r"\b(file|open|start|submit|report) (a )?new claim\b|\bnew claim\b", low):
        return "new_claim"
    if re.search(r"\b(change|update) (my )?(address|beneficiary|policy|plan|coverage)\b|\bcancel (my )?policy\b|"
                 r"\bpremium\b|\bbilling\b", low):
        return "policy_change"
    if re.search(r"why .*(denied|rejected|declined)|reason .*(denied|denial)|denial reason|\bwhy\b.*\bdeny", low):
        return "denial_question"
    if re.search(r"\b(submit|upload|send|fax|mail)\b.*\b(document|report|note|file|paper|record|estimate|photo)s?|"
                 r"\b(document|paperwork|pathology|office note|records?)\b", low):
        return "document_submission"
    if re.search(r"next step|what (should|do|can) i do|what now|how (do|can) i (fix|appeal|get it)|\bappeal\b", low):
        return "next_steps"
    if re.search(r"how much|amount|\bpaid\b|payment|reimburs|\bpay\b|\bowe\b|check|money", low):
        return "payment_question"
    if re.search(r"\bstatus\b|update on|where is|progress|check on|what'?s happening|any news", low):
        return "status_inquiry"
    if re.search(r"\b(all|list|show|which|what) (of )?my claims\b|\bclaims (do i|on my)", low):
        return "list_claims"
    if r.case_hints.get("status") == "denied":
        return "denial_question"
    if re.search(r"\bclaim\b", low) and (r.is_question or re.search(r"calling about|question about|help with", low)):
        return "general_claim_question"
    return None


def _aspects(low: str) -> list[str]:
    a = []
    rules = [
        ("denial_reason", r"why|reason|denied|denial|rejected"),
        ("document_alternatives", r"don'?t have|do not have|can'?t (get|find)|cannot (get|find)|lost|missing|"
                                  r"unavailable|no longer have|what if i (can'?t|don'?t)|instead|alternative|substitute"),
        ("document_requirements", r"format|file type|photo|scan|pdf|image|clear enough|what should .* include|"
                                  r"requirement|what do (they|it) need to (show|include)"),
        ("submission_method", r"how (do|can|should) i (submit|send|upload)|where (do|should|can) i (submit|send|upload)|"
                              r"portal|upload link|fax|mail it"),
        ("submission_timing", r"(how soon|when) (do|should) i (need to )?(submit|send)|by when"),
        ("processing_time", r"how long|processing time|review time|once i submit|after i (submit|send)|"
                            r"how long does it take|how soon will"),
        ("receipt_confirmation", r"know you got it|confirm(ation)? (of )?receipt|will i get (a )?confirmation|"
                                 r"show up in (the )?status"),
        ("amounts", r"how much|amount|paid|payment|reimburs|allowed|net pay|\bpay\b|money|\$"),
        ("appeal_deadline", r"deadline|how long do i have|last day|appeal by|time limit|too late"),
        ("documents_needed", r"what (documents|do you need|do i need|is missing)|which documents|need to send|"
                             r"documents needed|what.*missing"),
        ("status", r"\bstatus\b|where is|update|progress"),
    ]
    for name, pat in rules:
        if re.search(pat, low):
            a.append(name)
    return a


def _scope(text: str, r: NLUResult) -> str:
    if r.identity or r.caller or r.case_hints or r.declined_fields or r.unavailable_fields:
        return "in_scope"
    if r.act in ("request_human", "done", "affirm", "deny", "choose_option"):
        return "in_scope"
    if r.intent:
        return "in_scope"
    if CHITCHAT.match(text):
        return "chitchat"
    if r.emotions and not OOS_CUES.search(text):
        return "in_scope"
    if DOMAIN.search(text):
        return "in_scope"
    if OOS_CUES.search(text):
        return "out_of_scope"     # a question/request with nothing to do with insurance support
    return "chitchat"             # unclear statement: not penalised, the agent just re-prompts


# --------------------------------------------------------------------------- LLM extraction + merge
NLU_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "identity": {"type": "object", "description": "Identity details THE CALLER STATED in this message for the "
                     "policyholder. Omit anything not said. Never guess.",
                     "properties": {"full_name": {"type": "string"},
                                    "dob": {"type": "string", "description": "YYYY-MM-DD"},
                                    "phone": {"type": "string"}, "email": {"type": "string"},
                                    "id_last4": {"type": "string", "description": "last 4 digits of SSN or national ID"},
                                    "policy_number": {"type": "string"}}},
        "caller": {"type": "object", "properties": {
            "is_representative": {"type": "boolean", "description": "true only if calling on behalf of someone else"},
            "caller_name": {"type": "string", "description": "the representative's own name, if a representative"},
            "relationship": {"type": "string"}}},
        "case_hints": {"type": "object", "properties": {
            "case_id": {"type": "string"},
            "case_type": {"type": "string", "enum": ["healthcare", "dental", "auto"]},
            "status": {"type": "string", "enum": ["denied", "open", "closed"]},
            "month": {"type": "integer", "minimum": 1, "maximum": 12},
            "year": {"type": "integer"}}},
        "intent": {"type": ["string", "null"], "enum": INTENTS + [None]},
        "aspects": {"type": "array", "items": {"type": "string", "enum": ASPECTS},
                    "description": "Which facts about the claim the caller is asking for in this message"},
        "act": {"type": "string", "enum": ACTS},
        "choice_index": {"type": ["integer", "null"], "description": "0-based index if picking from offered options"},
        "declined_fields": {"type": "array", "items": {"type": "string",
                                                       "enum": ["full_name", "dob", "phone", "email", "id_last4"]}},
        "unavailable_fields": {"type": "array", "items": {"type": "string",
                                                          "enum": ["full_name", "dob", "phone", "email", "id_last4"]}},
        "scope": {"type": "string", "enum": ["in_scope", "chitchat", "out_of_scope"],
                  "description": "out_of_scope = unrelated to this insurance/claims support call (e.g. general "
                                 "knowledge, coding, trivia). Identity answers, emotions, claim questions are in_scope."},
        "emotions": {"type": "array", "items": {"type": "string", "enum": EMOTIONS}},
    },
    "required": ["act", "scope"],
}

NLU_SYSTEM = """You are the language-understanding module of an insurance claims support phone agent.
Extract structured information from the caller's latest message. You do NOT reply to the caller.
Rules:
- Only extract values the caller actually stated in THIS message. Never invent or complete values.
- Dates of birth -> YYYY-MM-DD. Claim dates go to case_hints.month/year, never to dob.
- If the caller speaks for someone else (e.g. their mother), identity fields describe the POLICYHOLDER, and the caller's
  own name goes to caller.caller_name.
- Remember hints even if they belong to a later step (e.g. "my denied healthcare claim from January").
- act=affirm/deny when answering a yes/no question from the agent; act=done when they have no more questions;
  act=request_human when they ask for a person/agent/representative to talk to.
- Emotions: tag frustration, anger, anxiety, confusion, or refusal only when clearly expressed."""


def merge_llm(rule: NLUResult, data: dict, text: str) -> NLUResult:
    out = rule.model_copy(deep=True)
    out.source = "rules+llm"
    low = text.lower()
    digits = re.sub(r"\D", "", text)
    for k, v in (data.get("identity") or {}).items():
        if not v or k in out.identity:
            continue  # rules win for exact formats
        v = str(v).strip()
        if k == "full_name" and all(w.lower() in low for w in v.split()):
            out.identity[k] = v.title() if v.islower() else v
        elif k == "dob" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) and v[:4] in text:
            out.identity[k] = v
        elif k == "phone" and len(norm_phone(v)) == 10 and norm_phone(v) in digits:
            out.identity[k] = norm_phone(v)
        elif k == "email" and "@" in v and v.split("@")[0].lower() in low:
            out.identity[k] = norm_email(v)
        elif k == "id_last4" and len(norm_last4(v)) == 4 and norm_last4(v) in digits:
            out.identity[k] = norm_last4(v)
        elif k == "policy_number" and re.sub(r"\D", "", v) and re.sub(r"\D", "", v) in digits:
            out.identity[k] = norm_policy(v)
    caller = data.get("caller") or {}
    if caller.get("is_representative") is not None and "is_representative" not in out.caller:
        out.caller["is_representative"] = bool(caller["is_representative"])
    for k in ("caller_name", "relationship"):
        if caller.get(k) and k not in out.caller and (k != "caller_name" or
                                                      all(w.lower() in low for w in str(caller[k]).split())):
            out.caller[k] = caller[k]
    for k, v in (data.get("case_hints") or {}).items():
        if v in (None, "") or k in out.case_hints:
            continue
        if k == "case_id":
            v = norm_case_id(str(v))
            if re.sub(r"\D", "", v) not in digits:
                continue
        out.case_hints[k] = v
    if data.get("intent") in INTENTS:
        out.intent = data["intent"]
    if data.get("aspects"):
        out.aspects = list(dict.fromkeys([a for a in data["aspects"] if a in ASPECTS] + out.aspects))
    if data.get("act") in ACTS:
        # LLM understands nuance better, but keep a rule-detected explicit human request.
        if not (out.act == "request_human" and data["act"] != "request_human"):
            out.act = data["act"]
    if isinstance(data.get("choice_index"), int):
        out.choice_index = data["choice_index"]
    for k in ("declined_fields", "unavailable_fields", "emotions"):
        merged = list(dict.fromkeys(getattr(out, k) + [x for x in (data.get(k) or []) if isinstance(x, str)]))
        setattr(out, k, merged)
    if data.get("scope") in ("in_scope", "chitchat", "out_of_scope"):
        # If we extracted anything useful, the message is in scope regardless of the model's label.
        out.scope = "in_scope" if out.has_slots() else data["scope"]
    return out


def understand(text: str, ctx: dict, llm=None) -> NLUResult:
    result = rule_extract(text, ctx)
    if llm is None or not llm.enabled:
        return result
    try:
        context_lines = [f"Current SOP phase: {ctx.get('phase')}",
                         f"Agent is waiting for: {ctx.get('pending_prompt') or ctx.get('expecting') or 'nothing specific'}",
                         f"Agent's last message: {ctx.get('last_agent', '')[:600]}"]
        if ctx.get("options"):
            context_lines.append("Options the agent offered (0-based): " + " | ".join(ctx["options"]))
        prompt = "\n".join(context_lines) + f"\n\nCaller's latest message:\n\"\"\"{text}\"\"\""
        data = llm.extract(NLU_SYSTEM, prompt, NLU_TOOL_SCHEMA)
        return merge_llm(result, data, text)
    except Exception as exc:  # never let the model break the SOP
        result.llm_error = f"{exc.__class__.__name__}: {str(exc)[:160]}"
        return result
