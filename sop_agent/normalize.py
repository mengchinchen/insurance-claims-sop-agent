"""Normalization helpers for identity fields and dates."""
from __future__ import annotations

import re
from datetime import date

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]

IDENTITY_FIELDS = ["full_name", "dob", "phone", "email", "id_last4"]
FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number on file",
    "email": "email on file",
    "id_last4": "last 4 of SSN (or national ID)",
    "policy_number": "policy number",
}


def norm_name(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"[^a-z\s'-]", " ", value.lower())
    return " ".join(value.split())


def norm_phone(value: str | None) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[-10:] if len(digits) >= 10 else digits


def norm_email(value: str | None) -> str:
    return (value or "").strip().lower()


def norm_last4(value: str | None) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[-4:] if len(digits) >= 4 else digits


def norm_policy(value: str | None) -> str:
    v = re.sub(r"[^A-Za-z0-9]", "", value or "").upper()
    m = re.match(r"POL(\d+)", v)
    return f"POL-{m.group(1)}" if m else v


def norm_case_id(value: str | None) -> str:
    v = re.sub(r"[^A-Za-z0-9]", "", value or "").upper()
    m = re.match(r"CL(\d+)", v)
    return f"CL-{m.group(1)}" if m else v


def parse_date(text: str) -> date | None:
    """Parse common spoken/typed date formats into a date."""
    text = text.strip()
    m = re.search(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
    if m:
        y, mo, d = map(int, m.groups())
        return _safe_date(y, mo, d)
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.]((?:19|20)\d{2})\b", text)
    if m:  # US style MM/DD/YYYY
        mo, d, y = map(int, m.groups())
        return _safe_date(y, mo, d) or _safe_date(y, d, mo)
    month_re = "|".join(sorted(MONTHS, key=len, reverse=True))
    m = re.search(rf"\b({month_re})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", text, re.I)
    if m:
        return _safe_date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_re})\.?,?\s+((?:19|20)\d{{2}})\b", text, re.I)
    if m:
        return _safe_date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def _safe_date(y: int, mo: int, d: int) -> date | None:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def human_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{MONTH_NAMES[d.month]} {d.day}, {d.year}"


def money(amount: str | None) -> str:
    try:
        return f"${float(amount or 0):,.2f}"
    except ValueError:
        return str(amount)


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}{'*' * max(len(local) - 1, 3)}@{domain}"


def mask_value(field: str, value: str) -> str:
    if field == "email":
        return mask_email(value)
    if field == "phone":
        return f"***-***-{value[-4:]}" if len(value) >= 4 else "***"
    if field == "id_last4":
        return "••••"
    if field == "dob":
        return "••••-••-••"
    if field == "full_name":
        return value
    return value


def join_words(items: list[str], conj: str = "and") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return f"{', '.join(items[:-1])}, {conj} {items[-1]}"
