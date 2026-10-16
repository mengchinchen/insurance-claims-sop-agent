"""Read-only repository over the insurance fixtures."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path


class Repository:
    def __init__(self, fixtures_dir: Path):
        self.dir = Path(fixtures_dir)

    def _load(self, name: str):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    @cached_property
    def policyholders(self) -> list[dict]:
        return self._load("policyholders.json")

    @cached_property
    def claims(self) -> list[dict]:
        return self._load("claims.json")

    @cached_property
    def representatives(self) -> list[dict]:
        return self._load("representatives.json")

    @cached_property
    def consent_scenarios(self) -> dict:
        return self._load("consent_scenarios.json")

    @cached_property
    def guidelines(self) -> dict:
        return self._load("required_document_guideline.json")

    @cached_property
    def claim_schema(self) -> dict:
        return self._load("claim_schema.json")

    def policyholder(self, party_id: str) -> dict | None:
        return next((p for p in self.policyholders if p["party_id"] == party_id), None)

    def claims_for(self, party_id: str) -> list[dict]:
        return [c for c in self.claims if c["party_id"] == party_id]

    # Values that must never appear in an agent reply before the caller is verified.
    @cached_property
    def sensitive_claim_markers(self) -> list[str]:
        markers: set[str] = set()
        for c in self.claims:
            markers.add(c["case_id"].lower())
            for k in ("denial_reason", "summary"):
                if c.get(k):
                    markers.add(c[k].lower())
            for doc in c.get("documents_needed", []):
                markers.add(doc.lower())
            for k in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee"):
                v = c.get(k)
                if v and float(v) > 0:
                    markers.add(f"{float(v):,.2f}")
                    markers.add(v)
            if c.get("appeal_deadline"):
                markers.add(c["appeal_deadline"])
        return sorted(markers)

    @cached_property
    def sensitive_pii_markers(self) -> list[str]:
        out = []
        for p in self.policyholders:
            out += [p["phone"][-10:], p["email"].lower(), p["dob"]]
            out += [e.lower() for e in p.get("email_aliases", [])]
        return out
