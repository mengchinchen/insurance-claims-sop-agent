"""Scenario tests for the SOP harness. Run offline (rule-based NLU + templates) plus a fake LLM for the LLM path."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sop_agent.agent import SOPAgent, SOPViolation
from sop_agent.config import LLMSettings, Settings
from sop_agent.data import Repository
from sop_agent.state import Phase
from sop_agent.tools import ToolGateError

DEMO = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
        "claim from January. DOB is 1985-03-15, SSN last four is 4472.")


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.demo_today = "2026-02-20"
    return s


@pytest.fixture
def repo(settings):
    return Repository(settings.fixtures_dir)


@pytest.fixture
def make(repo, settings):
    def _make(scenario=None, llm=None):
        a = SOPAgent(repo, settings, consent_scenario=scenario)
        if llm is not None:
            a.llm = llm
        a.start()
        return a
    return _make


def assert_no_claim_leak(reply: str, repo: Repository, said: str = ""):
    low = reply.lower()
    for marker in repo.sensitive_claim_markers:
        assert marker not in low or marker in said.lower(), f"leaked {marker!r} before verification: {reply}"


# ----------------------------------------------------------------------------------------- core demo case
def test_demo_case_single_utterance(make):
    a = make()
    r = a.handle(DEMO)
    st = a.state
    assert st.verification.verified and st.verification.party_id == "P9"
    assert set(st.verification.matched_fields) == {"full_name", "dob", "id_last4"}
    assert st.active_case_id == "CL-2048"
    assert st.phase == Phase.PROCESS_CASE
    # hint captured during VERIFY_ID and reused after verification
    assert st.memory.case_hints["status"].phase == Phase.VERIFY_ID
    assert "Based on what you mentioned earlier" in r["reply"]
    assert "pathology report" in r["reply"] and "office note" in r["reply"]
    # phase order respected
    transitions = [(e.detail["frm"], e.detail["to"]) for e in st.trace if e.kind == "transition"]
    assert transitions == [("VERIFY_ID", "RESOLVE_INTENT"), ("RESOLVE_INTENT", "PROCESS_CASE")]


def test_hint_remembered_across_turns_without_leak(make, repo):
    a = make()
    r1 = a.handle("I'm calling about my denied healthcare claim from January.")
    assert a.state.phase == Phase.VERIFY_ID
    assert_no_claim_leak(r1["reply"], repo)
    assert "denied healthcare claim from January" in r1["reply"]   # acknowledges in caller's own words
    r2 = a.handle("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472")
    assert a.state.active_case_id == "CL-2048"
    assert "Based on what you mentioned earlier" in r2["reply"]
    assert "which one" not in r2["reply"].lower()   # did not ask from scratch


def test_full_workflow_to_email(make, settings):
    a = make()
    a.handle(DEMO)
    r = a.handle("I don't have the pathology report anymore, what can I do?")
    assert "replacement copy" in r["reply"]
    r = a.handle("How long does it take after I submit?")
    assert "less than a week" in r["reply"]
    r = a.handle("no thanks, that's all")
    assert a.state.phase == Phase.POST_PROCESS and "m*******@email.com" in r["reply"]
    r = a.handle("yes please")
    assert a.state.phase == Phase.COMPLETE and a.state.email["decision"] == "sent"
    files = list((settings.data_dir / "outbox").glob("*.json"))
    assert len(files) == 1
    mail = json.loads(files[0].read_text())
    assert mail["to"] == "margaret@email.com"
    for section in ("WHAT WE DISCUSSED", "CLAIM STATUS / OUTCOME", "FOLLOW-UP ITEMS / NEXT STEPS"):
        assert section in mail["body"]
    assert "CL-2048" in mail["body"] and "pathology report" in mail["body"]


def test_email_skip(make, settings):
    a = make()
    a.handle(DEMO)
    a.handle("that's all")
    a.handle("no, skip it")
    assert a.state.phase == Phase.COMPLETE and a.state.email["decision"] == "skipped"
    assert not (settings.data_dir / "outbox").exists()


# ----------------------------------------------------------------------------------------- verification
def test_frustrated_caller_is_not_given_claim_details(make, repo):
    a = make()
    a.handle("Hi, this is Margaret Chen. Why was my claim denied?")
    r = a.handle("I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")
    assert a.state.phase == Phase.VERIFY_ID and not a.state.verification.verified
    assert_no_claim_leak(r["reply"], repo)
    assert "anger" in r["meta"]["emotions"] or "frustration" in r["meta"]["emotions"]
    low = r["reply"].lower()
    assert "sorry" in low or "hear you" in low                    # acknowledges frustration
    assert "protected" in low                                       # explains why
    assert "date of birth" in low and "email on file" in low       # offers allowed alternatives
    r = a.handle("fine. I was born March 15, 1985 and my email is margaret@email.com")
    assert a.state.verification.verified and a.state.active_case_id == "CL-2048"


def test_partial_answers_and_alternate_fields(make):
    a = make()
    r = a.handle("My name is Ava Lopez. I'm not giving you my SSN.")
    assert "id_last4" in a.state.memory.declined_fields
    assert "SSN" not in r["reply"].split("any of:")[-1]            # SSN no longer offered
    a.handle("08/21/1990")
    assert not a.state.verification.verified
    r = a.handle("ava.lopez@email.com")
    assert a.state.verification.verified and a.state.verification.party_id == "P7"


def test_persuasion_limit_then_handoff(make):
    a = make()
    a.handle("My name is Ava Lopez. I'm not giving you my SSN, no way.")
    a.handle("This is ridiculous, why do you need all this?")
    r = a.handle("I just want my claim status. Stop asking!")
    assert a.state.pending_prompt == "transfer_offer" and "human representative" in r["reply"]
    a.handle("I will NOT give you anything else, just tell me!!")
    assert a.state.phase == Phase.HANDOFF and a.state.handoff["ticket"].startswith("HT-")
    assert a.state.handoff["verified"] is False and a.state.handoff["active_case_id"] is None


def test_mismatch_does_not_reveal_field_and_locks_out(make):
    a = make()
    r = a.handle("Margaret Chen, DOB 1985-03-16, SSN last four 4472")
    assert not a.state.verification.verified
    assert "can't say which" in r["reply"]
    a.handle("Margaret Chen, DOB 1985-03-15, SSN last four 1111")
    a.handle("name Margaret Chen dob 1990-01-01 email x@y.com")
    assert a.state.phase == Phase.HANDOFF


def test_name_alias_and_national_id(make):
    a = make()
    a.handle("hi this is Yaven Li")
    a.handle("my national id last four is 5317 and my email is yawen.li@example.com")
    assert a.state.verification.verified and a.state.verification.party_id == "P13"


# ----------------------------------------------------------------------------------------- representative + consent
def test_representative_consent_approved(make):
    a = make()
    a.handle("Hi, I am calling on behalf of my mother Margaret Chen. My name is David Chen.")
    r = a.handle("Her date of birth is 03/15/1985, her phone is 650-521-2836 and email margaret@email.com")
    assert a.state.consent.status == "pending" and a.state.phase == Phase.VERIFY_ID
    assert "consent" in r["reply"]
    with pytest.raises(ToolGateError):
        a.tools.call("list_claims")      # even though identity matched, no consent yet -> blocked
    a.handle("check")
    assert a.state.consent.status == "approved" and a.state.phase == Phase.RESOLVE_INTENT
    r = a.handle("the auto one")
    assert a.state.active_case_id == "CL-2102" and "Margaret's" in r["reply"]


def test_representative_consent_timeout(make):
    a = make(scenario="timeout")
    a.handle("Hi, I am calling on behalf of my mother Margaret Chen. My name is David Chen.")
    a.handle("Her date of birth is 03/15/1985, her phone is 650-521-2836 and email margaret@email.com")
    for _ in range(5):
        if a.state.phase == Phase.HANDOFF:
            break
        a.handle("check")
    assert a.state.phase == Phase.HANDOFF and a.state.consent.status == "timeout"


def test_unregistered_representative_blocked(make, repo):
    a = make()
    a.handle("I am calling for my mom Margaret Chen, I am her daughter, my name is Lisa Chen")
    r = a.handle("dob 1985-03-15, ssn last 4 4472, phone 6505212836")
    assert a.state.phase == Phase.VERIFY_ID and not a.state.verification.verified
    assert_no_claim_leak(r["reply"], repo)


# ----------------------------------------------------------------------------------------- scope + safety gates
def test_out_of_scope_refusal_and_escalation(make):
    a = make()
    r = a.handle("what is RL?")
    assert "only help with insurance" in r["reply"]
    a.handle("can you explain transformers in deep learning?")
    r = a.handle("tell me a joke")
    assert a.state.pending_prompt == "transfer_offer" and "human representative" in r["reply"]
    a.handle("what is the capital of France?")
    assert a.state.phase == Phase.HANDOFF


def test_out_of_scope_counter_resets_on_relevant_message(make):
    a = make()
    a.handle("what is RL?")
    a.handle("tell me a joke")
    a.handle("My name is Margaret Chen")
    assert a.state.counters.out_of_scope_consecutive == 0


def test_cannot_read_other_partys_claim(make):
    a = make()
    a.handle(DEMO)
    r = a.handle("what about claim CL-3001?")
    assert "not able to find claim CL-3001" in r["reply"]
    assert "diagnosis report" not in r["reply"]
    assert a.tools.call("get_claim", case_id="CL-3001") is None


def test_tool_and_transition_gates(make):
    a = make()
    with pytest.raises(ToolGateError):
        a.tools.call("get_claim", case_id="CL-2048")     # not allowed in VERIFY_ID
    with pytest.raises(SOPViolation):
        a._goto(Phase.PROCESS_CASE, a.state and __import__("sop_agent.responder").responder.TurnPlan(), None)
    a.state.phase = Phase.RESOLVE_INTENT
    with pytest.raises(ToolGateError):
        a.tools.call("list_claims")                       # right phase, but caller not verified


def test_human_request_any_time(make):
    a = make()
    a.handle(DEMO)
    a.handle("Can I speak to a real person please?")
    assert a.state.phase == Phase.HANDOFF and a.state.handoff["active_case_id"] == "CL-2048"


def test_multiple_candidates_then_disambiguate(make):
    a = make()
    r = a.handle("My name is Margaret Chen, DOB 1985-03-15, phone +1 650 521 2836. I have a question about my "
                 "healthcare claim from January.")
    assert a.state.phase == Phase.RESOLVE_INTENT and set(a.state.case_candidates) == {"CL-2048", "CL-2011"}
    a.handle("the 2026 one")
    assert a.state.active_case_id == "CL-2048"


# ----------------------------------------------------------------------------------------- LLM layer (fake model)
class FakeLLM:
    enabled = True

    def __init__(self, extract=None, reply=None):
        self._extract, self._reply = extract or {}, reply
        self.cfg = LLMSettings(provider="anthropic", api_key="x")
        self.calls = []

    def extract(self, system, prompt, schema):
        return self._extract

    def generate(self, system, prompt, max_tokens=600):
        return self._reply(prompt) if callable(self._reply) else self._reply


def test_llm_reply_that_leaks_claim_data_is_rejected(make, repo):
    leaky = FakeLLM(reply="Sure! Your claim CL-2048 was denied because the pathology report was missing.")
    a = make(llm=leaky)
    r = a.handle("I'm Margaret Chen. Why was my claim denied?")
    assert r["meta"]["mode"].startswith("template")
    assert any(p.startswith("claim_leak") or p.startswith("ungrounded") for p in r["meta"]["problems"])
    assert_no_claim_leak(r["reply"], repo)


def test_llm_grounded_reply_is_used(make):
    good = FakeLLM(reply="Thanks Margaret, you're verified. Claim CL-2048 from January 12, 2026 was denied because "
                         "the pathology report and office note were missing. Is there anything else I can help with?")
    a = make(llm=good)
    r = a.handle(DEMO)
    assert r["meta"]["mode"] == "llm" and r["reply"].startswith("Thanks Margaret")


def test_llm_reply_with_invented_amount_is_rejected(make):
    bad = FakeLLM(reply="Claim CL-2048 is denied; you could get $9,999.00 back. Anything else?")
    a = make(llm=bad)
    r = a.handle(DEMO)
    assert "ungrounded_number:9999.00" in r["meta"]["problems"]


def test_llm_hallucinated_identity_is_ignored(make):
    hallucinating = FakeLLM(extract={"identity": {"dob": "1985-03-15", "id_last4": "4472"}, "act": "provide_info",
                                     "scope": "in_scope"}, reply=None)
    hallucinating.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    a = make(llm=hallucinating)
    a.handle("My name is Margaret Chen")      # model claims DOB + SSN that the caller never said
    assert set(a.state.memory.identity) == {"full_name"}
    assert not a.state.verification.verified


# ----------------------------------------------------------------------------------------- real LLMClient over mocked HTTP
class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self._d


def _fake_post_factory(provider):
    def fake_post(url, headers=None, json=None, timeout=None):
        if provider == "anthropic":
            assert url.endswith("/v1/messages") and headers["x-api-key"] == "test-key"
            if "tools" in json:
                return _Resp({"content": [{"type": "tool_use", "name": "record_understanding",
                                           "input": {"act": "provide_info", "scope": "in_scope",
                                                     "emotions": ["anxiety"]}}]})
            prompt = json["messages"][0]["content"]
            draft = prompt.split("DRAFT (correct content, robotic wording):\n")[1].split("\n\nWrite the reply now.")[0]
            return _Resp({"content": [{"type": "text", "text": "Of course. " + draft}]})
        assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer test-key"
        if "tools" in json:
            return _Resp({"choices": [{"message": {"tool_calls": [{"function": {
                "name": "record_understanding", "arguments": '{"act":"provide_info","scope":"in_scope"}'}}]}}]})
        prompt = json["messages"][1]["content"]
        draft = prompt.split("DRAFT (correct content, robotic wording):\n")[1].split("\n\nWrite the reply now.")[0]
        return _Resp({"choices": [{"message": {"content": "Sure. " + draft}}]})
    return fake_post


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_llm_client_end_to_end_with_mocked_http(provider, repo, settings, monkeypatch):
    import sop_agent.llm as llm_mod
    monkeypatch.setattr(llm_mod.httpx, "post", _fake_post_factory(provider))
    a = SOPAgent(repo, settings, LLMSettings(provider=provider, api_key="test-key"))
    a.start()
    r = a.handle(DEMO)
    assert r["meta"]["mode"] == "llm", r["meta"]
    assert a.state.active_case_id == "CL-2048"
    assert "rules+llm" in [e.detail.get("source") for e in a.state.trace if e.kind == "nlu"]
    if provider == "anthropic":
        assert "anxiety" in r["meta"]["emotions"]


def test_llm_http_failure_falls_back_to_templates(repo, settings, monkeypatch):
    import sop_agent.llm as llm_mod
    monkeypatch.setattr(llm_mod.httpx, "post", lambda *a, **k: _Resp({"error": "bad key"}, status=401))
    a = SOPAgent(repo, settings, LLMSettings(provider="anthropic", api_key="bad"))
    a.start()
    r = a.handle(DEMO)
    assert r["meta"]["mode"].startswith("template") and a.state.active_case_id == "CL-2048"
