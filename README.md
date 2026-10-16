# Insurance Claims SOP Agent

A **Standard Operating Procedure (SOP) harness** for an insurance claims support agent. The agent follows a fixed
business workflow but still talks naturally:

```
VERIFY_ID  ──►  RESOLVE_INTENT  ──►  PROCESS_CASE  ──►  POST_PROCESS  ──►  COMPLETE
  strict           guided              grounded            strict
      └──────────────── any phase ──► HANDOFF (human representative) ◄─────────┘
```

**Core idea:** *code owns the workflow, the LLM owns the language.* A deterministic state machine decides the phase,
the safety gates, which tools may run and what must be said. The LLM (optional) interprets messy input and phrases
replies, with a different amount of freedom in each phase, and every LLM reply is checked by validators before it
is sent. If the LLM is missing, fails, or breaks a rule, the agent uses a safe template reply, so the demo works with
or without an API key.

---

## 1. Quick start

### Option A: Docker

```bash
cp .env.example .env              # optionally put your token in LLM_API_KEY=...
docker compose up --build         # or: docker build -t sop-agent . && docker run -p 8000:8000 --env-file .env sop-agent
open http://localhost:8000
```

### Option B: Local Python (3.10+)

```bash
pip install -r requirements.txt
cp .env.example .env              # optional
uvicorn sop_agent.server:app --port 8000
open http://localhost:8000
```

### Providing the AI model API token

You can provide the token in either of these ways:

| Where | How |
|---|---|
| Environment / `.env` | `LLM_PROVIDER=anthropic` + `LLM_API_KEY=sk-ant-...` (the server also reads `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) |
| Test UI | **Settings** → choose a provider → paste the **API auth token** → **Test token** → **Save & restart**. The token is kept in server memory for that session only. |

Providers: `anthropic` (default model `claude-sonnet-5-5`), `openai` (any OpenAI-compatible endpoint through
`LLM_BASE_URL`), or `none`. `LLM_MODEL` overrides the model.

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 25 scenario tests, offline (fake LLM + mocked HTTP for the model path)
```

---

## 2. Demo script (test UI)

The UI is a chat window with an **SOP inspector** beside it. The inspector shows the phase stepper, the
verification gate, cross-phase memory (with the phase and turn when each fact was captured), guards, the email
preview, the handoff ticket, and a live trace of tools, gates, and transitions. The chips above the input box fill
in sample caller lines.

| # | Caller says | What to watch |
|---|---|---|
| 1 | *Hi, I'm calling about my denied healthcare claim from January.* | Stays in **VERIFY_ID**. The hint is stored in memory with phase=VERIFY_ID. No claim data is shared. |
| 2 | *I already told you who I am. This is ridiculous. Just tell me why my claim was denied.* | Detects anger/frustration. Responds with empathy, explains why verification is needed, and lists the allowed ID options. Claim details stay hidden. |
| 3 | *My name is Margaret Chen, DOB is 1985-03-15, SSN last four 4472* | 3 of 5 fields match P9 → verified → **RESOLVE_INTENT** uses the remembered hint → CL-2048 is found without asking again → **PROCESS_CASE** explains the denial, the needed documents, and the deadline. |
| 4 | *I don't have the pathology report anymore — what can I do?* | Grounded answer from `document_alternative_guidance`. |
| 5 | *How soon do I need to submit, and how long does it take after I submit?* | Retrieves the `submission_timing` and `processing_time_after_submission` rules. |
| 6 | *No, that's all, thanks.* | **POST_PROCESS** offers an email summary to the masked address on file. |
| 7 | *Yes, please send it.* (or *No*) | The email is written to `data/outbox/*.json` (or sent over SMTP if configured), then **COMPLETE**. |

Other scenarios to try: *What is RL?* ×3–4 (polite refusal, then a human offer, then handoff), the **Rep caller**
flow (David Chen for his mother: verification, then a consent request that is *pending → approved*; set the consent
scenario to `timeout` in Settings to see escalation), *Refuse SSN* (other ID fields are offered instead), a wrong
DOB (no hint about which field was wrong; lockout after 3 tries), *what about claim CL-3001?* (another party's
claim, refused), and *Can I talk to a real person?* at any point.

> **Demo date.** The fixtures are set in early 2026 (CL-2048 was denied in Jan 2026 with an appeal deadline of
> 2026-03-18), so `DEMO_TODAY` defaults to `2026-02-20`. Set `DEMO_TODAY=real` to use the system clock. The agent
> then correctly says that the deadline has passed and points to a human representative.

---

## 3. How the requirements are met

### Different freedom per phase

| Phase | Freedom | What the LLM may do | What code enforces |
|---|---|---|---|
| VERIFY_ID | **strict** | Extract fields; rephrase the template draft | ≥3 of {name, DOB, phone, email, SSN/national-ID last 4} must match **one** record with **no** mismatches. Policy number is used only for lookup and is not counted. A failed match never says which field was wrong. 3 failures → handoff. The LLM **never sees claim data** in this phase (it is kept out of the prompt, not just forbidden by instruction). |
| RESOLVE_INTENT | **guided** | Interpret messy wording, pick among the offered options | Candidates come only from the verified party's claims. A chosen case must belong to that party. Unsupported intents (new claim, policy change) → human offer |
| PROCESS_CASE | **grounded** | Answer follow-ups freely, but only from the FACTS packet (claim record + retrieved guidance) | A validator rejects any number, date, amount, or claim ID that is not in the facts. Other parties' claims are unreachable. |
| POST_PROCESS | **strict** | Rephrase the offer | Explicit yes/no consent. Sends only to the email on file (a different address is refused). The summary is built deterministically from session records. |

### SOP enforcement layers
1. **Transition graph** (`state.ALLOWED_TRANSITIONS`). An illegal phase jump raises `SOPViolation`.
2. **Gates** (`agent._gate`). For example, `RESOLVE_INTENT` requires a verified caller (plus consent for
   representatives), and `PROCESS_CASE` requires a resolved case that the caller owns.
3. **Phase-scoped tools** (`tools.PHASE_TOOLS`). `get_claim` called during VERIFY_ID raises `ToolGateError`. Claim
   tools also check verification and consent, and they are scoped to the verified `party_id`.
4. **Response validators** (`responder.validate_reply`). They check for claim-data leaks before verification, record
   PII leaks, ungrounded numbers or claim IDs, and missing required content (masked email, ticket, consent ID).
   If any check fails, the safe template is sent instead. The UI shows the reason.
5. **Anti-hallucination on extraction** (`nlu.merge_llm`). An identity value from the LLM is accepted only if it can
   be traced back to the caller's actual words.

### Memory across phases
`Memory` stores every useful fact **when it is said**, with provenance (`turn`, `phase`, `source`): identity fields,
caller/representative info, case hints (type, status, month, year, case ID), intent, and questions asked before
verification. Example: *"my denied healthcare claim from January"* during VERIFY_ID → `status=denied,
case_type=healthcare, month=1`. After verification these hints resolve CL-2048 directly ("Based on what you
mentioned earlier…"), and the deferred question is answered in PROCESS_CASE.

### Scope control
Every message gets a scope label: `in_scope`, `chitchat`, or `out_of_scope` (rules + LLM; anything that yields a
useful slot counts as in scope). An out-of-scope question gets a polite refusal plus a pointer back to the current
step. On the 3rd consecutive one (`MAX_OUT_OF_SCOPE`) the agent offers a human representative, and if the caller
keeps going it transfers them. A relevant message resets the counter.

### Bonus: emotional support and SOP recovery
- Detects **frustration, anger, anxiety, confusion, and refusal** (rules + LLM).
- Leads with **empathy** before moving the workflow forward, then **explains why** (verification protects health and
  financial data; consent protects the policyholder's privacy).
- **Offers acceptable alternatives**: other identity fields (refused or unavailable ones are removed from the list),
  a different ID type (national ID), or a human transfer.
- **Knows when to stop**: on the 2nd pushback it gently mentions the human option; on the 3rd it explicitly offers a
  transfer; after that it escalates with a handoff ticket that carries context (verification state, intent, hints).
  It never sends claim details to someone who is not verified.
- Representative calls: verify the policyholder's identity → check that the caller is a registered representative →
  request consent (`consent_scenarios.json`: `default` = pending→approved, `timeout` = stays pending → handoff).

---

## 4. Architecture

```
sop_agent/
  config.py      env/.env settings, LLM settings, DEMO_TODAY
  state.py       Phase enum, transition graph, per-phase freedom, Memory (slots with provenance), SessionState, trace
  data.py        read-only fixtures repository (+ sensitive markers for leak detection)
  normalize.py   name/phone/email/DOB/SSN normalization, masking
  nlu.py         rule-based extraction + optional LLM forced-tool extraction, safe merge
  tools.py       phase-gated Toolbox: verify_identity, lookup_representative, request/check_consent,
                 list/find/get_claim, document & follow-up guidance retrieval, compose/send email, transfer_to_human
  agent.py       the SOP orchestrator (phase handlers, gates, auto-advance, emotion/scope policy)
  responder.py   TurnPlan → template draft → (LLM phrasing by freedom level) → validators → reply
  llm.py         tiny httpx client for Anthropic Messages / OpenAI-compatible Chat Completions
  server.py      FastAPI: /api/sessions, /api/sessions/{id}/messages, /api/llm/test, static UI
  static/index.html  test UI (chat + SOP inspector)
apps/insurance_claims/fixtures/   starter data (unchanged)
tests/test_sop_agent.py           25 scenario tests
```

**What happens in one turn:** NLU (rules, plus the LLM if configured) → store new facts in memory → global rules
(human request, scope guard) → the current phase handler runs tools through the gated Toolbox and may move to the
next phase (several in one turn when the caller gives everything at once) → a `TurnPlan` is built (content blocks +
FACTS + required strings) → responder → validators → reply + state snapshot.

### HTTP API

| Method | Path | Body |
|---|---|---|
| `POST` | `/api/sessions` | `{provider?, api_key?, model?, base_url?, consent_scenario?}` → greeting + state |
| `POST` | `/api/sessions/{id}/messages` | `{text}` → `{reply, meta, state}` |
| `GET`  | `/api/sessions/{id}` | full state + transcript |
| `POST` | `/api/llm/test` | checks that a token works |

---

## 5. Notes and limitations
- Sessions live in memory (fine for a demo). For production, put `SessionState` (a pydantic model, already
  serialisable) in Redis or a database.
- Email is a mock outbox by default. Set `SMTP_*` to send real mail.
- Rule-only mode covers the demo scenarios and common phrasings. Configure an LLM to handle freer language; the
  SOP guarantees stay the same either way.
