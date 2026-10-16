"""Runtime configuration (env vars, optionally loaded from a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
DEFAULT_BASE_URLS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1"}


@dataclass
class LLMSettings:
    provider: str = "anthropic"          # anthropic | openai (any OpenAI-compatible endpoint) | none
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout_s: float = 30.0

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and bool(self.api_key)

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    def resolved_base_url(self) -> str:
        return (self.base_url or DEFAULT_BASE_URLS.get(self.provider, "")).rstrip("/")


@dataclass
class Settings:
    fixtures_dir: Path = field(default_factory=lambda: Path(os.getenv("FIXTURES_DIR", ROOT / "apps/insurance_claims/fixtures")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "data")))
    # The fixtures were authored around early 2026 (claim CL-2048 denied Jan 2026, appeal deadline 2026-03-18).
    # DEMO_TODAY pins "today" so deadline math matches the scenario. Set DEMO_TODAY=real to use the system date.
    demo_today: str = field(default_factory=lambda: os.getenv("DEMO_TODAY", "2026-02-20"))
    min_identity_fields: int = int(os.getenv("MIN_IDENTITY_FIELDS", "3"))
    max_verification_failures: int = int(os.getenv("MAX_VERIFICATION_FAILURES", "3"))
    max_out_of_scope: int = int(os.getenv("MAX_OUT_OF_SCOPE", "3"))
    max_persuasion_attempts: int = int(os.getenv("MAX_PERSUASION_ATTEMPTS", "3"))
    consent_scenario: str = field(default_factory=lambda: os.getenv("CONSENT_SCENARIO", "default"))

    def today(self) -> date:
        if self.demo_today and self.demo_today.lower() != "real":
            return date.fromisoformat(self.demo_today)
        return date.today()

    def default_llm(self) -> LLMSettings:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        key = os.getenv("LLM_API_KEY") or None
        if not provider:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "anthropic"
        if not key:
            key = os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic" else os.getenv("OPENAI_API_KEY")
        return LLMSettings(
            provider=provider,
            api_key=key or None,
            model=os.getenv("LLM_MODEL") or None,
            base_url=os.getenv("LLM_BASE_URL") or None,
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
        )


settings = Settings()
