"""Configuration and Anthropic client construction.

The API key is never hardcoded. It lives in a gitignored `.env` file; see
`.env.example` for the template. The SDK reads ANTHROPIC_API_KEY from the
environment itself, so we only need to load `.env` before constructing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default to Opus 5 — the card-writing quality difference over smaller models
# is the whole point of the project. Override via STUDYCARDS_MODEL in .env.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class MissingAPIKeyError(RuntimeError):
    """Raised when no ANTHROPIC_API_KEY is available."""


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str

    @property
    def has_api_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load `.env` (if present) and read project settings from the environment."""
    load_dotenv(PROJECT_ROOT / ".env")

    effort = os.environ.get("STUDYCARDS_EFFORT", DEFAULT_EFFORT).lower()
    if effort not in VALID_EFFORTS:
        raise ValueError(
            f"STUDYCARDS_EFFORT must be one of {VALID_EFFORTS}, got {effort!r}"
        )

    return Settings(
        model=os.environ.get("STUDYCARDS_MODEL", DEFAULT_MODEL),
        effort=effort,
    )


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Return a shared Anthropic client, or explain how to set the key up."""
    settings = load_settings()
    if not settings.has_api_key:
        raise MissingAPIKeyError(
            "No ANTHROPIC_API_KEY found.\n"
            f"  1. cp {PROJECT_ROOT / '.env.example'} {PROJECT_ROOT / '.env'}\n"
            "  2. Paste your key into .env (it is gitignored)\n"
            "Get a key at https://platform.claude.com/settings/keys"
        )
    # The SDK picks the key up from the environment — don't pass it explicitly.
    return anthropic.Anthropic()
