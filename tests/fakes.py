"""A minimal fake Anthropic client for testing generate.py without network calls.

Duck-types only the attributes generate.py actually reads off a response
(`stop_reason`, `stop_details.category`, `parsed_output`, `usage.*`) rather
than constructing real SDK response objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeStopDetails:
    category: str | None = None


@dataclass
class FakeResponse:
    stop_reason: str = "end_turn"
    parsed_output: Any = None
    stop_details: FakeStopDetails | None = None
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessagesResource:
    """Records every call to `.parse(**kwargs)` and replays canned responses."""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessagesResource ran out of canned responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeMessagesResource(responses)
