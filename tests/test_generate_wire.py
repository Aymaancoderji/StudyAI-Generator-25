"""Exercises the real anthropic.Anthropic client against a mocked HTTP
transport — no network call, but real request serialization and response
parsing, unlike test_generate.py's duck-typed FakeClient.

This is what catches SDK-shape mistakes the fake client can't: wrong
parameter names, a schema anthropic.messages.parse can't serialize, cache
breakpoints landing on the wrong block, etc.
"""

from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from studycards.chunk import Chunk, estimate_tokens
from studycards.generate import generate_cards

CARD_PAYLOAD = {
    "cards": [
        {
            "question": "What is the function of mitochondria?",
            "answer": "They generate most of a cell's chemical energy as ATP.",
            "card_type": "definition",
            "difficulty": "easy",
            "topic": "Mitochondria",
            "source_excerpt": "Mitochondria are membrane-bound organelles",
        }
    ]
}

CHUNK_TEXT = "Mitochondria are membrane-bound organelles that generate ATP."


def make_chunk() -> Chunk:
    return Chunk(
        index=0, text=CHUNK_TEXT, start_page=1, end_page=1,
        token_count=estimate_tokens(CHUNK_TEXT),
    )


@pytest.fixture
def wired_client():
    """A real Anthropic client whose HTTP layer is a MockTransport.

    Yields (client, captured) where captured['body'] holds the last request
    body once a call has been made.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["body"] = body
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(CARD_PAYLOAD)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 250,
                    "output_tokens": 40,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    client = anthropic.Anthropic(
        api_key="sk-ant-fake-for-transport-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, captured


class TestRealSerialization:
    def test_round_trip_produces_expected_cards(self, wired_client):
        client, _ = wired_client
        result = generate_cards(make_chunk(), client=client, model="claude-opus-5")
        assert [c.question for c in result.cards] == [
            "What is the function of mitochondria?"
        ]
        assert result.input_tokens == 250
        assert result.cache_read_tokens == 200

    def test_output_config_schema_forbids_extra_properties(self, wired_client):
        client, captured = wired_client
        generate_cards(make_chunk(), client=client)
        schema = captured["body"]["output_config"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["Card"]["additionalProperties"] is False

    def test_effort_lands_in_output_config(self, wired_client):
        client, captured = wired_client
        generate_cards(make_chunk(), client=client, effort="high")
        assert captured["body"]["output_config"]["effort"] == "high"

    def test_cache_control_lands_on_system_block(self, wired_client):
        client, captured = wired_client
        generate_cards(make_chunk(), client=client)
        assert captured["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_lands_on_chunk_content_block(self, wired_client):
        client, captured = wired_client
        generate_cards(make_chunk(), client=client)
        content = captured["body"]["messages"][0]["content"][0]
        assert content["cache_control"] == {"type": "ephemeral"}
        assert CHUNK_TEXT in content["text"]

    def test_required_card_fields_are_all_present_in_schema(self, wired_client):
        client, captured = wired_client
        generate_cards(make_chunk(), client=client)
        required = set(captured["body"]["output_config"]["format"]["schema"]["$defs"]["Card"]["required"])
        assert required == {
            "question", "answer", "card_type", "difficulty", "topic", "source_excerpt",
        }
