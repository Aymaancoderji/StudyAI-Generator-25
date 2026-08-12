from __future__ import annotations

import pytest

from studycards.chunk import Chunk, estimate_tokens
from studycards.generate import (
    DeckGenerationSummary,
    GenerationError,
    generate_cards,
    generate_deck,
)
from studycards.schema import Card, CardBatch

from .fakes import FakeClient, FakeResponse, FakeStopDetails, FakeUsage

CHUNK_TEXT = (
    "Mitochondria are membrane-bound organelles that generate most of the "
    "chemical energy needed to power a cell's biochemical reactions. "
    "Chemical energy is stored in a molecule called ATP."
)


def make_chunk(text: str = CHUNK_TEXT, index: int = 0) -> Chunk:
    return Chunk(
        index=index,
        text=text,
        start_page=1,
        end_page=1,
        token_count=estimate_tokens(text),
    )


def make_card(**overrides) -> Card:
    defaults = dict(
        question="What is the function of mitochondria?",
        answer="They generate most of a cell's chemical energy as ATP.",
        card_type="definition",
        difficulty="easy",
        topic="Mitochondria",
        source_excerpt="Mitochondria are membrane-bound organelles",
    )
    defaults.update(overrides)
    return Card(**defaults)


class TestGenerateCardsHappyPath:
    def test_returns_cards_from_parsed_output(self):
        card = make_card()
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[card]))])

        result = generate_cards(make_chunk(), client=client)

        assert result.cards == [card]
        assert result.stop_reason == "end_turn"
        assert result.dropped_ungrounded == 0

    def test_usage_is_propagated(self):
        usage = FakeUsage(
            input_tokens=321,
            output_tokens=64,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=0,
        )
        client = FakeClient(
            [FakeResponse(parsed_output=CardBatch(cards=[make_card()]), usage=usage)]
        )

        result = generate_cards(make_chunk(), client=client)

        assert result.input_tokens == 321
        assert result.output_tokens == 64
        assert result.cache_read_tokens == 200

    def test_chunk_index_is_preserved(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        result = generate_cards(make_chunk(index=7), client=client)
        assert result.chunk_index == 7


class TestRequestShape:
    def test_output_format_is_cardbatch(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client)
        assert client.messages.calls[0]["output_format"] is CardBatch

    def test_effort_forwarded_via_output_config(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client, effort="xhigh")
        assert client.messages.calls[0]["output_config"] == {"effort": "xhigh"}

    def test_model_is_forwarded(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client, model="claude-opus-5")
        assert client.messages.calls[0]["model"] == "claude-opus-5"

    def test_system_prompt_has_cache_breakpoint(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client)
        system = client.messages.calls[0]["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_chunk_text_has_cache_breakpoint(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client)
        content = client.messages.calls[0]["messages"][0]["content"]
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_chunk_text_appears_in_prompt(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client)
        content = client.messages.calls[0]["messages"][0]["content"]
        assert CHUNK_TEXT in content[0]["text"]

    def test_requested_card_types_appear_in_prompt(self):
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[]))])
        generate_cards(make_chunk(), client=client, card_types=["cloze", "compare"])
        content = client.messages.calls[0]["messages"][0]["content"]
        assert "cloze" in content[0]["text"]
        assert "compare" in content[0]["text"]


class TestRefusalHandling:
    def test_refusal_raises_generation_error(self):
        client = FakeClient(
            [
                FakeResponse(
                    stop_reason="refusal",
                    stop_details=FakeStopDetails(category="cyber"),
                )
            ]
        )
        with pytest.raises(GenerationError, match="cyber"):
            generate_cards(make_chunk(index=3), client=client)

    def test_refusal_error_carries_chunk_index(self):
        client = FakeClient(
            [FakeResponse(stop_reason="refusal", stop_details=FakeStopDetails())]
        )
        with pytest.raises(GenerationError) as exc_info:
            generate_cards(make_chunk(index=3), client=client)
        assert exc_info.value.chunk_index == 3

    def test_missing_stop_details_does_not_crash(self):
        client = FakeClient([FakeResponse(stop_reason="refusal", stop_details=None)])
        with pytest.raises(GenerationError, match="category=None"):
            generate_cards(make_chunk(), client=client)


class TestUnparsedOutput:
    def test_none_parsed_output_raises(self):
        client = FakeClient([FakeResponse(stop_reason="max_tokens", parsed_output=None)])
        with pytest.raises(GenerationError, match="max_tokens"):
            generate_cards(make_chunk(), client=client)


class TestGroundingFilter:
    def test_card_with_verbatim_excerpt_is_kept(self):
        card = make_card(source_excerpt="Chemical energy is stored in a molecule called ATP")
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[card]))])
        result = generate_cards(make_chunk(), client=client)
        assert result.cards == [card]
        assert result.dropped_ungrounded == 0

    def test_card_with_fabricated_excerpt_is_dropped(self):
        fabricated = make_card(source_excerpt="Mitochondria were discovered by Aristotle")
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[fabricated]))])
        result = generate_cards(make_chunk(), client=client)
        assert result.cards == []
        assert result.dropped_ungrounded == 1

    def test_mixed_batch_keeps_only_grounded_cards(self):
        real = make_card(source_excerpt="Mitochondria are membrane-bound organelles")
        fake = make_card(source_excerpt="This sentence never appears anywhere")
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[real, fake]))])
        result = generate_cards(make_chunk(), client=client)
        assert result.cards == [real]
        assert result.dropped_ungrounded == 1

    def test_whitespace_differences_do_not_break_matching(self):
        # Model output can normalize internal whitespace/newlines relative to
        # the source; the check should tolerate that without being fooled by
        # genuinely absent text.
        card = make_card(source_excerpt="Mitochondria   are membrane-bound\norganelles")
        client = FakeClient([FakeResponse(parsed_output=CardBatch(cards=[card]))])
        result = generate_cards(make_chunk(), client=client)
        assert result.cards == [card]


class TestGenerateDeck:
    def test_aggregates_cards_across_chunks(self):
        client = FakeClient(
            [
                FakeResponse(parsed_output=CardBatch(cards=[make_card()])),
                FakeResponse(parsed_output=CardBatch(cards=[make_card(topic="ATP")])),
            ]
        )
        summary = generate_deck([make_chunk(index=0), make_chunk(index=1)], client=client)

        assert len(summary.cards) == 2
        assert len(summary.results) == 2
        assert summary.errors == []

    def test_continues_past_a_refused_chunk(self):
        client = FakeClient(
            [
                FakeResponse(stop_reason="refusal", stop_details=FakeStopDetails(category="cyber")),
                FakeResponse(parsed_output=CardBatch(cards=[make_card()])),
            ]
        )
        summary = generate_deck([make_chunk(index=0), make_chunk(index=1)], client=client)

        assert len(summary.results) == 1
        assert len(summary.errors) == 1
        assert summary.errors[0].chunk_index == 0
        assert len(summary.cards) == 1

    def test_all_chunks_still_attempted_after_a_failure(self):
        client = FakeClient(
            [
                FakeResponse(stop_reason="refusal", stop_details=FakeStopDetails()),
                FakeResponse(stop_reason="refusal", stop_details=FakeStopDetails()),
                FakeResponse(parsed_output=CardBatch(cards=[make_card()])),
            ]
        )
        chunks = [make_chunk(index=i) for i in range(3)]
        summary = generate_deck(chunks, client=client)

        assert len(client.messages.calls) == 3
        assert len(summary.errors) == 2
        assert len(summary.results) == 1

    def test_token_totals_sum_across_chunks(self):
        client = FakeClient(
            [
                FakeResponse(
                    parsed_output=CardBatch(cards=[]),
                    usage=FakeUsage(input_tokens=100, output_tokens=20),
                ),
                FakeResponse(
                    parsed_output=CardBatch(cards=[]),
                    usage=FakeUsage(input_tokens=150, output_tokens=30),
                ),
            ]
        )
        summary = generate_deck([make_chunk(index=0), make_chunk(index=1)], client=client)

        assert summary.total_input_tokens == 250
        assert summary.total_output_tokens == 50

    def test_on_progress_called_once_per_chunk(self):
        client = FakeClient(
            [
                FakeResponse(parsed_output=CardBatch(cards=[])),
                FakeResponse(parsed_output=CardBatch(cards=[])),
            ]
        )
        seen: list[int] = []
        chunks = [make_chunk(index=0), make_chunk(index=1)]

        generate_deck(
            chunks, client=client, on_progress=lambda chunk, summary: seen.append(chunk.index)
        )

        assert seen == [0, 1]

    def test_empty_chunk_list_returns_empty_summary(self):
        summary = generate_deck([], client=FakeClient([]))
        assert isinstance(summary, DeckGenerationSummary)
        assert summary.cards == []
        assert summary.results == []
        assert summary.errors == []
