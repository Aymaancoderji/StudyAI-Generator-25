from __future__ import annotations

import pandas as pd
import pytest

from studycards.deck import (
    Deck,
    cards_to_dataframe,
    coverage_by_topic,
    dataframe_to_cards,
    deduplicate,
    filter_quality,
    find_near_duplicates,
    load_csv,
    load_parquet,
    save_csv,
    save_parquet,
)
from studycards.generate import DeckGenerationSummary, GenerationResult
from studycards.schema import Card


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


# --------------------------------------------------------------------------
# Card <-> DataFrame
# --------------------------------------------------------------------------


class TestCardsToDataFrame:
    def test_all_fields_round_trip(self):
        card = make_card()
        df = cards_to_dataframe([card])
        row = df.iloc[0]
        assert row["question"] == card.question
        assert row["answer"] == card.answer
        assert row["card_type"] == card.card_type
        assert row["difficulty"] == card.difficulty
        assert row["topic"] == card.topic
        assert row["source_excerpt"] == card.source_excerpt

    def test_ids_are_sequential(self):
        df = cards_to_dataframe([make_card(), make_card(), make_card()])
        assert list(df["id"]) == [0, 1, 2]

    def test_empty_list_produces_empty_frame_with_columns(self):
        df = cards_to_dataframe([])
        assert len(df) == 0
        assert "question" in df.columns

    def test_chunk_indices_attached(self):
        df = cards_to_dataframe([make_card(), make_card()], chunk_indices=[2, 5])
        assert list(df["chunk_index"]) == [2, 5]

    def test_chunk_indices_default_to_none(self):
        df = cards_to_dataframe([make_card()])
        assert df["chunk_index"].iloc[0] is None

    def test_mismatched_chunk_indices_length_raises(self):
        with pytest.raises(ValueError, match="chunk_indices"):
            cards_to_dataframe([make_card(), make_card()], chunk_indices=[0])


class TestDataFrameToCards:
    def test_round_trip_equal(self):
        original = [make_card(topic="A"), make_card(topic="B", card_type="cloze")]
        df = cards_to_dataframe(original)
        restored = dataframe_to_cards(df)
        assert restored == original

    def test_invalid_row_raises_validation_error(self):
        df = cards_to_dataframe([make_card()])
        df.loc[0, "card_type"] = "not_a_real_type"
        with pytest.raises(Exception):
            dataframe_to_cards(df)


# --------------------------------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------------------------------


class TestFindNearDuplicates:
    def test_identical_questions_same_topic_are_grouped(self):
        df = cards_to_dataframe(
            [
                make_card(question="What is ATP?", topic="Mitochondria"),
                make_card(question="What is ATP?", topic="Mitochondria"),
            ]
        )
        groups = find_near_duplicates(df)
        assert groups == [{0, 1}]

    def test_near_identical_phrasing_is_grouped(self):
        df = cards_to_dataframe(
            [
                make_card(
                    question="What is the function of mitochondria in a cell?",
                    topic="Mitochondria",
                ),
                make_card(
                    question="What is the function of mitochondria in the cell?",
                    topic="Mitochondria",
                ),
            ]
        )
        groups = find_near_duplicates(df, threshold=0.85)
        assert groups == [{0, 1}]

    def test_distinct_questions_same_topic_are_not_grouped(self):
        df = cards_to_dataframe(
            [
                make_card(question="What is the function of mitochondria?", topic="X"),
                make_card(question="Where does the Calvin cycle occur?", topic="X"),
            ]
        )
        assert find_near_duplicates(df) == []

    def test_same_question_different_topics_not_grouped(self):
        # Scoped to topic on purpose — a shared phrase across unrelated
        # topics is not evidence of duplication.
        df = cards_to_dataframe(
            [
                make_card(question="What is the main product?", topic="Photosynthesis"),
                make_card(question="What is the main product?", topic="Respiration"),
            ]
        )
        assert find_near_duplicates(df) == []

    def test_topic_matching_is_case_insensitive(self):
        df = cards_to_dataframe(
            [
                make_card(question="What is ATP?", topic="Mitochondria"),
                make_card(question="What is ATP?", topic="MITOCHONDRIA"),
            ]
        )
        assert find_near_duplicates(df) == [{0, 1}]

    def test_transitive_duplicates_merge_into_one_group(self):
        # A~B and B~C should merge into {A, B, C}, not stay as two groups.
        df = cards_to_dataframe(
            [
                make_card(question="What is the function of mitochondria?", topic="X"),
                make_card(question="What is the function of the mitochondria?", topic="X"),
                make_card(question="What is the function of a mitochondria?", topic="X"),
            ]
        )
        groups = find_near_duplicates(df, threshold=0.8)
        assert groups == [{0, 1, 2}]

    def test_empty_dataframe_returns_no_groups(self):
        assert find_near_duplicates(cards_to_dataframe([])) == []

    def test_single_card_returns_no_groups(self):
        assert find_near_duplicates(cards_to_dataframe([make_card()])) == []


class TestDeduplicate:
    def test_keeps_lowest_id_drops_the_rest(self):
        df = cards_to_dataframe(
            [
                make_card(question="What is ATP?", topic="X"),
                make_card(question="What is ATP?", topic="X"),
                make_card(question="What is ATP?", topic="X"),
            ]
        )
        result = deduplicate(df)
        assert len(result.deck) == 1
        assert result.deck.iloc[0]["id"] == 0
        assert len(result.dropped) == 2

    def test_dropped_rows_record_duplicate_of(self):
        df = cards_to_dataframe(
            [
                make_card(question="What is ATP?", topic="X"),
                make_card(question="What is ATP?", topic="X"),
            ]
        )
        result = deduplicate(df)
        assert list(result.dropped["duplicate_of"]) == [0]

    def test_no_duplicates_leaves_deck_unchanged(self):
        df = cards_to_dataframe(
            [
                make_card(question="What is ATP?", topic="X"),
                make_card(question="Where is chlorophyll found?", topic="Y"),
            ]
        )
        result = deduplicate(df)
        assert len(result.deck) == 2
        assert len(result.dropped) == 0

    def test_empty_deck(self):
        result = deduplicate(cards_to_dataframe([]))
        assert len(result.deck) == 0
        assert len(result.dropped) == 0


# --------------------------------------------------------------------------
# Quality filters
# --------------------------------------------------------------------------


class TestFilterQuality:
    def test_good_card_is_kept(self):
        df = cards_to_dataframe([make_card()])
        result = filter_quality(df)
        assert len(result.deck) == 1
        assert len(result.dropped) == 0

    def test_short_answer_is_dropped(self):
        df = cards_to_dataframe([make_card(answer="ATP.")])
        result = filter_quality(df, min_answer_chars=15)
        assert len(result.deck) == 0
        assert result.dropped.iloc[0]["reason"] == "answer too short"

    def test_answer_embedded_in_question_is_dropped(self):
        df = cards_to_dataframe(
            [
                make_card(
                    question=(
                        "True or false: mitochondria generate a cell's "
                        "chemical energy as ATP through cellular respiration."
                    ),
                    answer="Mitochondria generate a cell's chemical energy as ATP",
                )
            ]
        )
        result = filter_quality(df, min_answer_chars=15)
        assert len(result.deck) == 0
        assert result.dropped.iloc[0]["reason"] == "answer restates question"

    def test_answer_equal_to_question_is_dropped(self):
        df = cards_to_dataframe(
            [make_card(question="Mitochondria produce ATP", answer="Mitochondria produce ATP")]
        )
        result = filter_quality(df, min_answer_chars=5)
        assert result.dropped.iloc[0]["reason"] == "answer restates question"

    def test_too_short_takes_priority_over_restates(self):
        # An answer that's both a substring of the question AND under the
        # length floor should be reported once, as "too short".
        df = cards_to_dataframe([make_card(question="What is ATP?", answer="ATP")])
        result = filter_quality(df, min_answer_chars=10)
        assert result.dropped.iloc[0]["reason"] == "answer too short"

    def test_mixed_batch_keeps_only_good_cards(self):
        df = cards_to_dataframe(
            [
                make_card(question="Q1", answer="A perfectly reasonable answer."),
                make_card(question="Q2", answer="no"),
            ]
        )
        result = filter_quality(df, min_answer_chars=15)
        assert len(result.deck) == 1
        assert result.deck.iloc[0]["question"] == "Q1"

    def test_empty_deck(self):
        result = filter_quality(cards_to_dataframe([]))
        assert len(result.deck) == 0
        assert len(result.dropped) == 0
        assert "reason" in result.dropped.columns


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


class TestCoverageByTopic:
    def test_counts_per_topic(self):
        df = cards_to_dataframe(
            [
                make_card(topic="Mitochondria"),
                make_card(topic="Mitochondria"),
                make_card(topic="Photosynthesis"),
            ]
        )
        cov = coverage_by_topic(df)
        counts = dict(zip(cov["topic"], cov["card_count"]))
        assert counts == {"Mitochondria": 2, "Photosynthesis": 1}

    def test_sorted_by_count_descending(self):
        df = cards_to_dataframe(
            [
                make_card(topic="Rare"),
                make_card(topic="Common"),
                make_card(topic="Common"),
                make_card(topic="Common"),
            ]
        )
        cov = coverage_by_topic(df)
        assert list(cov["topic"]) == ["Common", "Rare"]

    def test_difficulty_breakdown(self):
        df = cards_to_dataframe(
            [
                make_card(topic="X", difficulty="easy"),
                make_card(topic="X", difficulty="easy"),
                make_card(topic="X", difficulty="hard"),
            ]
        )
        cov = coverage_by_topic(df)
        row = cov[cov["topic"] == "X"].iloc[0]
        assert row["easy"] == 2
        assert row["medium"] == 0
        assert row["hard"] == 1

    def test_card_types_listed_uniquely(self):
        df = cards_to_dataframe(
            [
                make_card(topic="X", card_type="definition"),
                make_card(topic="X", card_type="definition"),
                make_card(topic="X", card_type="cloze"),
            ]
        )
        cov = coverage_by_topic(df)
        assert cov[cov["topic"] == "X"].iloc[0]["card_types"] == ["cloze", "definition"]

    def test_empty_deck_returns_empty_frame_with_columns(self):
        cov = coverage_by_topic(cards_to_dataframe([]))
        assert len(cov) == 0
        assert list(cov.columns) == [
            "topic", "card_count", "easy", "medium", "hard", "card_types",
        ]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


class TestPersistence:
    def test_csv_round_trip(self, tmp_path):
        df = cards_to_dataframe([make_card(), make_card(topic="Y")])
        path = tmp_path / "deck.csv"
        save_csv(df, path)
        restored = load_csv(path)
        assert list(restored["question"]) == list(df["question"])
        assert list(restored["topic"]) == list(df["topic"])

    def test_parquet_round_trip(self, tmp_path):
        df = cards_to_dataframe([make_card(), make_card(topic="Y")])
        path = tmp_path / "deck.parquet"
        save_parquet(df, path)
        restored = load_parquet(path)
        assert list(restored["question"]) == list(df["question"])
        assert list(restored["card_type"]) == list(df["card_type"])


# --------------------------------------------------------------------------
# Deck wrapper
# --------------------------------------------------------------------------


class TestDeck:
    def test_from_cards(self):
        deck = Deck.from_cards([make_card(), make_card()])
        assert len(deck) == 2

    def test_from_summary_flattens_results_and_tracks_chunk_index(self):
        summary = DeckGenerationSummary(
            results=[
                GenerationResult(
                    chunk_index=0,
                    cards=[make_card(topic="A")],
                    dropped_ungrounded=0,
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    stop_reason="end_turn",
                ),
                GenerationResult(
                    chunk_index=1,
                    cards=[make_card(topic="B"), make_card(topic="B")],
                    dropped_ungrounded=0,
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    stop_reason="end_turn",
                ),
            ],
            errors=[],
        )
        deck = Deck.from_summary(summary)
        assert len(deck) == 3
        assert list(deck.df["chunk_index"]) == [0, 1, 1]

    def test_cards_property_round_trips(self):
        original = [make_card(topic="A"), make_card(topic="B")]
        deck = Deck.from_cards(original)
        assert deck.cards == original

    def test_coverage_delegates_to_module_function(self):
        deck = Deck.from_cards([make_card(topic="A"), make_card(topic="A")])
        cov = deck.coverage()
        assert cov.iloc[0]["card_count"] == 2

    def test_clean_removes_duplicates_and_low_quality_cards(self):
        deck = Deck.from_cards(
            [
                make_card(question="What is ATP?", topic="X"),
                make_card(question="What is ATP?", topic="X"),  # near-dup
                make_card(question="Q2", answer="short", topic="Y"),  # too short
                make_card(question="Q3", answer="A perfectly good answer here.", topic="Z"),
            ]
        )
        cleaned, stats = deck.clean(min_answer_chars=15)
        assert len(cleaned) == 2
        assert stats["duplicates_dropped"] == 1
        assert stats["quality_dropped"] == 1

    def test_csv_round_trip_via_deck(self, tmp_path):
        deck = Deck.from_cards([make_card(), make_card(topic="Y")])
        path = tmp_path / "deck.csv"
        deck.to_csv(path)
        restored = Deck.read_csv(path)
        assert len(restored) == len(deck)
        assert list(restored.df["topic"]) == list(deck.df["topic"])

    def test_parquet_round_trip_via_deck(self, tmp_path):
        deck = Deck.from_cards([make_card(), make_card(topic="Y")])
        path = tmp_path / "deck.parquet"
        deck.to_parquet(path)
        restored = Deck.read_parquet(path)
        assert len(restored) == len(deck)
