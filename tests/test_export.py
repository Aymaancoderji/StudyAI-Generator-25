from __future__ import annotations

import csv
import io
import json

from studycards.export import to_anki_csv, to_json, to_quizlet_tsv
from studycards.schema import Card


def make_card(**overrides) -> Card:
    defaults = dict(
        question="What is the function of mitochondria?",
        answer="They generate most of a cell's chemical energy as ATP.",
        card_type="definition",
        difficulty="easy",
        topic="Cell Biology",
        source_excerpt="Mitochondria are membrane-bound organelles",
    )
    defaults.update(overrides)
    return Card(**defaults)


class TestAnkiCsv:
    def test_header_row(self):
        rows = list(csv.reader(io.StringIO(to_anki_csv([make_card()]))))
        assert rows[0] == ["Front", "Back", "Tags"]

    def test_question_and_answer_preserved(self):
        card = make_card()
        rows = list(csv.reader(io.StringIO(to_anki_csv([card]))))
        assert rows[1][0] == card.question
        assert rows[1][1] == card.answer

    def test_multiword_topic_becomes_single_tag(self):
        card = make_card(topic="Cell Biology")
        rows = list(csv.reader(io.StringIO(to_anki_csv([card]))))
        tags = rows[1][2].split()
        assert "Cell_Biology" in tags

    def test_tags_include_type_and_difficulty(self):
        card = make_card(card_type="cloze", difficulty="hard")
        rows = list(csv.reader(io.StringIO(to_anki_csv([card]))))
        tags = rows[1][2].split()
        assert "cloze" in tags
        assert "hard" in tags

    def test_empty_deck_still_has_header(self):
        rows = list(csv.reader(io.StringIO(to_anki_csv([]))))
        assert rows == [["Front", "Back", "Tags"]]

    def test_embedded_comma_does_not_break_columns(self):
        card = make_card(answer="ATP, the cell's energy currency.")
        rows = list(csv.reader(io.StringIO(to_anki_csv([card]))))
        assert len(rows[1]) == 3
        assert rows[1][1] == card.answer

    def test_multiple_cards_produce_multiple_rows(self):
        rows = list(csv.reader(io.StringIO(to_anki_csv([make_card(), make_card()]))))
        assert len(rows) == 3  # header + 2 cards


class TestQuizletTsv:
    def test_no_header_row(self):
        card = make_card()
        lines = to_quizlet_tsv([card]).strip("\n").split("\n")
        assert len(lines) == 1
        assert lines[0].split("\t") == [card.question, card.answer]

    def test_tab_separated_two_fields_per_line(self):
        text = to_quizlet_tsv([make_card(), make_card()])
        lines = text.strip("\n").split("\n")
        assert len(lines) == 2
        for line in lines:
            assert len(line.split("\t")) == 2

    def test_embedded_tab_in_field_is_collapsed(self):
        card = make_card(question="What\tis ATP?")
        text = to_quizlet_tsv([card])
        # A literal tab inside the question would create a phantom third
        # field — must not happen.
        assert len(text.strip("\n").split("\t")) == 2

    def test_embedded_newline_in_field_is_collapsed(self):
        card = make_card(answer="Line one\nLine two")
        text = to_quizlet_tsv([card])
        lines = text.strip("\n").split("\n")
        assert len(lines) == 1  # the embedded newline must not create a new row

    def test_empty_deck_produces_empty_string(self):
        assert to_quizlet_tsv([]) == ""


class TestJsonExport:
    def test_round_trips_all_fields(self):
        card = make_card()
        data = json.loads(to_json([card]))
        assert data == [card.model_dump()]

    def test_multiple_cards(self):
        cards = [make_card(), make_card(topic="Genetics")]
        data = json.loads(to_json(cards))
        assert len(data) == 2
        assert data[1]["topic"] == "Genetics"

    def test_empty_deck_is_empty_array(self):
        assert json.loads(to_json([])) == []

    def test_non_ascii_characters_preserved_unescaped(self):
        card = make_card(answer="La mitochondrie produit de l'énergie.")
        text = to_json([card])
        assert "é" in text  # ensure_ascii=False keeps it readable, not \u escaped
