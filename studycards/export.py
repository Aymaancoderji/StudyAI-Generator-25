"""Serialize a deck to formats other tools can import.

Every function returns a string (not a file) so the Streamlit app can hand it
straight to `st.download_button` without touching disk.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Sequence

from .schema import Card


def to_anki_csv(cards: Sequence[Card]) -> str:
    """Anki-importable CSV: Front, Back, Tags.

    Anki treats each whitespace-separated word in the Tags field as a
    separate tag, so a multi-word topic like "Cell Biology" is joined with an
    underscore rather than left as two tags.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Front", "Back", "Tags"])
    for card in cards:
        tags = " ".join(
            [card.card_type, card.difficulty, card.topic.strip().replace(" ", "_")]
        )
        writer.writerow([card.question, card.answer, tags])
    return buf.getvalue()


def to_quizlet_tsv(cards: Sequence[Card]) -> str:
    """Quizlet's plain-text import format: one card per line, term<TAB>definition.

    Quizlet's importer treats the file itself as the delimited data — no
    header row. A literal tab or newline inside a field would corrupt row
    boundaries, so whitespace within each field is collapsed to single spaces.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for card in cards:
        term = " ".join(card.question.split())
        definition = " ".join(card.answer.split())
        writer.writerow([term, definition])
    return buf.getvalue()


def to_json(cards: Sequence[Card]) -> str:
    """Full-fidelity export — every field, including source_excerpt and
    difficulty, for re-importing into this app or archiving."""
    return json.dumps(
        [card.model_dump() for card in cards], indent=2, ensure_ascii=False
    )
