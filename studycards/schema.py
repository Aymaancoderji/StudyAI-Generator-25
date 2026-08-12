"""The card data model.

This is also the JSON schema the model is constrained to via structured
outputs — see `generate.py`. Every field here becomes a required property in
the request, so add fields deliberately.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CardType = Literal["definition", "concept", "application", "compare", "cloze"]
Difficulty = Literal["easy", "medium", "hard"]


class Card(BaseModel):
    """One flashcard, grounded in a specific excerpt of the source material."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="The front of the card.")
    answer: str = Field(..., min_length=1, description="The back of the card.")
    card_type: CardType
    difficulty: Difficulty
    topic: str = Field(
        ..., min_length=1, description="Short topic label for grouping, e.g. 'Mitochondria'."
    )
    source_excerpt: str = Field(
        ...,
        min_length=1,
        description=(
            "A short verbatim quote copied from the source text that supports "
            "this card. Used to verify the card isn't hallucinated."
        ),
    )


class CardBatch(BaseModel):
    """Wrapper model — structured outputs needs a single top-level schema,
    and the model naturally produces several cards per source excerpt."""

    model_config = ConfigDict(extra="forbid")

    cards: list[Card] = Field(default_factory=list)
