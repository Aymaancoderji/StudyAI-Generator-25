"""SM-2 spaced repetition scheduling.

Implements the SuperMemo SM-2 algorithm (the same one Anki's original
scheduler was based on): every review of a card produces a new ease factor,
repetition count, and interval, which together determine the next due date.

This module is pure scheduling logic — no Streamlit, no I/O. `deck.py`
attaches the state columns this module works with to a deck DataFrame, and
`app.py` wires the rating buttons to `schedule()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

Rating = Literal["again", "hard", "good", "easy"]

# SM-2 grades reviews on a 0-5 scale; these four Anki-style buttons map onto
# the two ends and two middle points of that scale. "again" (0) always resets
# the card regardless of its current ease/interval, per the original algorithm.
_RATING_TO_QUALITY: dict[Rating, int] = {
    "again": 0,
    "hard": 3,
    "good": 4,
    "easy": 5,
}

DEFAULT_EASE_FACTOR = 2.5
MIN_EASE_FACTOR = 1.3


@dataclass(frozen=True)
class ReviewState:
    """A card's spaced-repetition state as of its last review."""

    repetitions: int = 0
    ease_factor: float = DEFAULT_EASE_FACTOR
    interval_days: int = 0
    due_date: date | None = None
    last_reviewed: date | None = None

    def is_due(self, as_of: date) -> bool:
        """A card with no due date yet (never reviewed) is always due."""
        return self.due_date is None or self.due_date <= as_of


def schedule(
    state: ReviewState, rating: Rating, *, today: date | None = None
) -> ReviewState:
    """Apply one SM-2 review and return the resulting state.

    - "again" (quality 0): resets repetitions to 0 and the interval to 1 day,
      but leaves the ease factor untouched (a lapse doesn't in itself lower
      how "easy" the card is graded to be) since the ease factor already
      records long-run difficulty.
    - Otherwise: repetitions increments, and interval grows by the classic
      SM-2 schedule (1 day, then 6 days, then `interval * ease_factor`).
    - Ease factor is updated every review via the standard SM-2 formula and
      floored at 1.3 so a hard card never gets reviewed *less* often than
      once every `interval` days shrinks it toward.
    """
    today = today or date.today()
    quality = _RATING_TO_QUALITY[rating]

    ease_factor = state.ease_factor + (
        0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )
    ease_factor = max(MIN_EASE_FACTOR, ease_factor)

    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        repetitions = state.repetitions + 1
        if repetitions == 1:
            interval_days = 1
        elif repetitions == 2:
            interval_days = 6
        else:
            interval_days = round(state.interval_days * ease_factor)
        interval_days = max(1, interval_days)

    return ReviewState(
        repetitions=repetitions,
        ease_factor=ease_factor,
        interval_days=interval_days,
        due_date=today + timedelta(days=interval_days),
        last_reviewed=today,
    )
