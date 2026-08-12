"""The deck lives as a Pandas DataFrame from the moment cards come back.

This module covers the DataFrame <-> Card conversion, near-duplicate
detection (a real problem when overlapping chunks each surface the same
concept), quality filtering, coverage stats, and CSV/Parquet persistence.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .generate import DeckGenerationSummary
from .schema import Card

CARD_COLUMNS = [
    "id",
    "question",
    "answer",
    "card_type",
    "difficulty",
    "topic",
    "source_excerpt",
    "chunk_index",
]

DEFAULT_DUPLICATE_THRESHOLD = 0.85
DEFAULT_MIN_ANSWER_CHARS = 15

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Card <-> DataFrame
# --------------------------------------------------------------------------


def cards_to_dataframe(
    cards: Sequence[Card],
    chunk_indices: Sequence[int | None] | None = None,
) -> pd.DataFrame:
    """Build the deck DataFrame. `chunk_indices[i]` is the source chunk for
    `cards[i]`, when known — it lets a dropped-duplicate report point back to
    which page range a card came from."""
    if chunk_indices is not None and len(chunk_indices) != len(cards):
        raise ValueError(
            f"chunk_indices has {len(chunk_indices)} entries but there are "
            f"{len(cards)} cards"
        )
    if chunk_indices is None:
        chunk_indices = [None] * len(cards)

    rows = [
        {
            "id": i,
            "question": card.question,
            "answer": card.answer,
            "card_type": card.card_type,
            "difficulty": card.difficulty,
            "topic": card.topic,
            "source_excerpt": card.source_excerpt,
            "chunk_index": chunk_idx,
        }
        for i, (card, chunk_idx) in enumerate(zip(cards, chunk_indices))
    ]
    return pd.DataFrame(rows, columns=CARD_COLUMNS)


def dataframe_to_cards(df: pd.DataFrame) -> list[Card]:
    """Round-trip a deck DataFrame back into validated Card objects.

    Re-validates through Pydantic on the way out — a hand-edited CSV (e.g.
    from the Streamlit review table) gets the same guarantees a freshly
    generated card does.
    """
    return [
        Card(
            question=row.question,
            answer=row.answer,
            card_type=row.card_type,
            difficulty=row.difficulty,
            topic=row.topic,
            source_excerpt=row.source_excerpt,
        )
        for row in df.itertuples()
    ]


# --------------------------------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def find_near_duplicates(
    df: pd.DataFrame, *, threshold: float = DEFAULT_DUPLICATE_THRESHOLD
) -> list[set[int]]:
    """Group DataFrame index values whose questions are near-duplicates.

    Comparisons are scoped to cards sharing the same topic (case-insensitive)
    — this keeps the comparison count roughly linear instead of quadratic
    over the whole deck, and two cards on genuinely different topics that
    happen to share phrasing are not duplicates worth merging.
    """
    if df.empty:
        return []

    normalized = df["question"].map(_normalize_text)
    groups: list[set[int]] = []
    assigned: dict[int, int] = {}

    for _, indices in df.groupby(df["topic"].str.lower()).groups.items():
        indices = list(indices)
        for a in range(len(indices)):
            i = indices[a]
            for b in range(a + 1, len(indices)):
                j = indices[b]
                ratio = difflib.SequenceMatcher(
                    None, normalized[i], normalized[j]
                ).ratio()
                if ratio >= threshold:
                    _union(groups, assigned, i, j)

    return [g for g in groups if g]


def _union(groups: list[set[int]], assigned: dict[int, int], i: int, j: int) -> None:
    gi, gj = assigned.get(i), assigned.get(j)
    if gi is None and gj is None:
        groups.append({i, j})
        idx = len(groups) - 1
        assigned[i] = assigned[j] = idx
    elif gi is not None and gj is None:
        groups[gi].add(j)
        assigned[j] = gi
    elif gi is None and gj is not None:
        groups[gj].add(i)
        assigned[i] = gj
    elif gi != gj:
        groups[gi] |= groups[gj]
        for k in groups[gj]:
            assigned[k] = gi
        groups[gj].clear()


@dataclass(frozen=True)
class DedupeResult:
    deck: pd.DataFrame
    dropped: pd.DataFrame  # same columns as `deck`, plus `duplicate_of`


def deduplicate(
    df: pd.DataFrame, *, threshold: float = DEFAULT_DUPLICATE_THRESHOLD
) -> DedupeResult:
    """Collapse near-duplicate cards, keeping the first occurrence of each group."""
    groups = find_near_duplicates(df, threshold=threshold)

    drop_indices: set[int] = set()
    duplicate_of: dict[int, int] = {}
    for group in groups:
        keeper = min(group)
        for idx in group:
            if idx != keeper:
                drop_indices.add(idx)
                duplicate_of[idx] = int(df.loc[keeper, "id"])

    kept = df.drop(index=drop_indices).reset_index(drop=True)

    ordered = sorted(drop_indices)
    dropped = df.loc[ordered].copy()
    dropped["duplicate_of"] = [duplicate_of[i] for i in ordered]
    dropped = dropped.reset_index(drop=True)

    return DedupeResult(deck=kept, dropped=dropped)


# --------------------------------------------------------------------------
# Quality filters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityFilterResult:
    deck: pd.DataFrame
    dropped: pd.DataFrame  # same columns as `deck`, plus `reason`


def filter_quality(
    df: pd.DataFrame, *, min_answer_chars: int = DEFAULT_MIN_ANSWER_CHARS
) -> QualityFilterResult:
    """Drop cards that are too thin to be worth reviewing.

    Two checks, applied in order so each card gets exactly one reason:
    - the answer is shorter than `min_answer_chars`
    - the answer's text is already embedded verbatim in the question, so
      answering the card requires no recall at all
    """
    if df.empty:
        empty = df.copy()
        empty["reason"] = pd.Series(dtype=object)
        return QualityFilterResult(deck=df.copy(), dropped=empty)

    reasons = pd.Series([None] * len(df), index=df.index, dtype=object)

    too_short = df["answer"].str.len() < min_answer_chars
    reasons[too_short] = "answer too short"

    normalized_q = df["question"].map(_normalize_text)
    normalized_a = df["answer"].map(_normalize_text)
    restates = pd.Series(
        [
            bool(a) and (a == q or a in q)
            for q, a in zip(normalized_q, normalized_a)
        ],
        index=df.index,
    )
    still_unflagged = reasons.isna()
    reasons[restates & still_unflagged] = "answer restates question"

    drop_mask = reasons.notna()
    kept = df[~drop_mask].reset_index(drop=True)
    dropped = df[drop_mask].copy()
    dropped["reason"] = reasons[drop_mask]
    dropped = dropped.reset_index(drop=True)

    return QualityFilterResult(deck=kept, dropped=dropped)


# --------------------------------------------------------------------------
# Coverage stats
# --------------------------------------------------------------------------


def coverage_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Per-topic card counts and difficulty mix, sorted by count descending.

    Surfaces the "14 cards on X, 1 on Y" problem right after generation, so
    a lopsided deck is visible before the student starts reviewing it.
    """
    columns = ["topic", "card_count", "easy", "medium", "hard", "card_types"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    grouped = df.groupby("topic")
    counts = grouped.size().rename("card_count")

    difficulty = grouped["difficulty"].value_counts().unstack(fill_value=0)
    for level in ("easy", "medium", "hard"):
        if level not in difficulty.columns:
            difficulty[level] = 0
    difficulty = difficulty[["easy", "medium", "hard"]]

    card_types = grouped["card_type"].apply(lambda s: sorted(s.unique())).rename(
        "card_types"
    )

    result = pd.concat([counts, difficulty, card_types], axis=1).reset_index()
    result = result.sort_values("card_count", ascending=False).reset_index(drop=True)
    return result[columns]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Deck — convenience wrapper
# --------------------------------------------------------------------------


@dataclass
class Deck:
    """A thin wrapper around the deck DataFrame for ergonomic call sites."""

    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_cards(
        cls, cards: Sequence[Card], chunk_indices: Sequence[int | None] | None = None
    ) -> "Deck":
        return cls(cards_to_dataframe(cards, chunk_indices))

    @classmethod
    def from_summary(cls, summary: DeckGenerationSummary) -> "Deck":
        cards: list[Card] = []
        chunk_indices: list[int] = []
        for result in summary.results:
            for card in result.cards:
                cards.append(card)
                chunk_indices.append(result.chunk_index)
        return cls.from_cards(cards, chunk_indices)

    @classmethod
    def read_csv(cls, path: str | Path) -> "Deck":
        return cls(load_csv(path))

    @classmethod
    def read_parquet(cls, path: str | Path) -> "Deck":
        return cls(load_parquet(path))

    @property
    def cards(self) -> list[Card]:
        return dataframe_to_cards(self.df)

    def coverage(self) -> pd.DataFrame:
        return coverage_by_topic(self.df)

    def clean(
        self,
        *,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        min_answer_chars: int = DEFAULT_MIN_ANSWER_CHARS,
    ) -> tuple["Deck", dict]:
        """Deduplicate then quality-filter, returning the cleaned deck plus a
        stats dict with both dropped-row reports for user-facing feedback."""
        dedupe_result = deduplicate(self.df, threshold=duplicate_threshold)
        quality_result = filter_quality(
            dedupe_result.deck, min_answer_chars=min_answer_chars
        )
        stats = {
            "duplicates_dropped": len(dedupe_result.dropped),
            "quality_dropped": len(quality_result.dropped),
            "dropped_duplicates": dedupe_result.dropped,
            "dropped_quality": quality_result.dropped,
        }
        return Deck(quality_result.deck), stats

    def to_csv(self, path: str | Path) -> None:
        save_csv(self.df, path)

    def to_parquet(self, path: str | Path) -> None:
        save_parquet(self.df, path)
