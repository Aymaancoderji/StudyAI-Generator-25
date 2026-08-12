"""Turn source chunks into study cards via structured outputs.

Uses `client.messages.parse(output_format=CardBatch)` — the SDK converts the
Pydantic model to a JSON schema, strips constraints the API doesn't support
(min_length etc.), and hands back `response.parsed_output` as a validated
CardBatch. No retry-on-bad-JSON loop, no manual `json.loads`.

Every card also gets a cheap grounding check: its `source_excerpt` must
actually appear in the chunk it was generated from, or it's dropped. The
schema guarantees the field exists; it doesn't guarantee it's honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import anthropic

from .chunk import Chunk
from .config import get_client, load_settings
from .schema import Card, CardBatch

SYSTEM_PROMPT = """You are an expert study-card writer helping a student build a \
spaced-repetition deck from their own notes.

For each card:
- The question and answer must be self-contained. A student reviewing the \
card later, with no other context, must be able to understand and answer it.
- source_excerpt must be a short verbatim quote — a clause or sentence — \
copied exactly from the provided text that supports the answer. Never write \
a source_excerpt that does not appear in the text.
- Prefer cards that test understanding over cards that test recall of an \
isolated fact, except for `definition` cards, where precise recall is the \
point.
- Do not invent facts, numbers, or examples that are not in the provided \
text.
- Keep answers concise: one to three sentences, or a short list for \
enumerable facts."""

DEFAULT_CARD_TYPES: tuple[str, ...] = ("definition", "concept", "application")
DEFAULT_MAX_TOKENS = 4096


class GenerationError(RuntimeError):
    """Raised when a chunk could not be turned into cards."""

    def __init__(self, message: str, *, chunk_index: int):
        super().__init__(message)
        self.chunk_index = chunk_index


@dataclass(frozen=True)
class GenerationResult:
    chunk_index: int
    cards: list[Card]
    dropped_ungrounded: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    stop_reason: str


def generate_cards(
    chunk: Chunk,
    *,
    count: int = 8,
    card_types: Sequence[str] = DEFAULT_CARD_TYPES,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """Generate a batch of cards from one chunk.

    Two `cache_control` breakpoints are set: one on the system prompt (fixed
    across every call in a run) and one on the chunk text (often revisited —
    e.g. a follow-up pass asking for more `cloze` cards over the same
    material). Both are worth caching; see shared/prompt-caching.md.
    """
    settings = load_settings()
    client = client or get_client()
    model = model or settings.model
    effort = effort or settings.effort

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_format=CardBatch,
        output_config={"effort": effort},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _user_prompt(chunk, count, card_types),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise GenerationError(
            f"chunk {chunk.index} ({chunk.page_label}): generation refused "
            f"(category={category})",
            chunk_index=chunk.index,
        )

    batch = response.parsed_output
    if batch is None:
        raise GenerationError(
            f"chunk {chunk.index} ({chunk.page_label}): model did not return "
            f"parseable cards (stop_reason={response.stop_reason})",
            chunk_index=chunk.index,
        )

    grounded, dropped = _filter_grounded(batch.cards, chunk.text)

    usage = response.usage
    return GenerationResult(
        chunk_index=chunk.index,
        cards=grounded,
        dropped_ungrounded=dropped,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
        stop_reason=response.stop_reason,
    )


@dataclass
class DeckGenerationSummary:
    """Aggregated result of generating cards across every chunk in a document."""

    results: list[GenerationResult] = field(default_factory=list)
    errors: list[GenerationError] = field(default_factory=list)

    @property
    def cards(self) -> list[Card]:
        return [card for r in self.results for card in r.cards]

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens for r in self.results)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(r.cache_creation_tokens for r in self.results)

    @property
    def total_dropped_ungrounded(self) -> int:
        return sum(r.dropped_ungrounded for r in self.results)


def generate_deck(
    chunks: Sequence[Chunk],
    *,
    count_per_chunk: int = 8,
    card_types: Sequence[str] = DEFAULT_CARD_TYPES,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    effort: str | None = None,
    on_progress: Callable[[Chunk, DeckGenerationSummary], None] | None = None,
) -> DeckGenerationSummary:
    """Generate cards for every chunk, continuing past per-chunk failures.

    A single bad chunk — a safety refusal, a truncated response — should not
    abort an entire multi-chapter run. Its failure is collected in
    `summary.errors` and every other chunk still gets processed. Chunks are
    processed in order, which also means each one is generated right after
    its neighbor — good for cache reuse on the (identical) system prompt.
    """
    settings = load_settings()
    client = client or get_client()
    model = model or settings.model
    effort = effort or settings.effort

    summary = DeckGenerationSummary()
    for chunk in chunks:
        try:
            result = generate_cards(
                chunk,
                count=count_per_chunk,
                card_types=card_types,
                client=client,
                model=model,
                effort=effort,
            )
            summary.results.append(result)
        except GenerationError as exc:
            summary.errors.append(exc)
        if on_progress is not None:
            on_progress(chunk, summary)

    return summary


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _user_prompt(chunk: Chunk, count: int, card_types: Sequence[str]) -> str:
    types_line = ", ".join(card_types)
    return (
        f"Generate about {count} study cards from the excerpt below "
        f"({chunk.page_label}). Use a mix of these card types where the "
        f"material supports them: {types_line}. If the excerpt only "
        f"supports fewer than {count} good cards, return fewer rather than "
        "padding with weak or repetitive ones.\n\n"
        f"---\n{chunk.text}\n---"
    )


def _filter_grounded(cards: list[Card], source_text: str) -> tuple[list[Card], int]:
    """Drop cards whose source_excerpt doesn't actually appear in the chunk."""
    normalized_source = " ".join(source_text.split())
    kept: list[Card] = []
    dropped = 0
    for card in cards:
        excerpt = " ".join(card.source_excerpt.split())
        if excerpt and excerpt in normalized_source:
            kept.append(card)
        else:
            dropped += 1
    return kept, dropped
