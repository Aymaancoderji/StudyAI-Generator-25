"""Split a Document into token-bounded chunks on semantic boundaries.

Chunks break at headings and paragraph breaks, never mid-sentence, and each
one carries the page range it came from so cards can cite their source.

Token counting is dependency-injected. By default we use a fast local
estimate; pass `api_token_counter()` to measure exactly via the Messages API
(`count_tokens`). We never use tiktoken — that is OpenAI's tokenizer and
undercounts Claude tokens substantially.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .ingest import Document, Page

TokenCounter = Callable[[str], int]

DEFAULT_MAX_TOKENS = 1500
DEFAULT_MIN_TOKENS = 120

# Conservative chars-per-token for the local estimate. Real English prose runs
# ~4; we use a lower number so the estimate overshoots rather than undershoots
# and we stay under the ceiling.
_CHARS_PER_TOKEN = 3.5

_HEADING_RE = re.compile(
    r"""^(
        \#{1,6}\s+\S           # markdown heading
      | \d+(\.\d+)*[.)]\s+\S   # numbered heading: "3.2 Cell Division"
      | [A-Z][A-Z0-9 ,\-:/&']{3,60}$  # SHORT ALL-CAPS LINE
    )""",
    re.VERBOSE,
)

# Split on sentence-ending punctuation followed by whitespace + a capital.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start_page: int
    end_page: int
    token_count: int

    @property
    def page_label(self) -> str:
        if self.start_page == self.end_page:
            return f"p. {self.start_page}"
        return f"pp. {self.start_page}-{self.end_page}"


@dataclass(frozen=True)
class _Block:
    """A paragraph (possibly preceded by its heading) from one page."""

    text: str
    page: int


def estimate_tokens(text: str) -> int:
    """Fast offline token estimate. Deliberately errs high."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def api_token_counter(client=None, model: str | None = None) -> TokenCounter:
    """Return a TokenCounter backed by the Messages API `count_tokens` endpoint.

    Accurate but costs a round trip per call, so use it to verify final chunks
    rather than inside the packing loop.
    """
    from .config import get_client, load_settings

    client = client or get_client()
    model = model or load_settings().model

    def count(text: str) -> int:
        result = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return result.input_tokens

    return count


def chunk_document(
    document: Document,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    token_counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Pack `document` into chunks of at most `max_tokens`.

    Trailing chunks smaller than `min_tokens` are merged backwards so we don't
    send a two-sentence fragment to the model and get a shallow card out of it.
    """
    if max_tokens < min_tokens:
        raise ValueError("max_tokens must be >= min_tokens")

    count = token_counter or estimate_tokens
    blocks = list(_split_blocks(document.pages))
    blocks = _explode_oversized(blocks, max_tokens, count)

    chunks = _pack(blocks, max_tokens, count)
    chunks = _merge_runt(chunks, min_tokens, max_tokens, count)

    return [
        Chunk(
            index=i,
            text=text,
            start_page=start,
            end_page=end,
            token_count=count(text),
        )
        for i, (text, start, end) in enumerate(chunks)
    ]


# --------------------------------------------------------------------------
# Block splitting
# --------------------------------------------------------------------------


def _split_blocks(pages: Iterable[Page]) -> Iterable[_Block]:
    """Split each page into paragraphs, keeping headings attached to what follows."""
    for page in pages:
        pending_heading: str | None = None

        for para in _paragraphs(page.text):
            if _is_heading(para):
                # A heading alone is not a block — attach it to the next paragraph.
                pending_heading = (
                    f"{pending_heading}\n{para}" if pending_heading else para
                )
                continue

            text = f"{pending_heading}\n\n{para}" if pending_heading else para
            pending_heading = None
            yield _Block(text=text, page=page.number)

        if pending_heading:
            # Trailing heading with nothing after it on this page.
            yield _Block(text=pending_heading, page=page.number)


def _paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs, tolerating PDF-style line breaks.

    Markdown and plain text separate paragraphs with a blank line. PDF text
    extraction usually does not — it emits one newline per rendered line — so
    when there is no blank-line structure we group lines instead, starting a
    new group at each heading.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    groups: list[list[str]] = []
    current: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if current and _HEADING_RE.match(line):
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)

    return ["\n".join(group) for group in groups]


def _is_heading(paragraph: str) -> bool:
    """True only for a standalone heading line.

    A heading already followed by its body on the next line is not a heading
    for our purposes — it is a complete block.
    """
    lines = [ln for ln in paragraph.split("\n") if ln.strip()]
    return len(lines) == 1 and bool(_HEADING_RE.match(lines[0].strip()))


def _explode_oversized(
    blocks: list[_Block], max_tokens: int, count: TokenCounter
) -> list[_Block]:
    """Break any single block that exceeds the ceiling into smaller pieces."""
    out: list[_Block] = []
    for block in blocks:
        if count(block.text) <= max_tokens:
            out.append(block)
            continue
        for piece in _hard_split(block.text, max_tokens, count):
            out.append(_Block(text=piece, page=block.page))
    return out


def _hard_split(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split on sentence boundaries; fall back to words if that isn't enough.

    Tables, code listings and bullet dumps can carry no sentence punctuation at
    all, so a sentence-only split would silently emit an over-ceiling chunk.
    """
    pieces: list[str] = []
    for piece in _group(_SENTENCE_RE.split(text), max_tokens, count):
        if count(piece) <= max_tokens:
            pieces.append(piece)
        else:
            pieces.extend(_group(piece.split(), max_tokens, count))
    return pieces


def _group(units: list[str], max_tokens: int, count: TokenCounter) -> list[str]:
    """Greedily join `units` with spaces, staying under the ceiling."""
    out: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and count(" ".join(current + [unit])) > max_tokens:
            out.append(" ".join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        out.append(" ".join(current))
    return out


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def _pack(
    blocks: list[_Block], max_tokens: int, count: TokenCounter
) -> list[tuple[str, int, int]]:
    """Greedily fill chunks up to the token ceiling. Returns (text, start, end)."""
    chunks: list[tuple[str, int, int]] = []
    current: list[_Block] = []

    for block in blocks:
        candidate = _join(current + [block])
        if current and count(candidate) > max_tokens:
            chunks.append(_finish(current))
            current = [block]
        else:
            current.append(block)

    if current:
        chunks.append(_finish(current))
    return chunks


def _merge_runt(
    chunks: list[tuple[str, int, int]],
    min_tokens: int,
    max_tokens: int,
    count: TokenCounter,
) -> list[tuple[str, int, int]]:
    """Fold a too-small final chunk into its predecessor when it fits."""
    if len(chunks) < 2:
        return chunks

    text, start, end = chunks[-1]
    if count(text) >= min_tokens:
        return chunks

    prev_text, prev_start, _ = chunks[-2]
    merged = f"{prev_text}\n\n{text}"
    # Allow a modest overflow rather than emitting a fragment.
    if count(merged) <= int(max_tokens * 1.2):
        return chunks[:-2] + [(merged, prev_start, end)]
    return chunks


def _join(blocks: list[_Block]) -> str:
    return "\n\n".join(b.text for b in blocks)


def _finish(blocks: list[_Block]) -> tuple[str, int, int]:
    pages = [b.page for b in blocks]
    return _join(blocks), min(pages), max(pages)
