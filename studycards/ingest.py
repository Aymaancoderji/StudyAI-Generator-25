"""Turn a source file into normalized text with page attribution.

Supports PDF (text layer only — no OCR), plain text, and Markdown. Every
chunk of text keeps the page it came from so generated cards can cite it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

# A line is treated as a running header/footer if it appears in the same
# edge position on at least this fraction of pages.
_REPEAT_THRESHOLD = 0.6
_EDGE_LINES = 2  # how many lines from the top/bottom to inspect


class UnsupportedFileError(ValueError):
    """Raised for a file extension we cannot read."""


class EmptyDocumentError(ValueError):
    """Raised when a file yields no extractable text (e.g. a scanned PDF)."""


@dataclass(frozen=True)
class Page:
    """One page of source text. Non-paginated formats get a single page 1."""

    number: int
    text: str


@dataclass(frozen=True)
class Document:
    source: str
    pages: tuple[Page, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


def load(path: str | Path) -> Document:
    """Read `path` into a normalized Document. Dispatches on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(
            f"{path.name}: unsupported extension {suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    pages = _read_pdf(path) if suffix == ".pdf" else _read_plaintext(path)
    pages = tuple(p for p in pages if p.text)

    if not pages:
        raise EmptyDocumentError(
            f"{path.name}: no extractable text. If this is a scanned PDF, it has "
            "no text layer — OCR would be needed, which this project does not do."
        )

    return Document(source=path.name, pages=pages)


def load_bytes(data: bytes, filename: str) -> Document:
    """Same as `load`, for an in-memory upload (used by the Streamlit app)."""
    import io

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(
            f"{filename}: unsupported extension {suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        raw_pages = _extract_pdf_pages(PdfReader(io.BytesIO(data)))
    else:
        raw_pages = [data.decode("utf-8", errors="replace")]

    pages = tuple(p for p in _finalize(raw_pages) if p.text)
    if not pages:
        raise EmptyDocumentError(f"{filename}: no extractable text.")
    return Document(source=filename, pages=pages)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _read_pdf(path: Path) -> list[Page]:
    return _finalize(_extract_pdf_pages(PdfReader(str(path))))


def _read_plaintext(path: Path) -> list[Page]:
    return _finalize([path.read_text(encoding="utf-8", errors="replace")])


def _extract_pdf_pages(reader: PdfReader) -> list[str]:
    return [page.extract_text() or "" for page in reader.pages]


def _finalize(raw_pages: list[str]) -> list[Page]:
    """Strip running headers/footers, then normalize each page."""
    stripped = _strip_running_lines(raw_pages)
    return [
        Page(number=i, text=normalize(text)) for i, text in enumerate(stripped, start=1)
    ]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Clean up extracted text without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")  # form feed
    text = text.replace(" ", " ")  # non-breaking space

    # Rejoin words split across a line break: "photo-\nsynthesis" -> "photosynthesis"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Trim trailing whitespace on every line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse runs of blank lines down to a single paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs, but leave newlines alone.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _strip_running_lines(raw_pages: list[str]) -> list[str]:
    """Remove page headers/footers that repeat across most pages.

    PDF exports of slides and textbooks put the chapter title or page number on
    every page. Left in, those lines get chunked and turned into junk cards.
    """
    if len(raw_pages) < 3:
        return raw_pages

    counts: Counter[str] = Counter()
    for text in raw_pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        edges = lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]
        # Count each distinct edge line once per page.
        counts.update(set(_fingerprint(ln) for ln in edges))

    min_hits = max(2, int(len(raw_pages) * _REPEAT_THRESHOLD))
    repeating = {fp for fp, n in counts.items() if n >= min_hits}
    if not repeating:
        return raw_pages

    cleaned: list[str] = []
    for text in raw_pages:
        lines = text.split("\n")
        keep = [
            ln
            for i, ln in enumerate(lines)
            if not (
                _is_edge(i, len(lines)) and _fingerprint(ln.strip()) in repeating
            )
        ]
        cleaned.append("\n".join(keep))
    return cleaned


def _is_edge(index: int, total: int) -> bool:
    return index < _EDGE_LINES or index >= total - _EDGE_LINES


def _fingerprint(line: str) -> str:
    """Normalize a line so 'Page 3' and 'Page 47' compare equal."""
    return re.sub(r"\d+", "#", line.strip().lower())
