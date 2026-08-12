from __future__ import annotations

import pytest

from studycards import chunk as chunk_mod
from studycards import ingest
from studycards.ingest import Document, Page


def words(n: int, seed: str = "cell") -> str:
    return " ".join(f"{seed}{i}" for i in range(n))


def doc_from(*page_texts: str) -> Document:
    return Document(
        source="test",
        pages=tuple(Page(number=i, text=t) for i, t in enumerate(page_texts, start=1)),
    )


class TestEstimateTokens:
    def test_scales_with_length(self):
        assert chunk_mod.estimate_tokens("x" * 350) > chunk_mod.estimate_tokens("x" * 35)

    def test_never_returns_zero(self):
        assert chunk_mod.estimate_tokens("") == 1


class TestChunking:
    def test_respects_token_ceiling(self, notes_md):
        chunks = chunk_mod.chunk_document(
            ingest.load(notes_md), max_tokens=80, min_tokens=20
        )
        assert chunks
        assert all(c.token_count <= 80 for c in chunks)

    def test_indices_are_sequential(self, notes_md):
        chunks = chunk_mod.chunk_document(
            ingest.load(notes_md), max_tokens=80, min_tokens=20
        )
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_content_is_lost(self):
        document = doc_from("Alpha beta.\n\nGamma delta.\n\nEpsilon zeta.")
        chunks = chunk_mod.chunk_document(document, max_tokens=8, min_tokens=1)
        joined = " ".join(c.text for c in chunks)
        for token in ("Alpha", "Gamma", "Epsilon", "zeta"):
            assert token in joined

    def test_small_document_stays_one_chunk(self, notes_md):
        chunks = chunk_mod.chunk_document(ingest.load(notes_md), max_tokens=100_000)
        assert len(chunks) == 1

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="max_tokens"):
            chunk_mod.chunk_document(doc_from("a"), max_tokens=10, min_tokens=99)


class TestPageAttribution:
    def test_single_page_chunk_labels_one_page(self):
        chunks = chunk_mod.chunk_document(doc_from("Alpha beta gamma."))
        assert chunks[0].start_page == chunks[0].end_page == 1
        assert chunks[0].page_label == "p. 1"

    def test_chunk_spanning_pages_reports_a_range(self):
        document = doc_from("Alpha beta.", "Gamma delta.", "Epsilon zeta.")
        chunks = chunk_mod.chunk_document(document, max_tokens=100_000)
        assert len(chunks) == 1
        assert (chunks[0].start_page, chunks[0].end_page) == (1, 3)
        assert chunks[0].page_label == "pp. 1-3"

    def test_page_range_is_monotonic_across_chunks(self):
        document = doc_from(*[f"Page body number {i} with words." for i in range(1, 7)])
        chunks = chunk_mod.chunk_document(document, max_tokens=12, min_tokens=1)
        starts = [c.start_page for c in chunks]
        assert starts == sorted(starts)


class TestOversizedBlocks:
    def test_long_paragraph_is_split_on_sentences(self):
        para = " ".join(f"Sentence number {i} explains a concept." for i in range(40))
        chunks = chunk_mod.chunk_document(doc_from(para), max_tokens=60, min_tokens=1)
        assert len(chunks) > 1
        assert all(c.token_count <= 60 for c in chunks)

    def test_split_does_not_break_mid_sentence(self):
        para = " ".join(f"Sentence number {i} explains a concept." for i in range(40))
        chunks = chunk_mod.chunk_document(doc_from(para), max_tokens=60, min_tokens=1)
        for c in chunks:
            assert c.text.strip().endswith(".")


class TestHeadings:
    def test_heading_stays_attached_to_its_body(self, notes_md):
        chunks = chunk_mod.chunk_document(
            ingest.load(notes_md), max_tokens=60, min_tokens=20
        )
        owner = next(c for c in chunks if "membrane-bound organelles" in c.text)
        assert "## Mitochondria" in owner.text

    def test_pdf_style_single_newlines_still_split(self):
        # PDF extraction gives one newline per line, never a blank line.
        text = "\n".join(
            [
                "# Chapter One",
                "The cell is the basic unit of life.",
                "# Chapter Two",
                "Mitochondria produce ATP for the cell.",
            ]
        )
        blocks = list(chunk_mod._split_blocks(doc_from(text).pages))
        assert len(blocks) == 2
        assert "Chapter One" in blocks[0].text
        assert "Chapter Two" in blocks[1].text


class TestRuntMerging:
    def test_tiny_tail_is_folded_backwards(self):
        document = doc_from(f"{words(60)}.\n\nTail.")
        chunks = chunk_mod.chunk_document(document, max_tokens=200, min_tokens=50)
        assert chunks[-1].text.endswith("Tail.")
        assert len(chunks) == 1


class TestInjectedCounter:
    def test_custom_counter_is_used(self):
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return len(text.split())

        chunks = chunk_mod.chunk_document(
            doc_from("one two three four five six seven eight"),
            max_tokens=4,
            min_tokens=1,
            token_counter=counter,
        )
        assert calls, "injected counter was never called"
        assert all(c.token_count <= 4 for c in chunks)
