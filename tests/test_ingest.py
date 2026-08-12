from __future__ import annotations

import pytest

from studycards import ingest


class TestPlaintext:
    def test_markdown_loads_as_single_page(self, notes_md):
        doc = ingest.load(notes_md)
        assert doc.source == "biology.md"
        assert doc.page_count == 1
        assert doc.pages[0].number == 1
        assert "Mitochondria" in doc.text

    def test_txt_and_md_produce_identical_text(self, notes_md, notes_txt):
        assert ingest.load(notes_md).text == ingest.load(notes_txt).text

    def test_paragraph_structure_survives(self, notes_md):
        # Blank-line separation is what the chunker splits on — it must not be
        # collapsed by normalization.
        assert "\n\n" in ingest.load(notes_md).text

    def test_unsupported_extension_rejected(self, tmp_path):
        path = tmp_path / "slides.pptx"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ingest.UnsupportedFileError, match="pptx"):
            ingest.load(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_text("   \n\n  ", encoding="utf-8")
        with pytest.raises(ingest.EmptyDocumentError):
            ingest.load(path)


class TestPdf:
    def test_pages_are_numbered_from_one(self, multipage_pdf):
        doc = ingest.load(multipage_pdf)
        assert doc.page_count == 4
        assert [p.number for p in doc.pages] == [1, 2, 3, 4]

    def test_body_text_extracted(self, multipage_pdf):
        doc = ingest.load(multipage_pdf)
        assert "Mitochondria generate most of the chemical energy" in doc.text
        assert "Photosynthesis converts light energy" in doc.text

    def test_running_header_stripped(self, multipage_pdf):
        # "BIOL 101 Lecture Notes" appears on all 4 pages and is pure noise.
        assert "BIOL 101 Lecture Notes" not in ingest.load(multipage_pdf).text

    def test_page_numbers_stripped_despite_differing_digits(self, multipage_pdf):
        # "Page 1".."Page 4" differ per page, so they only collapse if the
        # fingerprint normalizes digits.
        text = ingest.load(multipage_pdf).text
        assert "Page 1" not in text
        assert "Page 4" not in text

    def test_load_bytes_matches_load(self, multipage_pdf):
        from_disk = ingest.load(multipage_pdf)
        from_memory = ingest.load_bytes(multipage_pdf.read_bytes(), "lecture.pdf")
        assert from_memory.text == from_disk.text
        assert from_memory.source == "lecture.pdf"


class TestNormalize:
    def test_dehyphenates_across_line_breaks(self):
        assert "photosynthesis" in ingest.normalize("photo-\nsynthesis occurs")

    def test_collapses_blank_line_runs(self):
        assert ingest.normalize("a\n\n\n\n\nb") == "a\n\nb"

    def test_collapses_repeated_spaces_but_keeps_newlines(self):
        assert ingest.normalize("a    b\nc") == "a b\nc"

    def test_strips_form_feed_and_nbsp(self):
        assert "\x0c" not in ingest.normalize("a\x0cb")
        assert "\xa0" not in ingest.normalize("a\xa0b")

    def test_short_documents_keep_their_edges(self):
        # Under 3 pages there is no reliable repetition signal, so nothing is
        # stripped — better to keep a header than eat real content.
        pages = ingest._strip_running_lines(["Header\nBody one", "Header\nBody two"])
        assert all("Header" in p for p in pages)
