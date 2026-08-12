"""Shared fixtures. No test in this suite makes a live API call."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

BIOLOGY_NOTES = textwrap.dedent(
    """\
    # Cell Biology

    The cell is the basic structural unit of all living organisms. Robert Hooke
    first observed cells in cork tissue in 1665 using an early compound
    microscope.

    ## Mitochondria

    Mitochondria are membrane-bound organelles that generate most of the
    chemical energy needed to power a cell's biochemical reactions. They are
    often called the powerhouse of the cell.

    Chemical energy produced by the mitochondria is stored in a small molecule
    called adenosine triphosphate, or ATP.

    ## Photosynthesis

    Photosynthesis is the process by which plants convert light energy into
    chemical energy. It occurs in the chloroplasts and produces glucose and
    oxygen from carbon dioxide and water.
    """
)


@pytest.fixture
def notes_md(tmp_path: Path) -> Path:
    path = tmp_path / "biology.md"
    path.write_text(BIOLOGY_NOTES, encoding="utf-8")
    return path


@pytest.fixture
def notes_txt(tmp_path: Path) -> Path:
    path = tmp_path / "biology.txt"
    path.write_text(BIOLOGY_NOTES, encoding="utf-8")
    return path


@pytest.fixture
def multipage_pdf(tmp_path: Path) -> Path:
    """A 4-page PDF where every page carries the same header and a page footer.

    Built with pypdf alone so the test suite needs no extra dependency.
    """
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    bodies = [
        "The cell is the basic structural unit of all living organisms.",
        "Mitochondria generate most of the chemical energy in a cell.",
        "Chemical energy is stored in a molecule called ATP.",
        "Photosynthesis converts light energy into chemical energy.",
    ]

    for i, body in enumerate(bodies, start=1):
        page = writer.add_blank_page(width=612, height=792)
        # A font resource is required, or extract_text() returns "".
        page[NameObject("/Resources")] = _helvetica_resources(writer)
        content = _text_stream(
            [
                (72, 740, "BIOL 101 Lecture Notes"),  # running header
                (72, 700, body),
                (72, 60, f"Page {i}"),  # running footer
            ]
        )
        stream = DecodedStreamObject()
        stream.set_data(content.encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    path = tmp_path / "lecture.pdf"
    with path.open("wb") as fh:
        writer.write(fh)
    return path


def _helvetica_resources(writer):
    from pypdf.generic import DictionaryObject, NameObject

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")

    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)

    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    return resources


def _text_stream(items: list[tuple[int, int, str]]) -> str:
    parts = ["BT", "/F1 12 Tf"]
    for x, y, text in items:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        parts.append(f"1 0 0 1 {x} {y} Tm ({escaped}) Tj")
    parts.append("ET")
    return "\n".join(parts)
