"""Streamlit UI tests via AppTest — runs app.py in-process, no live server
and no network calls. Generation itself needs a real API key, so these tests
cover everything reachable without one: the upload/preview path, sidebar
defaults, and the results/review/download UI (exercised by pre-seeding
`st.session_state` the way a completed `generate_deck()` run would leave it).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from studycards.deck import cards_to_dataframe, deduplicate, filter_quality
from studycards.generate import DeckGenerationSummary, GenerationError, GenerationResult
from studycards.schema import Card

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def make_card(**overrides) -> Card:
    defaults = dict(
        question="What is ATP?",
        answer="A molecule that stores chemical energy for the cell.",
        card_type="definition",
        difficulty="easy",
        topic="Mitochondria",
        source_excerpt="stores chemical energy",
    )
    defaults.update(overrides)
    return Card(**defaults)


def run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    return at


def seed_completed_generation(at: AppTest, cards: list[Card], chunk_indices: list[int]):
    """Populate session_state the way main() does after a real run finishes,
    so the results UI can be tested without calling the live API."""
    df = cards_to_dataframe(cards, chunk_indices)
    results = [
        GenerationResult(
            chunk_index=i,
            cards=[card],
            dropped_ungrounded=0,
            input_tokens=100,
            output_tokens=40,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            stop_reason="end_turn",
        )
        for i, card in zip(chunk_indices, cards)
    ]
    stats = {
        "duplicates_dropped": 0,
        "quality_dropped": 0,
        "dropped_duplicates": deduplicate(df).dropped,
        "dropped_quality": filter_quality(df).dropped,
    }
    at.session_state["deck_df"] = df
    at.session_state["clean_stats"] = stats
    at.session_state["summary"] = DeckGenerationSummary(results=results, errors=[])
    return at


class TestAppLoads:
    def test_no_exception_on_load(self):
        at = run_app()
        assert not at.exception

    def test_title_renders(self):
        at = run_app()
        assert at.title[0].value == "📚 Study Card Generator"

    def test_file_uploader_present(self):
        at = run_app()
        # The main notes uploader, plus the "resume a saved deck" CSV uploader.
        assert len(at.get("file_uploader")) == 2

    def test_footer_credit_present(self):
        at = run_app()
        assert any("Pandas" in c.value and "Streamlit" in c.value for c in at.caption)

    def test_how_it_works_panel_shown_before_upload(self):
        at = run_app()
        headings = " ".join(m.value for m in at.markdown)
        assert "Upload" in headings and "Generate" in headings


class TestNoApiKey:
    """This test environment has no ANTHROPIC_API_KEY set, which is exactly
    the state these tests want to exercise."""

    def test_warning_shown(self):
        at = run_app()
        assert any("No API key configured" in w.value for w in at.warning)

    def test_generate_button_disabled_without_a_file(self):
        at = run_app()
        assert at.button[0].disabled is True


class TestSidebarDefaults:
    def test_model_options_present(self):
        at = run_app()
        select = at.sidebar.selectbox[0]
        assert select.label == "Model"
        assert select.value == "claude-opus-5"

    def test_default_card_types(self):
        at = run_app()
        ms = at.sidebar.multiselect[0]
        assert set(ms.value) == {"definition", "concept", "application"}

    def test_slider_defaults_match_library_defaults(self):
        from studycards.chunk import DEFAULT_MAX_TOKENS
        from studycards.deck import DEFAULT_DUPLICATE_THRESHOLD, DEFAULT_MIN_ANSWER_CHARS

        at = run_app()
        sliders = {s.label: s.value for s in at.sidebar.slider}
        assert sliders["Max tokens per chunk"] == DEFAULT_MAX_TOKENS
        assert sliders["Duplicate similarity threshold"] == DEFAULT_DUPLICATE_THRESHOLD
        assert sliders["Minimum answer length (characters)"] == DEFAULT_MIN_ANSWER_CHARS


class TestFileUpload:
    def test_markdown_upload_shows_doc_stats(self, notes_md):
        at = run_app()
        fu = at.get("file_uploader")[0]
        fu.upload(notes_md.name, notes_md.read_bytes(), "text/markdown")
        at.run(timeout=30)

        assert not at.exception
        assert len(at.success) == 1
        assert notes_md.name in at.success[0].value

    def test_generate_still_disabled_without_key_even_with_file(self, notes_md):
        at = run_app()
        fu = at.get("file_uploader")[0]
        fu.upload(notes_md.name, notes_md.read_bytes(), "text/markdown")
        at.run(timeout=30)

        assert at.button[0].disabled is True

    # No test for an unsupported extension via the widget: st.file_uploader's
    # own `type=[...]` restriction rejects it before our code ever runs
    # (raises StreamlitAPIException at the widget level). The UnsupportedFileError
    # branch in app.py is defensive — it matches ingest.load_bytes()'s public
    # contract for callers of the library outside this UI — and is covered
    # directly in tests/test_ingest.py.

    def test_empty_document_shows_error(self):
        at = run_app()
        fu = at.get("file_uploader")[0]
        fu.upload("empty.md", b"   \n\n  ", "text/markdown")
        at.run(timeout=30)

        assert not at.exception
        assert any("no extractable text" in e.value for e in at.error)

    def test_deselecting_all_card_types_shows_hint_and_disables_button(self, notes_md):
        at = run_app()
        fu = at.get("file_uploader")[0]
        fu.upload(notes_md.name, notes_md.read_bytes(), "text/markdown")
        at.run(timeout=30)

        at.sidebar.multiselect[0].set_value([])
        at.run(timeout=30)

        assert at.button[0].disabled is True
        assert any("Select at least one card type" in i.value for i in at.info)


class TestResultsUI:
    """Exercised by pre-seeding session_state — see seed_completed_generation."""

    def test_metrics_render(self):
        at = run_app()
        seed_completed_generation(at, [make_card()], [0])
        at.run(timeout=30)

        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Cards in deck"] == "1"
        assert metrics["Input tokens"] == "100"

    def test_coverage_table_and_review_table_both_render(self):
        at = run_app()
        seed_completed_generation(
            at,
            [make_card(topic="A"), make_card(topic="A"), make_card(topic="B")],
            [0, 0, 1],
        )
        at.run(timeout=30)

        assert not at.exception
        # Both the read-only coverage table and the editable review table are
        # `st.dataframe`-family elements in the test tree.
        assert len(at.dataframe) == 2

    def test_download_buttons_present_for_a_valid_deck(self):
        at = run_app()
        seed_completed_generation(at, [make_card()], [0])
        at.run(timeout=30)

        labels = {b.label for b in at.download_button}
        assert labels == {
            "📇 Anki (.csv)",
            "🎓 Quizlet (.tsv)",
            "🗄️ JSON",
            "💾 Save deck (with progress)",
        }

    def test_chunk_errors_shown_in_expander(self):
        at = run_app()
        seed_completed_generation(at, [make_card()], [0])
        at.session_state["summary"].errors.append(
            GenerationError("chunk 3 (p. 4): generation refused (category=cyber)", chunk_index=3)
        )
        at.run(timeout=30)

        assert not at.exception
        assert any("1 chunk(s) failed" in e.label for e in at.expander)

    def test_dropped_cards_expander_shown_when_something_was_dropped(self):
        at = run_app()
        seed_completed_generation(at, [make_card()], [0])
        at.session_state["clean_stats"]["duplicates_dropped"] = 1
        at.run(timeout=30)

        assert not at.exception
        assert any("dropped and why" in e.label for e in at.expander)

    def test_no_results_section_before_any_generation(self):
        at = run_app()
        assert len(at.metric) == 0
        assert len(at.download_button) == 0


class TestStudyMode:
    def test_study_tab_shows_progress_metrics(self):
        at = run_app()
        seed_completed_generation(at, [make_card(), make_card(topic="B")], [0, 0])
        at.run(timeout=30)

        assert not at.exception
        labels = {m.label for m in at.metric}
        assert {"Total cards", "New", "Due today", "Mastered"} <= labels

    def test_start_review_reveals_first_card_question(self):
        at = run_app()
        seed_completed_generation(at, [make_card()], [0])
        at.run(timeout=30)

        start_button = next(b for b in at.button if "Start review session" in b.label)
        start_button.click().run(timeout=30)

        assert not at.exception
        assert any(make_card().question in m.value for m in at.markdown)

    def test_rating_a_card_advances_and_updates_schedule(self):
        at = run_app()
        seed_completed_generation(at, [make_card()], [0])
        at.run(timeout=30)

        next(b for b in at.button if "Start review session" in b.label).click().run(
            timeout=30
        )
        next(b for b in at.button if "Show answer" in b.label).click().run(timeout=30)
        next(b for b in at.button if "Good" in b.label).click().run(timeout=30)

        assert not at.exception
        assert any("Session complete" in s.value for s in at.success)
        assert at.session_state["deck_df"]["repetitions"].iloc[0] == 1
