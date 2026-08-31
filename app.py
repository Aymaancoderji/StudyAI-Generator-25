"""Study Card Generator — Streamlit UI.

Upload notes -> configure generation -> review/edit the deck -> download.
Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from studycards.chunk import DEFAULT_MAX_TOKENS, DEFAULT_MIN_TOKENS, chunk_document
from studycards.config import MissingAPIKeyError, load_settings
from studycards.deck import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_MIN_ANSWER_CHARS,
    Deck,
    coverage_by_topic,
    dataframe_to_cards,
)
from studycards.export import to_anki_csv, to_json, to_quizlet_tsv
from studycards.generate import DEFAULT_CARD_TYPES, generate_deck
from studycards.ingest import EmptyDocumentError, UnsupportedFileError, load_bytes
from studycards.schema import Card

ALL_CARD_TYPES = ("definition", "concept", "application", "compare", "cloze")
ALL_DIFFICULTIES = ("easy", "medium", "hard")
MODEL_OPTIONS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
EFFORT_OPTIONS = ("low", "medium", "high", "xhigh", "max")

st.set_page_config(
    page_title="Study Card Generator", page_icon="📚", layout="wide"
)


# --------------------------------------------------------------------------
# Styling — theme colors live in .streamlit/config.toml; this is just fonts,
# spacing, and hiding Streamlit's default chrome. No hardcoded colors here,
# so it holds up in both the light and dark theme variants.
# --------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Work+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Work Sans', sans-serif;
}
h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
}
[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
[data-testid="stCaptionContainer"] {
    font-size: 0.95rem;
}
footer {
    visibility: hidden;
}
.scg-tagline {
    color: var(--text-color);
    opacity: 0.72;
    font-size: 1.05rem;
    margin-top: -0.4rem;
    margin-bottom: 1rem;
}
.scg-howitworks h4 {
    font-family: 'Fraunces', serif;
    margin-bottom: 0.15rem;
}
.scg-howitworks p {
    opacity: 0.8;
    font-size: 0.92rem;
    margin-top: 0;
}
</style>
"""


def _inject_style() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _ingest(data: bytes, filename: str):
    return load_bytes(data, filename)


def _api_key_banner(settings) -> bool:
    """Show setup instructions if no key is configured. Returns has_key."""
    if settings.has_api_key:
        return True
    st.warning(
        "**No API key configured.** You can still upload a file and preview "
        "chunking, but generating cards needs a key.\n\n"
        "1. Copy `.env.example` to `.env`\n"
        "2. Paste your key into `.env` (it's gitignored)\n"
        "3. Restart this app\n\n"
        "Get a key at platform.claude.com/settings/keys"
    )
    return False


def _how_it_works() -> None:
    st.markdown('<div class="scg-howitworks">', unsafe_allow_html=True)
    cols = st.columns(3)
    steps = [
        ("📄", "Upload", "PDF, TXT, or Markdown lecture notes."),
        ("✨", "Generate", "Grounded, cited flashcards are written chunk by chunk."),
        ("✅", "Review & export", "Edit inline, then export to Anki, Quizlet, or JSON."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.markdown(f"<p>{desc}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _inject_style()

    st.title("📚 Study Card Generator")
    st.markdown(
        '<div class="scg-tagline">Turn lecture notes into a reviewable, '
        "grounded flashcard deck.</div>",
        unsafe_allow_html=True,
    )

    settings = load_settings()
    has_key = _api_key_banner(settings)

    with st.sidebar:
        st.header("⚙️ Generation settings")
        model = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(settings.model)
            if settings.model in MODEL_OPTIONS
            else 0,
        )
        effort = st.select_slider(
            "Effort",
            options=EFFORT_OPTIONS,
            value=settings.effort if settings.effort in EFFORT_OPTIONS else "medium",
            help="Higher effort costs more tokens but writes better cards.",
        )
        card_types = st.multiselect(
            "Card types to generate",
            options=list(ALL_CARD_TYPES),
            default=list(DEFAULT_CARD_TYPES),
        )
        count_per_chunk = st.slider("Cards per chunk", min_value=3, max_value=15, value=8)
        max_chunk_tokens = st.slider(
            "Max tokens per chunk",
            min_value=500,
            max_value=3000,
            value=DEFAULT_MAX_TOKENS,
            step=100,
        )

        st.divider()
        st.header("🧹 Cleanup settings")
        dedup_threshold = st.slider(
            "Duplicate similarity threshold",
            min_value=0.5,
            max_value=1.0,
            value=DEFAULT_DUPLICATE_THRESHOLD,
            step=0.05,
            help="Lower catches more near-duplicates, but risks merging distinct cards.",
        )
        min_answer_chars = st.slider(
            "Minimum answer length (characters)",
            min_value=5,
            max_value=50,
            value=DEFAULT_MIN_ANSWER_CHARS,
        )

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload your notes", type=["pdf", "txt", "md", "markdown"]
        )

        document = None
        if uploaded_file is not None:
            try:
                document = _ingest(uploaded_file.getvalue(), uploaded_file.name)
            except UnsupportedFileError as exc:
                st.error(str(exc))
            except EmptyDocumentError as exc:
                st.error(str(exc))

        if document is not None:
            st.success(
                f"Loaded **{document.source}** — {document.page_count} page(s), "
                f"{document.char_count:,} characters"
            )

    if document is None:
        _how_it_works()

    generate_disabled = document is None or not has_key or not card_types
    if document is not None and not card_types:
        st.info("Select at least one card type in the sidebar to generate.")

    gen_col, reset_col = st.columns([5, 1])
    generate_clicked = gen_col.button(
        "✨ Generate study cards", type="primary", disabled=generate_disabled
    )
    if reset_col.button(
        "↺ Start over", disabled="deck_df" not in st.session_state, width="stretch"
    ):
        for key in ("deck_df", "clean_stats", "summary", "card_editor"):
            st.session_state.pop(key, None)
        st.rerun()

    if generate_clicked:
        chunks = chunk_document(
            document, max_tokens=max_chunk_tokens, min_tokens=DEFAULT_MIN_TOKENS
        )
        estimated_tokens = sum(c.token_count for c in chunks)
        st.caption(
            f"Split into {len(chunks)} chunk(s), ~{estimated_tokens:,} estimated "
            "input tokens (rough offline estimate, not exact)."
        )

        with st.status("Generating cards...", expanded=True) as status_box:
            progress_bar = st.progress(0.0)
            log = st.empty()
            lines: list[str] = []

            def on_progress(chunk, summary):
                progress_bar.progress((chunk.index + 1) / len(chunks))
                lines.append(
                    f"chunk {chunk.index + 1}/{len(chunks)} ({chunk.page_label}) "
                    f"→ {len(summary.cards)} cards so far, {len(summary.errors)} error(s)"
                )
                log.code("\n".join(lines[-10:]))

            try:
                summary = generate_deck(
                    chunks,
                    count_per_chunk=count_per_chunk,
                    card_types=card_types,
                    model=model,
                    effort=effort,
                    on_progress=on_progress,
                )
            except MissingAPIKeyError as exc:
                status_box.update(label="Failed — no API key", state="error")
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001 — surface any SDK error to the UI
                status_box.update(label="Failed", state="error")
                st.error(f"Generation failed: {exc}")
                st.stop()

            status_box.update(
                label=f"Done — {len(summary.cards)} cards from {len(chunks)} chunk(s)",
                state="complete",
            )

        raw_deck = Deck.from_summary(summary)
        cleaned, stats = raw_deck.clean(
            duplicate_threshold=dedup_threshold, min_answer_chars=min_answer_chars
        )

        st.session_state["deck_df"] = cleaned.df
        st.session_state["clean_stats"] = stats
        st.session_state["summary"] = summary
        st.session_state.pop("card_editor", None)  # reset editor widget state

    if "deck_df" in st.session_state:
        _render_results()

    st.divider()
    st.caption("Built with Pandas · Streamlit")


def _render_results() -> None:
    deck_df = st.session_state["deck_df"]
    stats = st.session_state["clean_stats"]
    summary = st.session_state["summary"]

    st.subheader("📊 Generation summary")
    with st.container(border=True):
        cols = st.columns(5)
        cols[0].metric("Cards in deck", len(deck_df))
        cols[1].metric("Duplicates dropped", stats["duplicates_dropped"])
        cols[2].metric("Low-quality dropped", stats["quality_dropped"])
        cols[3].metric("Input tokens", f"{summary.total_input_tokens:,}")
        cache_pct = (
            100 * summary.total_cache_read_tokens / summary.total_input_tokens
            if summary.total_input_tokens
            else 0
        )
        cols[4].metric("Cache hit rate", f"{cache_pct:.0f}%")

    if summary.errors:
        with st.expander(f"⚠️ {len(summary.errors)} chunk(s) failed", expanded=False):
            for err in summary.errors:
                st.write(f"- {err}")

    if stats["duplicates_dropped"] or stats["quality_dropped"]:
        with st.expander("🗑️ What got dropped and why", expanded=False):
            if stats["duplicates_dropped"]:
                st.write("**Near-duplicates** (kept the first occurrence):")
                st.dataframe(
                    stats["dropped_duplicates"][["question", "duplicate_of"]],
                    hide_index=True,
                )
            if stats["quality_dropped"]:
                st.write("**Low-quality cards:**")
                st.dataframe(
                    stats["dropped_quality"][["question", "reason"]], hide_index=True
                )

    if len(deck_df):
        cov_col, dist_col = st.columns([3, 2])
        with cov_col:
            st.subheader("🗂️ Coverage by topic")
            st.dataframe(coverage_by_topic(deck_df), hide_index=True, width="stretch")
        with dist_col:
            st.subheader("🎯 Card mix")
            st.bar_chart(deck_df["card_type"].value_counts(), horizontal=True)

    st.subheader("✏️ Review & edit cards")
    st.caption(
        "Edit any cell, or delete a row with the trash icon. Downloads below "
        "reflect your edits."
    )
    search = st.text_input(
        "🔎 Preview a keyword search",
        placeholder="Search questions and answers...",
        help="Shows a quick read-only match count. Edit and download using the full table below.",
    )
    if search:
        mask = deck_df["question"].str.contains(
            search, case=False, na=False
        ) | deck_df["answer"].str.contains(search, case=False, na=False)
        st.caption(f"{int(mask.sum())} of {len(deck_df)} card(s) match \"{search}\".")

    edited_df = st.data_editor(
        deck_df,
        key="card_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "chunk_index": st.column_config.NumberColumn("Chunk", disabled=True),
            "card_type": st.column_config.SelectboxColumn(
                "Type", options=list(ALL_CARD_TYPES)
            ),
            "difficulty": st.column_config.SelectboxColumn(
                "Difficulty", options=list(ALL_DIFFICULTIES)
            ),
            "question": st.column_config.TextColumn("Question", width="large"),
            "answer": st.column_config.TextColumn("Answer", width="large"),
            "source_excerpt": st.column_config.TextColumn(
                "Source excerpt", width="medium", disabled=True
            ),
        },
    )

    _render_downloads(edited_df)


def _render_downloads(edited_df) -> None:
    try:
        cards: list[Card] = dataframe_to_cards(edited_df)
    except Exception as exc:  # noqa: BLE001 — Pydantic ValidationError, surfaced plainly
        st.error(
            "Some edited rows are invalid and can't be exported yet. Fix them "
            f"in the table above.\n\n{exc}"
        )
        return

    st.subheader("⬇️ Download")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "📇 Anki (.csv)",
        data=to_anki_csv(cards),
        file_name="study_cards_anki.csv",
        mime="text/csv",
        width="stretch",
    )
    col2.download_button(
        "🎓 Quizlet (.tsv)",
        data=to_quizlet_tsv(cards),
        file_name="study_cards_quizlet.tsv",
        mime="text/tab-separated-values",
        width="stretch",
    )
    col3.download_button(
        "🗄️ JSON",
        data=to_json(cards),
        file_name="study_cards.json",
        mime="application/json",
        width="stretch",
    )


main()
