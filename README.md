# Study Card Generator

Turn lecture notes (PDF, TXT, or Markdown) into a reviewable flashcard deck,
grounded in the source text and exportable to Anki, Quizlet, or JSON.

Built with [Pandas](https://pandas.pydata.org/) and
[Streamlit](https://streamlit.io/). Card generation calls the
[Anthropic API](https://platform.claude.com/) under the hood — see
[Configuration](#configuration) for setup.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# paste your Anthropic API key into .env — get one at
# https://platform.claude.com/settings/keys

.venv/bin/streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), upload one
or more PDF/TXT/Markdown files, and click **Generate study cards**.

## How it works

```
file → ingest → chunk → [Claude, structured outputs] → cards → Pandas deck → clean → export
```

1. **`ingest.py`** — reads PDF (via `pypdf`, text layer only), TXT, or MD.
   Strips repeated running headers/footers (e.g. "Page 3", a chapter title on
   every slide) by fingerprinting lines that recur at the same page edge
   across most pages. Tracks which page every chunk of text came from.

2. **`chunk.py`** — splits the document into token-bounded pieces on
   paragraph and heading boundaries, never mid-sentence. Handles both
   Markdown-style (blank-line-separated) and PDF-style (one-newline-per-line)
   text. Token counting is pluggable: a fast offline estimate by default, or
   the exact Messages API `count_tokens` endpoint when you want precision.

3. **`generate.py`** — calls `client.messages.parse(output_format=CardBatch)`
   per chunk. Structured outputs guarantee a parseable response — no
   retry-on-bad-JSON loop. Two prompt-cache breakpoints (system prompt +
   chunk text) mean a second pass over the same chunk re-reads from cache
   instead of paying full price. Every card's `source_excerpt` is checked
   against the source chunk and dropped if it doesn't actually appear there
   — a cheap, deterministic defense against hallucinated grounding.
   `generate_deck()` runs every chunk and keeps going past a refusal or a
   truncated response, so one bad chunk in a 40-page document doesn't kill
   the run.

4. **`deck.py`** — the deck lives as a Pandas DataFrame. Near-duplicate
   cards (common when two overlapping chunks both cover the same concept)
   are merged via a topic-scoped fuzzy match; low-quality cards (answer too
   short, or the answer is just embedded in the question) are dropped.
   `coverage_by_topic()` surfaces the "14 cards on X, 1 on Y" problem before
   you start reviewing.

5. **`export.py`** — serializes the cleaned deck to Anki-importable CSV,
   Quizlet's tab-separated format, or full-fidelity JSON.

6. **`app.py`** — the Streamlit UI wires all of the above together: upload
   (one file or a batch), configure generation and cleanup settings in the
   sidebar, watch live progress, review/edit every card in an editable
   table, download. A batch upload is chunked file-by-file (so page ranges
   never bleed across documents) and generated as one run into a single
   deck; each card's `document` column keeps track of which file it came
   from, and duplicate detection runs across the whole batch, not per-file.

7. **`srs.py`** — SM-2 spaced-repetition scheduling. The Study tab in the
   app reviews due cards one at a time (front, then back, then an
   Again/Hard/Good/Easy rating) and reschedules each card's next-due date
   accordingly. Progress lives in the deck itself, so downloading with
   **💾 Save deck (with progress)** and re-uploading it later via **📥 Resume
   a saved deck** picks the review schedule back up where it left off.

## Project layout

```
studycards/
├── config.py     Loads .env, constructs the shared Anthropic client
├── ingest.py      PDF/TXT/MD → normalized text with page attribution
├── chunk.py       Token-bounded, semantic-boundary chunking (single or multi-doc)
├── schema.py      Card / CardBatch — the structured-output schema
├── generate.py    Card generation via client.messages.parse
├── deck.py        Card <-> DataFrame, dedup, quality filters, coverage, SRS state
├── srs.py         SM-2 spaced-repetition scheduling
└── export.py      Anki CSV / Quizlet TSV / JSON

app.py             Streamlit UI
tests/             136 tests — unit, wire-level (mocked HTTP), and UI (AppTest)
data/samples/      A sample document for manual testing
```

## Configuration

All settings live in `.env` (copy `.env.example` to start):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Never commit this — `.env` is gitignored. |
| `STUDYCARDS_MODEL` | `claude-opus-5` | Model used for generation. |
| `STUDYCARDS_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max` — trades cost for card quality. |

Both can also be changed per-run from the Streamlit sidebar without touching
`.env`.

## Running the tests

```bash
.venv/bin/python -m pytest
```

136 tests, all offline — no API key required, no network calls. Three tiers:

- **Unit tests** (`test_ingest.py`, `test_chunk.py`, `test_deck.py`,
  `test_export.py`) — pure logic, fast.
- **Wire-level tests** (`test_generate.py`, `test_generate_wire.py`) — the
  latter runs a *real* `anthropic.Anthropic()` client through
  `httpx.MockTransport`, so request serialization (cache breakpoints, the
  structured-output schema) is verified against the actual SDK, not a stand-in.
- **UI tests** (`test_app.py`) — runs the real `app.py` through Streamlit's
  `AppTest` harness. The generation path itself needs a live key, so those
  tests seed `st.session_state` the way a completed run would leave it and
  exercise the results/review/download UI from there.

## Cost notes

- **Prompt caching**: the system prompt and each chunk's text are cached.
  Re-running generation over the same document (e.g. after tweaking card
  types) reads from cache instead of paying full input price.
- **`effort`**: `low`/`medium` are noticeably cheaper and often sufficient
  for straightforward material; save `xhigh`/`max` for dense or technical
  source text.
- Before generating, the app shows an estimated input token count per run
  (a fast offline estimate, not exact) so you can gauge cost before spending it.

## Known limitations

- PDF ingestion is text-layer only — scanned/image PDFs raise a clear error
  rather than silently producing an empty deck. OCR is out of scope.
- Duplicate detection is scoped to cards sharing the same `topic` label. If
  the model labels the same concept differently across two chunks (e.g.
  "ATP" vs. "Adenosine Triphosphate"), the near-duplicate won't be caught.

## License

Personal/educational project — no license file included.
