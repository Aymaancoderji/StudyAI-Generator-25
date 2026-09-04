"""AI study card generator — turn lecture notes into reviewable flashcards."""

from .chunk import Chunk, api_token_counter, chunk_document, chunk_documents, estimate_tokens
from .deck import (
    Deck,
    DedupeResult,
    QualityFilterResult,
    cards_to_dataframe,
    coverage_by_topic,
    dataframe_to_cards,
    deduplicate,
    filter_quality,
    find_near_duplicates,
    load_csv,
    load_parquet,
    save_csv,
    save_parquet,
)
from .export import to_anki_csv, to_json, to_quizlet_tsv
from .generate import (
    DeckGenerationSummary,
    GenerationError,
    GenerationResult,
    generate_cards,
    generate_deck,
)
from .ingest import Document, Page, load, load_bytes
from .schema import Card, CardBatch

__version__ = "0.1.0"

__all__ = [
    "Card",
    "CardBatch",
    "Chunk",
    "Deck",
    "DeckGenerationSummary",
    "DedupeResult",
    "Document",
    "GenerationError",
    "GenerationResult",
    "Page",
    "QualityFilterResult",
    "api_token_counter",
    "cards_to_dataframe",
    "chunk_document",
    "chunk_documents",
    "coverage_by_topic",
    "dataframe_to_cards",
    "deduplicate",
    "estimate_tokens",
    "filter_quality",
    "find_near_duplicates",
    "generate_cards",
    "generate_deck",
    "load",
    "load_bytes",
    "load_csv",
    "load_parquet",
    "save_csv",
    "save_parquet",
    "to_anki_csv",
    "to_json",
    "to_quizlet_tsv",
]
