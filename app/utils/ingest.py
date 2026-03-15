"""Orchestrates ingestion of all source directories."""

import logging
from pathlib import Path

from app import config
from app.models import DocumentChunk
from app.utils.chunker import chunk_documents
from app.utils.parsers import parse_json_csv, parse_markdown, parse_pdf

logger = logging.getLogger(__name__)

# Map source directories to their source_type and parser
_SOURCE_MAP: dict[Path, tuple[str, dict[str, callable]]] = {
    config.SOURCE_A_DIR: (
        "manual",
        {".pdf": parse_pdf, ".txt": parse_pdf},
    ),
    config.SOURCE_B_DIR: (
        "support_log",
        {".json": parse_json_csv, ".csv": parse_json_csv},
    ),
    config.SOURCE_C_DIR: (
        "wiki",
        {".md": parse_markdown},
    ),
}


def ingest_all_sources() -> list[DocumentChunk]:
    """Walk data/source_a/, source_b/, source_c/ and ingest all files.

    Detects file type by extension, dispatches to the correct parser,
    chunks everything, and returns the final list of DocumentChunks.
    Malformed files are logged and skipped.
    """
    all_chunks: list[DocumentChunk] = []

    for source_dir, (source_type, parsers) in _SOURCE_MAP.items():
        if not source_dir.exists():
            logger.warning("Source directory does not exist: %s", source_dir)
            continue

        files = sorted(source_dir.iterdir())
        for file_path in files:
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            parser = parsers.get(ext)
            if parser is None:
                logger.debug(
                    "Skipping unsupported file type %s in %s",
                    ext,
                    source_dir.name,
                )
                continue

            try:
                chunks = parser(str(file_path))
                logger.info(
                    "Parsed %d chunks from %s", len(chunks), file_path.name
                )
                all_chunks.extend(chunks)
            except Exception:
                logger.exception("Failed to parse %s, skipping", file_path)

    logger.info("Total raw chunks before re-chunking: %d", len(all_chunks))
    final_chunks = chunk_documents(all_chunks)
    logger.info("Total chunks after re-chunking: %d", len(final_chunks))

    return final_chunks
