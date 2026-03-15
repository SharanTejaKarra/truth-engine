"""Text chunking utilities for the Truth Engine pipeline."""

import logging

from app import config
from app.models import DocumentChunk

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into chunks respecting sentence boundaries.

    Uses config.CHUNK_SIZE and config.CHUNK_OVERLAP as defaults.
    Splits on sentence endings first, then newlines, then spaces.
    Never splits mid-word.
    """
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    if len(text) <= size:
        return [text]

    # Split into sentences, preserving the delimiters
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_len + sent_len > size and current:
            chunks.append("".join(current).strip())
            # Compute overlap: walk backwards through current until we reach overlap chars
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                if overlap_len + len(part) > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part)
            current = overlap_parts
            current_len = overlap_len

        # If a single sentence exceeds chunk_size, split it further
        if sent_len > size:
            sub_parts = _split_long_segment(sentence, size)
            for part in sub_parts:
                if current_len + len(part) > size and current:
                    chunks.append("".join(current).strip())
                    overlap_parts = []
                    overlap_len = 0
                    for p in reversed(current):
                        if overlap_len + len(p) > overlap:
                            break
                        overlap_parts.insert(0, p)
                        overlap_len += len(p)
                    current = overlap_parts
                    current_len = overlap_len
                current.append(part)
                current_len += len(part)
        else:
            current.append(sentence)
            current_len += sent_len

    if current:
        final = "".join(current).strip()
        if final:
            chunks.append(final)

    return chunks


def chunk_documents(
    chunks: list[DocumentChunk],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Take parsed DocumentChunks that may be too large, apply chunking.

    Generates unique chunk_ids as '{source_type}_{source_file}_{index}'.
    Preserves all original metadata on sub-chunks.
    """
    size = chunk_size or config.CHUNK_SIZE
    result: list[DocumentChunk] = []
    global_idx: dict[str, int] = {}  # per source_file counter

    for doc in chunks:
        key = f"{doc.source_type}_{doc.source_file}"
        if key not in global_idx:
            global_idx[key] = 0

        if len(doc.content) <= size:
            new_chunk = doc.model_copy(
                update={"chunk_id": f"{key}_{global_idx[key]}"}
            )
            result.append(new_chunk)
            global_idx[key] += 1
        else:
            sub_texts = chunk_text(doc.content, chunk_size, chunk_overlap)
            for sub_text in sub_texts:
                new_chunk = doc.model_copy(
                    update={
                        "chunk_id": f"{key}_{global_idx[key]}",
                        "content": sub_text,
                    }
                )
                result.append(new_chunk)
                global_idx[key] += 1

    return result


# ── Helpers ────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments preserving whitespace."""
    segments: list[str] = []
    current: list[str] = []

    i = 0
    while i < len(text):
        ch = text[i]
        current.append(ch)

        # Sentence boundary: '. ' or '! ' or '? ' or newline
        if ch in ".!?" and i + 1 < len(text) and text[i + 1] == " ":
            current.append(text[i + 1])
            segments.append("".join(current))
            current = []
            i += 2
            continue
        elif ch == "\n":
            segments.append("".join(current))
            current = []
            i += 1
            continue

        i += 1

    if current:
        segments.append("".join(current))

    return segments


def _split_long_segment(text: str, max_len: int) -> list[str]:
    """Split a segment that exceeds max_len on spaces (never mid-word)."""
    words = text.split(" ")
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current_len + addition > max_len and current:
            parts.append(" ".join(current) + " ")
            current = []
            current_len = 0
        current.append(word)
        current_len += addition

    if current:
        parts.append(" ".join(current))

    return parts
