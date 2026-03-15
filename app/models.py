"""Shared data models for the Truth Engine pipeline.

These Pydantic models define the interfaces between all modules.
Every component reads/writes these types so they can be developed independently.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DocumentChunk(BaseModel):
    """A single chunk produced by the ingestion pipeline."""
    chunk_id: str
    content: str
    source_type: str          # "manual" | "support_log" | "wiki"
    source_file: str          # original filename
    metadata: dict = Field(default_factory=dict)
    # metadata may contain: page_number, section_title, timestamp,
    # ticket_id, resolution_status, deprecation_flag, etc.


class RetrievalResult(BaseModel):
    """A chunk enriched with retrieval scores."""
    chunk: DocumentChunk
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0


class Citation(BaseModel):
    """A citation pointing back to a source."""
    source_type: str
    source_file: str
    excerpt: str              # the verbatim text snippet used
    page_or_section: Optional[str] = None


class ConflictInfo(BaseModel):
    """Describes a detected conflict between sources."""
    topic: str                         # what the conflict is about
    chunks: list[RetrievalResult]      # the conflicting chunks
    resolution: str                    # how it was resolved
    winning_source: str                # which source was trusted


class QueryResponse(BaseModel):
    """The final response object returned to the UI."""
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[ConflictInfo] = Field(default_factory=list)
    retrieval_metadata: dict = Field(default_factory=dict)
    # retrieval_metadata: top_k_scores, retrieval_method, reranker_used, etc.
