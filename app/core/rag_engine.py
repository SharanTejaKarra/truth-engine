"""Core RAG orchestration engine with Gemini LLM integration."""

import logging
import time

import google.generativeai as genai

from app.config import (
    CONFIDENCE_THRESHOLD,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SOURCE_LABELS,
    SOURCE_TRUST,
)
from app.core.conflict import ConflictDetector
from app.core.embeddings import EmbeddingManager
from app.core.retriever import HybridRetriever
from app.db.vector_store import VectorStore
from app.models import Citation, ConflictInfo, QueryResponse, RetrievalResult
from app.utils.ingest import ingest_all_sources

logger = logging.getLogger(__name__)


class RAGEngine:
    """Central orchestrator that ties retrieval, conflict detection, and LLM generation together."""

    def __init__(self):
        logger.info("Initializing RAG Engine components...")

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(self.embedding_manager)
        self.retriever = HybridRetriever(self.vector_store)
        self.conflict_detector = ConflictDetector(self.embedding_manager)

        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

        logger.info("RAG Engine initialized successfully.")

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest(self, force_reindex: bool = False):
        """Run the ingestion pipeline: parse, chunk, embed, and store all source documents."""
        existing_count = self.vector_store.collection_count()

        if existing_count > 0 and not force_reindex:
            logger.info(
                "Collection already contains %d documents; skipping ingestion. "
                "Use force_reindex=True to rebuild.",
                existing_count,
            )
            return

        if force_reindex and existing_count > 0:
            logger.info("Force re-index requested; clearing existing collection.")
            self.vector_store.clear()

        t0 = time.perf_counter()
        chunks = ingest_all_sources()

        if not chunks:
            logger.warning("Ingestion produced zero chunks. Check your data directories.")
            return

        self.vector_store.add_documents(chunks)
        elapsed = time.perf_counter() - t0

        # Log per-source stats
        source_counts: dict[str, int] = {}
        for c in chunks:
            source_counts[c.source_type] = source_counts.get(c.source_type, 0) + 1

        stats_str = ", ".join(
            f"{SOURCE_LABELS.get(st, st)}: {count}" for st, count in sorted(source_counts.items())
        )
        logger.info(
            "Ingestion complete in %.1fs: %d total chunks (%s)",
            elapsed,
            len(chunks),
            stats_str,
        )

    # ── Index Stats ─────────────────────────────────────────────────────────

    def get_index_stats(self) -> dict:
        """Return index statistics for the UI sidebar."""
        all_docs = self.vector_store.get_all_documents()
        stats = {"total": len(all_docs)}
        for source_type in ["manual", "support_log", "wiki"]:
            stats[source_type] = sum(1 for d in all_docs if d.source_type == source_type)
        return stats

    # ── Context & Prompt Building ─────────────────────────────────────────

    def _build_context(
        self, results: list[RetrievalResult], conflicts: list[ConflictInfo]
    ) -> str:
        """Build a structured context string for the LLM prompt."""
        parts: list[str] = ["=== RETRIEVED CONTEXT ===\n"]

        for i, r in enumerate(results, 1):
            label = SOURCE_LABELS.get(r.chunk.source_type, r.chunk.source_type)
            trust_level = SOURCE_TRUST.get(r.chunk.source_type, 0)
            trust_word = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(trust_level, "UNKNOWN")

            # Include short, structured metadata (skip long text already in content)
            meta_parts = []
            for key, val in r.chunk.metadata.items():
                val_str = str(val).strip()
                if val_str and len(val_str) < 100:
                    meta_parts.append(f"{key}: {val_str}")
            meta_line = ""
            if meta_parts:
                meta_line = f"\n  Metadata: {' | '.join(meta_parts)}"

            parts.append(
                f"[Source: {label} | Trust: {trust_word} | File: {r.chunk.source_file}]\n"
                f"{r.chunk.content}{meta_line}\n---"
            )

        if conflicts:
            parts.append("\n=== DETECTED CONFLICTS ===")
            for i, conflict in enumerate(conflicts, 1):
                parts.append(f"\nConflict {i}: {conflict.topic}")
                for chunk_result in conflict.chunks:
                    src_label = SOURCE_LABELS.get(
                        chunk_result.chunk.source_type, chunk_result.chunk.source_type
                    )
                    excerpt = chunk_result.chunk.content[:200]
                    parts.append(f"- {src_label} says: {excerpt}")
                parts.append(f"- Resolution: {conflict.resolution}")
            parts.append("===")

        return "\n".join(parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        conflicts: list[ConflictInfo],
        confidence: float,
    ) -> str:
        """Build the full LLM prompt with system instructions, context, and the user question."""
        # Build the source hierarchy string dynamically from config
        sorted_sources = sorted(SOURCE_TRUST.items(), key=lambda x: x[1], reverse=True)
        hierarchy = " > ".join(SOURCE_LABELS.get(st, st) for st, _ in sorted_sources)

        conflict_note = ""
        if conflicts:
            conflict_note = (
                "\nIMPORTANT: Conflicts were detected between sources. "
                "You MUST acknowledge each conflict and explain which source was trusted and why."
            )

        low_confidence_note = ""
        if confidence < CONFIDENCE_THRESHOLD:
            low_confidence_note = (
                "\nWARNING: Confidence is below the threshold. You MUST respond with: "
                "'I don't have sufficient information to answer this question confidently. "
                "The available sources do not provide a clear answer.'"
            )

        prompt = (
            f"SYSTEM INSTRUCTIONS:\n"
            f"You are the Truth Engine, an enterprise knowledge assistant. Answer questions "
            f"using ONLY the provided context. Follow these rules strictly:\n\n"
            f"1. SOURCE HIERARCHY: {hierarchy}\n"
            f"2. If sources conflict, ALWAYS trust the higher-ranked source.\n"
            f"3. ALWAYS cite your sources using phrases like 'According to [Source Name]...'\n"
            f"4. If a conflict was detected, acknowledge it: "
            f"'Note: There is a discrepancy between sources...'\n"
            f"5. If confidence is below the threshold, respond: "
            f"'I don't have sufficient information to answer this question confidently. "
            f"The available sources do not provide a clear answer.'\n"
            f"6. Never make up information not present in the context.\n"
            f"7. Pay attention to Metadata lines — they contain structured fields like "
            f"engineer names, ticket IDs, timestamps, error codes, and resolution status. "
            f"Use these to answer questions about who, when, and what.\n"
            f"{conflict_note}"
            f"{low_confidence_note}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Provide a clear, well-cited answer."
        )
        return prompt

    # ── LLM Interaction ───────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the Gemini API and return the generated text."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return "I encountered an error processing your question. Please try again."

    # ── Citation Extraction ───────────────────────────────────────────────

    def _extract_citations(
        self, answer: str, results: list[RetrievalResult]
    ) -> list[Citation]:
        """Extract citations by matching source labels mentioned in the LLM answer back to retrieved chunks."""
        citations: list[Citation] = []
        seen: set[str] = set()  # deduplicate by chunk_id

        for result in results:
            chunk = result.chunk
            label = SOURCE_LABELS.get(chunk.source_type, chunk.source_type)

            # Check if this source is referenced in the answer
            referenced = (
                label.lower() in answer.lower()
                or chunk.source_type.lower() in answer.lower()
                or chunk.source_file.lower() in answer.lower()
            )

            if referenced and chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                excerpt = chunk.content[:300]
                section = chunk.metadata.get("section_title") or chunk.metadata.get(
                    "page_number"
                )
                citations.append(
                    Citation(
                        source_type=chunk.source_type,
                        source_file=chunk.source_file,
                        excerpt=excerpt,
                        page_or_section=str(section) if section else None,
                    )
                )

        # If the LLM used generic "According to" patterns but we couldn't match,
        # include top results as implicit citations
        if not citations and results:
            top = results[0]
            citations.append(
                Citation(
                    source_type=top.chunk.source_type,
                    source_file=top.chunk.source_file,
                    excerpt=top.chunk.content[:300],
                )
            )

        return citations

    # ── Main Query Pipeline ───────────────────────────────────────────────

    def query(self, question: str) -> QueryResponse:
        """Execute the full RAG pipeline: retrieve, detect conflicts, generate, cite."""
        t0 = time.perf_counter()

        # Step 1: Retrieve relevant chunks
        results = self.retriever.retrieve(question)

        if not results:
            return QueryResponse(
                query=question,
                answer="I could not find any relevant information to answer your question.",
                confidence=0.0,
                retrieval_metadata={"num_chunks_retrieved": 0},
            )

        # Step 2: Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(results)

        # Step 3: Resolve conflicts (apply source trust hierarchy)
        resolved_conflicts = self.conflict_detector.resolve_conflicts(conflicts)

        # Step 4: Compute confidence
        confidence = self.conflict_detector.compute_confidence(results, resolved_conflicts)

        # Step 5: Low-confidence early return
        if confidence < CONFIDENCE_THRESHOLD:
            elapsed = time.perf_counter() - t0
            logger.info("Low confidence (%.3f) for query: %.80s", confidence, question)
            return QueryResponse(
                query=question,
                answer=(
                    "I don't have sufficient information to answer this question confidently. "
                    "The available sources do not provide a clear answer."
                ),
                confidence=confidence,
                conflicts=resolved_conflicts,
                retrieval_metadata={
                    "top_scores": [r.rerank_score for r in results[:5]],
                    "retrieval_method": "hybrid_bm25_semantic",
                    "reranker_used": True,
                    "num_conflicts": len(resolved_conflicts),
                    "num_chunks_retrieved": len(results),
                    "pipeline_time_s": round(elapsed, 3),
                },
            )

        # Step 6: Build context and prompt
        context = self._build_context(results, resolved_conflicts)
        prompt = self._build_prompt(question, context, resolved_conflicts, confidence)

        # Step 7: Call LLM
        answer = self._call_llm(prompt)

        # Step 8: Extract citations
        citations = self._extract_citations(answer, results)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Query pipeline completed in %.2fs (confidence=%.3f, conflicts=%d)",
            elapsed,
            confidence,
            len(resolved_conflicts),
        )

        # Step 9: Return full response
        return QueryResponse(
            query=question,
            answer=answer,
            confidence=confidence,
            citations=citations,
            conflicts=resolved_conflicts,
            retrieval_metadata={
                "top_scores": [r.rerank_score for r in results[:5]],
                "retrieval_method": "hybrid_bm25_semantic",
                "reranker_used": True,
                "num_conflicts": len(resolved_conflicts),
                "num_chunks_retrieved": len(results),
                "pipeline_time_s": round(elapsed, 3),
            },
        )
