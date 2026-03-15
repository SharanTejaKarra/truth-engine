"""Centralized configuration for the Truth Engine."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_A_DIR = DATA_DIR / "source_a"  # Technical Manual (PDF)
SOURCE_B_DIR = DATA_DIR / "source_b"  # Support Logs (JSON/CSV)
SOURCE_C_DIR = DATA_DIR / "source_c"  # Legacy Wiki (Markdown)
INDEX_DIR = DATA_DIR / "indices"

# ── Source Hierarchy (higher = more trusted) ───────────────────────────
SOURCE_TRUST = {
    "manual": 3,      # Source A – Golden source
    "support_log": 2,  # Source B – Real-world but unverified
    "wiki": 1,         # Source C – Legacy, possibly deprecated
}

SOURCE_LABELS = {
    "manual": "Technical Manual (Source A)",
    "support_log": "Support Logs (Source B)",
    "wiki": "Legacy Wiki (Source C)",
}

# ── Embedding Model ───────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── ChromaDB ──────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "truth_engine"
CHROMA_PERSIST_DIR = str(INDEX_DIR / "chroma")

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens (approx chars / 4)
CHUNK_OVERLAP = 64

# ── Retrieval ─────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10      # initial retrieval count
TOP_K_RERANK = 5          # after cross-encoder re-ranking
SEMANTIC_WEIGHT = 0.6     # weight for semantic score in hybrid
BM25_WEIGHT = 0.4         # weight for BM25 score in hybrid

# ── Re-Ranker ─────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Confidence & Conflict ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4    # below this → "I don't know"
CONFLICT_SIMILARITY_THRESHOLD = 0.75  # semantic similarity above this + different answers → conflict

# ── Gemini LLM ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
