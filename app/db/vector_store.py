"""ChromaDB vector store for document chunk storage and retrieval."""

import chromadb

from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from app.models import DocumentChunk, RetrievalResult
from app.core.embeddings import EmbeddingManager

_BATCH_SIZE = 100


def _flatten_metadata(metadata: dict) -> dict:
    """Flatten metadata so all values are str, int, float, or bool (ChromaDB requirement)."""
    flat = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif value is None:
            continue
        else:
            flat[key] = str(value)
    return flat


class VectorStore:
    """ChromaDB-backed vector store with cosine similarity search."""

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[DocumentChunk]):
        """Embed and upsert chunks into ChromaDB in batches."""
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            embeddings = self.embedding_manager.embed_batch(documents)

            metadatas = []
            for c in batch:
                meta = {
                    "source_type": c.source_type,
                    "source_file": c.source_file,
                }
                meta.update(_flatten_metadata(c.metadata))
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Semantic search: embed query, query ChromaDB, return scored results."""
        query_embedding = self.embedding_manager.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()) if self.collection.count() > 0 else top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        retrieval_results = []
        for doc_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance in ChromaDB: distance in [0, 2]. Similarity = 1 - distance.
            score = max(0.0, 1.0 - distance)

            # Separate ChromaDB-level fields from chunk metadata
            source_type = metadata.pop("source_type", "")
            source_file = metadata.pop("source_file", "")

            chunk = DocumentChunk(
                chunk_id=doc_id,
                content=document,
                source_type=source_type,
                source_file=source_file,
                metadata=metadata,
            )
            retrieval_results.append(
                RetrievalResult(chunk=chunk, semantic_score=score)
            )

        return retrieval_results

    def get_all_documents(self) -> list[DocumentChunk]:
        """Return all stored chunks (needed by BM25 indexer)."""
        data = self.collection.get(include=["documents", "metadatas"])

        if not data["ids"]:
            return []

        chunks = []
        for doc_id, document, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            meta = dict(metadata) if metadata else {}
            source_type = meta.pop("source_type", "")
            source_file = meta.pop("source_file", "")

            chunks.append(
                DocumentChunk(
                    chunk_id=doc_id,
                    content=document,
                    source_type=source_type,
                    source_file=source_file,
                    metadata=meta,
                )
            )
        return chunks

    def clear(self):
        """Delete and recreate the collection for re-indexing."""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def collection_count(self) -> int:
        """Return number of documents in the collection."""
        return self.collection.count()
