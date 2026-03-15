"""Embedding manager using sentence-transformers."""

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class EmbeddingManager:
    """Manages text embedding using a sentence-transformers model."""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts for efficiency."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=64)
        return embeddings.tolist()
