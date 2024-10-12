# embeddings.py

from sentence_transformers import SentenceTransformer

class EmbeddingsProcessor:
    """Compute embeddings using SentenceTransformer."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute_embeddings(self, text: str):
        """Compute the embedding for a given chunk of text."""
        return self.model.encode(text)
