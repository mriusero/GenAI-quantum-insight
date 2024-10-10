# vector_store_manager.py
import logging
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)

class VectorStoreManager:
    def __init__(self, embeddings, debug=False):
        self.embeddings = embeddings
        self.vector_store = None
        self.debug = debug

    def create_vector_store(self, documents):
        """Create a VectorStore with ChromaDB from the given documents."""
        self.vector_store = Chroma.from_documents(documents, self.embeddings)
        logging.info(f"VectorStore created with {len(documents)} documents.")

    def get_vector_store(self):
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please load documents first.")
        return self.vector_store
