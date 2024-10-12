# pdf_loader.py

import concurrent.futures
import logging
import spacy
import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict

class DocumentProcessor:
    """Process documents, split them into chunks, and compute embeddings."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 206, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", debug: bool = False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug
        self.nlp = self.load_spacy_model()
        self.embedding_model = SentenceTransformer(embedding_model)

    def load_spacy_model(self):
        """Load the spaCy model."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logging.error("spaCy model not found.")
            raise

    def load_pdf(self, url: str) -> List[str]:
        """Load a PDF document from a URL."""
        loader = PyPDFLoader(url)
        return loader.load()

    def load_and_split(self, pdf_urls: List[str]) -> List[Dict]:
        """Load and split multiple PDFs."""
        all_chunks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.load_pdf, url): url for url in pdf_urls}
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                url = futures[future]
                try:
                    all_chunks.extend(self.process_loaded_pdf(future.result(), url))
                except Exception as e:
                    logging.error(f"Error loading document {url}: {e}")
        return all_chunks
