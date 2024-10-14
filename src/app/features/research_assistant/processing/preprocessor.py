import concurrent.futures
import logging
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import spacy
import streamlit as st
import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 206,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", debug: bool = False):
        """Initialize processor with chunking and embedding models."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug
        self.nlp = self.load_spacy_model()
        self.bert_tokenizer = self.load_bert_pipeline()
        self.embedding_model = SentenceTransformer(embedding_model)

    def load_spacy_model(self):
        """Load spaCy language model."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logging.error("en_core_web_sm model not found. Run 'python -m spacy download en_core_web_sm'.")
            raise

    def load_bert_pipeline(self):
        """Load BERT sentiment analysis pipeline."""
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    def load_pdf(self, url: str) -> list[Document]:
        """Load PDF content from a URL."""
        return PyPDFLoader(url).load()

    def detect_document_structure(self, text: str) -> List[str]:
        """Extract sentence-level structure from text."""
        return [sent.text for sent in self.nlp(text).sents]

    def get_document_metadata(self, url: str) -> Dict:
        """Retrieve document metadata from session state."""
        df = st.session_state.get('data', {})
        return df[df['pdf_link'] == url].to_dict('records')[0]

    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return text.split("\n\n")

    def split_by_token_limit(self, text: str, token_limit: int = 512) -> List[str]:
        """Split text into chunks by token limit."""
        tokens = text.split()
        return [' '.join(tokens[i:i + token_limit]) for i in range(0, len(tokens), token_limit)]

    def load_and_split(self, new_pdfs: List[str]) -> List[Dict]:
        """Load and split PDFs into chunks."""
        all_chunks = []
        with ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.load_pdf, url): url for url in new_pdfs}
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url)):
                try:
                    all_chunks.extend(self.process_loaded_pdf(future.result(), future_to_url[future]))
                except Exception as e:
                    logging.error(f"Error loading document from {future_to_url[future]}: {e}")
        return all_chunks

    def process_loaded_pdf(self, docs: List[Document], url: str) -> List[Dict]:
        """Process loaded PDFs and split into chunks."""
        all_chunks, full_text = [], " ".join(doc.page_content for doc in docs)
        metadata = self.get_document_metadata(url)
        logical_chunks = [para for chunk in self.detect_document_structure(full_text) for para in
                          self.split_paragraphs(chunk)]

        current_chunk, chunk_index = "", 1
        for chunk in logical_chunks:
            current_chunk, chunk_index = self.handle_chunk(current_chunk, chunk, metadata, chunk_index, all_chunks)

        if current_chunk:
            all_chunks.append(self.create_chunk(current_chunk, chunk_index, metadata))
        return all_chunks

    def handle_chunk(self, current_chunk: str, chunk: str, metadata: Dict, chunk_index: int, all_chunks: List[Dict]):
        """Process and split individual chunks."""
        token_count_current = len(self.bert_tokenizer.tokenizer.tokenize(current_chunk))
        token_count_chunk = len(self.bert_tokenizer.tokenizer.tokenize(chunk))

        if token_count_current + token_count_chunk <= self.chunk_size:
            current_chunk = f"{current_chunk} {chunk}".strip()
        else:
            if current_chunk:
                all_chunks.append(self.create_chunk(current_chunk, chunk_index, metadata))
                chunk_index += 1

            if token_count_chunk > self.chunk_size:
                split_chunks = self.split_by_token_limit(chunk, self.chunk_size)
                for sc in split_chunks:
                    all_chunks.append(self.create_chunk(sc, chunk_index, metadata))
                    chunk_index += 1
            else:
                current_chunk = chunk.strip()

        return current_chunk, chunk_index

    def create_chunk(self, chunk_text: str, chunk_index: int, metadata: Dict) -> Dict:
        """Create a chunk with embeddings and metadata."""
        return {
            "id": f"{uuid.uuid4()}_chunk_{chunk_index}",
            "metadata": {"chunk_index": chunk_index, "text": chunk_text, **metadata},
            "embeddings": self.embedding_model.encode(chunk_text)
        }