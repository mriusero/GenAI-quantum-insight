# document_processing.py
import logging
import streamlit as st
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import spacy
import tqdm
import warnings
import uuid
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline  # Hugging Face transformers
from sentence_transformers import SentenceTransformer  # Sentence embedding models

warnings.filterwarnings("ignore", category=UserWarning, module='torch') # Suppress torch-related user warnings
logging.basicConfig(level=logging.INFO)  # Configure logging level

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 206, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", debug: bool = False):
        """Initialize the DocumentProcessor with parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug
        self.nlp = self.load_spacy_model()                           # Load spaCy model
        self.bert_tokenizer = self.load_bert_pipeline()              # Load BERT sentiment analysis
        self.embedding_model = SentenceTransformer(embedding_model)  # Load embedding model

    def load_spacy_model(self):
        """Load the spaCy model."""
        try:
            nlp = spacy.load("en_core_web_sm")
            logging.info("Model en_core_web_sm loaded successfully.")
            return nlp
        except OSError:
            logging.error("en_core_web_sm model not found. Install with 'python -m spacy download en_core_web_sm'.")
            raise

    def load_bert_pipeline(self):
        """Load the BERT pipeline for sentiment analysis."""
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")                   # Load tokenizer
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")  # Load model
        return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)           # Load sentiment pipeline

    def load_pdf(self, url: str) -> list[Document]:
        """Load the content of a PDF from a URL."""
        loader = PyPDFLoader(url)
        return loader.load()

    def detect_document_structure(self, text: str) -> List[str]:
        """Detect logical structure of text (titles, subtitles, sentences, etc.)."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def get_document_metadata(self, url: str) -> Dict:
        """Retrieve document metadata from session state into dictionary."""
        df = st.session_state.get('data', {})
        document_id = df[df['pdf_link'] == url].iloc[0]['id']
        return df[df['pdf_link'] == url].to_dict('records')[0]

    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return text.split("\n\n")

    def split_by_token_limit(self, text: str, token_limit: int = 512) -> List[str]:
        """Split text into chunks by token limit."""
        tokens = text.split()
        chunks, current_chunk = [], []

        for token in tokens:
            current_chunk.append(token)
            if len(current_chunk) >= token_limit:       # If chunk exceeds token limit
                chunks.append(' '.join(current_chunk))  # Add chunk
                current_chunk = []                      # Reset chunk

        if current_chunk:
            chunks.append(' '.join(current_chunk))      # Add remaining tokens

        return chunks

    def load_and_split(self, new_pdfs: List[str]) -> List[Dict]:
        """Load and split PDF documents into chunks with associated metadata."""
        all_chunks = []

        with ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.load_pdf, url): url for url in new_pdfs}                # Submit tasks to thread pool
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url)):  # Progress bar
                url = future_to_url[future]
                try:
                    all_chunks.extend(self.process_loaded_pdf(future.result(), url))  # Process PDF results
                except Exception as e:
                    logging.error(f"Error loading document from {url}: {e}")

        return all_chunks

    def process_loaded_pdf(self, docs: List[str], url: str) -> List[Dict]:
        """Process the loaded PDF content and split into chunks with metadata."""
        all_chunks = []
        full_text = " ".join(doc.page_content if not isinstance(doc, str) else doc for doc in docs)

        structured_chunks = self.detect_document_structure(full_text)  # Detect structure
        logical_chunks = [paragraph for chunk in structured_chunks for paragraph in self.split_paragraphs(chunk)]  # Split paragraphs

        document_metadata = self.get_document_metadata(url)     # Get metadata
        current_chunk, chunk_index = "", 1                      # Initialize variables

        for chunk in logical_chunks:
            current_chunk, chunk_index = self.handle_chunk(current_chunk, chunk, document_metadata, chunk_index, all_chunks)  # Handle chunks

        if current_chunk:
            all_chunks.append(self.create_chunk(current_chunk, chunk_index, document_metadata))  # Add last chunk if exists

        return all_chunks

    def handle_chunk(self, current_chunk: str, chunk: str, document_metadata: Dict, chunk_index: int, all_chunks: List[Dict]):
        """Handle splitting and processing of individual chunks."""
        token_count_current = len(self.bert_tokenizer.tokenizer.tokenize(current_chunk))    # Token count for current chunk
        token_count_chunk = len(self.bert_tokenizer.tokenizer.tokenize(chunk))              # Token count for new chunk

        if token_count_current + token_count_chunk <= self.chunk_size:      # If total token count is within limit
            current_chunk = f"{current_chunk} {chunk}".strip()              # Concatenate chunks
        else:
            if current_chunk:
                all_chunks.append(self.create_chunk(current_chunk, chunk_index, document_metadata))  # Add current chunk to list
                chunk_index += 1

            if token_count_chunk > self.chunk_size:                                         # If new chunk exceeds token limit
                split_chunks = self.split_by_token_limit(chunk.strip(), self.chunk_size)    # Split into smaller chunks
                for sc in split_chunks:
                    all_chunks.append(self.create_chunk(sc, chunk_index, document_metadata))  # Add split chunks
                    chunk_index += 1
            else:
                current_chunk = chunk.strip()  # Set current chunk

        return current_chunk, chunk_index

    def create_chunk(self, chunk_text: str, chunk_index: int, metadata: Dict) -> Dict:
        """Create a chunk with text, metadata & embeddings."""
        embeddings = self.embedding_model.encode(chunk_text)

        unique_id = str(uuid.uuid4())

        return {
            "id": f"{unique_id}_chunk_{chunk_index}",
            "metadata": {
                "chunk_index": chunk_index,
                "text": chunk_text,
                **metadata  # Add document metadata
            },
            "embeddings": embeddings
        }


