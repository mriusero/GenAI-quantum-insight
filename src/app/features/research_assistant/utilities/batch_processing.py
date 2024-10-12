# batch_processing.py

import streamlit as st
import gc
from .cache import save_processed_pdfs
from ..processing.pdf_loader import DocumentProcessor
import chromadb

VECTOR_STORE_FILE = "src/app/features/research_assistant/checkpoints/vector_store"

def create_batches(urls: list[str], batch_size_percentage: int) -> list[list[str]]:
    """Split URLs into batches based on percentage."""
    batch_size = max(1, int(len(urls) * batch_size_percentage / 100))
    return [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

def handle_document_loading(new_pdfs: list[str], document_processing: DocumentProcessor, processed_pdfs: list[str], debug: bool = False):
    """Handle the loading of documents and batch processing."""
    batches = create_batches(new_pdfs, 25)
    progress_bar = st.progress(0)
    total_batches = len(batches)

    for i, batch in enumerate(batches):
        st.write(f"Processing batch {i + 1}/{total_batches}: {batch}")
        process_pdfs_batch(batch, document_processing, processed_pdfs, debug)
        progress_bar.progress((i + 1) / total_batches)

def process_pdfs_batch(batch: list[str], document_processing: DocumentProcessor, processed_pdfs: list[str], debug: bool = False):
    """Process a batch of PDFs and update the vector store."""
    client = chromadb.PersistentClient(path=VECTOR_STORE_FILE)
    collection = client.get_or_create_collection(name="arxiv_papers_collection")

    for pdf in batch:
        document = document_processing.load_and_split([pdf])
        for chunk in document:
            collection.add(chunk["id"], chunk["embeddings"], chunk["metadata"])

        if debug:
            document_schema = [{key: type(value).__name__ for key, value in chunk.items()} for chunk in document]
            st.write("## Document schema:", document_schema[:3])

    processed_pdfs.extend(batch)
    save_processed_pdfs(processed_pdfs)
    gc.collect()
