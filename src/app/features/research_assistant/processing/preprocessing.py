import gc
from typing import List

import chromadb
import streamlit as st

from ..processing.preprocessor import DocumentProcessor
from ..utilities import save_processed_pdfs
from ..utilities.helper import VECTOR_STORE_FILE


def create_batches(urls: List[str], batch_size_percentage: int) -> List[List[str]]:
    """Split URLs into batches based on a percentage size."""
    batch_size = max(1, len(urls) * batch_size_percentage // 100)
    return [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

def handle_document_loading(new_pdfs: List[str], document_processor: DocumentProcessor, processed_pdfs: List[str], debug: bool = False):
    """Manage loading and processing of new documents in batches."""
    batches = create_batches(new_pdfs, 25)
    progress_bar = st.progress(0)

    for i, batch in enumerate(batches, 1):
        st.write(f"Processing batch {i} / {len(batches)}: {batch}")
        process_pdfs_batch(batch, document_processor, processed_pdfs, debug)
        progress_bar.progress(i / len(batches))

def process_pdfs_batch(batch: List[str], document_processing: DocumentProcessor, processed_pdfs: List[str], debug: bool = False):
    """Process a batch of PDFs and update the vector store."""
    client = chromadb.PersistentClient(path=VECTOR_STORE_FILE)
    collection = client.get_or_create_collection(name="arxiv_papers_collection")

    for pdf in batch:
        document = document_processing.load_and_split([pdf])
        for chunk in document:
            collection.add(chunk["id"], chunk["embeddings"], chunk["metadata"])

        if debug:
            st.write("## Document schema:")
            st.json([{key: type(value).__name__ for key, value in chunk.items()} for chunk in document])
            st.write("## Document chunks: (3 firsts)")
            st.json(document[:3])

    processed_pdfs.extend(batch)
    save_processed_pdfs(processed_pdfs)
    st.write(f"#### Total collection count (vector store): `{collection.count()}`")
    gc.collect()
