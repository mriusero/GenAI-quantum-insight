#skeleton.py
import os
import pickle
import streamlit as st
from tqdm import tqdm
import gc
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import pprint
import chromadb
from chromadb import Client, Settings
from .document_processing import DocumentProcessor
from .qa_system import QASystem
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_PDFS_FILE = "src/app/features/research_assistant/checkpoints/processed_pdfs.pkl"
VECTOR_STORE_FILE = "src/app/features/research_assistant/checkpoints/vector_store"

@st.cache_resource
def load_models(hg_api_key: str, debug: bool = False) -> Tuple[DocumentProcessor, QASystem]:
    """Load and cache models."""
    document_processing = DocumentProcessor(debug=debug)
    qa_system = QASystem(api_key=hg_api_key, debug=debug, model_name=MODEL_NAME, embedding_model=EMBEDDING_MODEL)
    # embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return document_processing, qa_system

def load_processed_pdfs() -> List[str]:
    """Load processed PDFs from a .pkl file."""
    if os.path.exists(PROCESSED_PDFS_FILE):
        try:
            with open(PROCESSED_PDFS_FILE, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            st.error(f"Error loading processed PDFs: {e}")
    return []

def save_processed_pdfs(processed_pdfs: List[str]) -> None:
    """Save the list of processed PDFs to a .pkl file."""
    try:
        with open(PROCESSED_PDFS_FILE, "wb") as f:
            pickle.dump(processed_pdfs, f)
        st.write("Batch saved successfully.")
    except Exception as e:
        st.error(f"Error saving batch: {e}")

def create_batches(urls: List[str], batch_size_percentage: int) -> List[List[str]]:
    """Split URLs into batches of a given percentage size."""
    total_count = len(urls)
    batch_size = max(1, int(total_count * batch_size_percentage / 100))
    return [urls[i:i + batch_size] for i in range(0, total_count, batch_size)]

def run_assistance(hg_api_key: str, debug: bool = False) -> None:
    """Main logic for the research assistant app."""
    st.title("Research Assistant")
    document_processing, qa_system = load_models(hg_api_key, debug)

    data = st.session_state.get('data', {})
    pdf_urls = data.get('pdf_link', [])[:250]
    print(f"pdf files in vector store:\n___________\n{pdf_urls}\n")

    processed_pdfs = load_processed_pdfs()
    new_pdfs = [url for url in pdf_urls if url not in processed_pdfs]
    print(f"new_pdfs:\n___________\n{new_pdfs}\n")

    if not new_pdfs:
        st.info("Vector store already up to date")

    if st.button("Load Documents"):
        handle_document_loading(new_pdfs, document_processing, processed_pdfs, debug)

    # User expertise and question input
    user_level = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])
    user_question = st.text_input("Ask a question:")
 #   if st.button("Submit"):
 #       handle_question_submission(user_question, qa_system, user_level)

def handle_document_loading(new_pdfs: List[str], document_processing: DocumentProcessor, processed_pdfs: List[str], debug: bool = False):
    """Handle the process of loading and processing new documents."""
    batch_size_percentage = 25
    batches = create_batches(new_pdfs, batch_size_percentage)
    progress_bar = st.progress(0)
    total_batches = len(batches)

    for i, batch in enumerate(batches):
        st.write(f"Processing batch {i + 1} / {total_batches}: {batch}")
        process_pdfs_batch(batch, document_processing, processed_pdfs, debug)
        progress_bar.progress((i + 1) / total_batches)


def process_pdfs_batch(batch: List[str], document_processing: DocumentProcessor, processed_pdfs: List[str], debug: bool = False):
    """Process a batch of PDFs and update processed state."""
    client = chromadb.PersistentClient(path=VECTOR_STORE_FILE)
    collection = client.get_or_create_collection(name="arxiv_papers_collection")

    for pdf in batch:
        document = document_processing.load_and_split([pdf])
        for chunk in document:
            collection.add(chunk["id"], chunk["embeddings"], chunk["metadata"])

        if debug:
            document_processed = document
            document_schema = [
                {key: type(value).__name__ for key, value in chunk.items()}
                for chunk in document_processed
            ]
            st.write("## Document schema:")
            st.json(document_schema)
            st.write("## Document chunks: (3 firsts)")
            st.json(document[:3])

    processed_pdfs.extend(batch)
    save_processed_pdfs(processed_pdfs)
    st.write(f"#### Total collection count (vector store): `{collection.count()}`")

    gc.collect()

#def handle_question_submission(user_question: str, qa_system: QASystem, user_level: str):
#    """Handle user question submission."""
#   try:
#        response = qa_system.ask_question(user_question, vector_store, user_level=user_level)
#        st.markdown(f"### Answer:\n{response}")
#    except ValueError as e:
#        st.error(str(e))
