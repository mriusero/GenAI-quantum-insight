# research_assistant.py

import streamlit as st
from .utilities.cache import load_processed_pdfs, save_processed_pdfs
from .utilities.batch_processing import create_batches, handle_document_loading
from .processing.pdf_loader import DocumentProcessor

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
PROCESSED_PDFS_FILE = "src/app/features/research_assistant/checkpoints/processed_pdfs.pkl"

@st.cache_resource
def load_models(hg_api_key: str, debug: bool = False) -> DocumentProcessor:
    """Load and cache the DocumentProcessor instance."""
    return DocumentProcessor(debug=debug)

def run_assistance(hg_api_key: str, debug: bool = False) -> None:
    """Main function for running the research assistant."""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.title("Research Assistant")
    with col2:
        st.text("")
        document_processing = load_models(hg_api_key, debug)

        data = st.session_state.get('data', {})
        pdf_urls = data.get('pdf_link', [])[:250]

        processed_pdfs = load_processed_pdfs(PROCESSED_PDFS_FILE)

        new_pdfs = [url for url in pdf_urls if url not in processed_pdfs]
        already_processed = [url for url in pdf_urls if url in processed_pdfs]

        if already_processed:
            print("\n=== Files in Vector Store ===")
            for i, pdf in enumerate(already_processed, 1):
                print(f"{i}. {pdf}")
        else:
            print("\nVector store is empty.")

        if new_pdfs:
            print("\n=== New files to vectorize ===")
            for i, pdf in enumerate(new_pdfs, 1):
                print(f"{i}. {pdf}")
        else:
            print("\nVector store is already up to date.")

        if not new_pdfs:
            st.info("Vector store is already up to date.")
        else:
            if st.button("Vectorize new documents"):
                handle_document_loading(new_pdfs, document_processing, processed_pdfs, debug)
                st.success("Documents loaded and processed successfully.")

    user_level = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])
    user_question = st.text_input("Ask a question:")
