import streamlit as st
from ..processing.preprocessing import handle_document_loading
from ..utilities.helper import load_processed_pdfs, display_files


def store_management(debug, arxiv, document_processor):
    """ Manages the store process: fetches new PDFs, displays them, and handles vectorization if necessary."""

    col1, _ = st.columns([3, 3])        # Search and update ARXIV database
    with col1:
        arxiv.search_and_update(
            db_name="./database/arxiv_data.db",
            query="all:quantum",
            max_results=300,
            total_results_limit=300
        )

    col1, col2 = st.columns([3, 2])     # Handle PDF documents
    with col1:
        pdf_urls = st.session_state.get('data', {}).get('pdf_link', [])[:300]
        processed_pdfs = load_processed_pdfs()

        new_pdfs = [url for url in pdf_urls if url not in processed_pdfs]
        already_processed = [url for url in pdf_urls if url in processed_pdfs]

        display_files(new_pdfs, already_processed)

    with col2:              # Vectorize documents
        if new_pdfs:
            if st.button("Vectorize Documents"):
                print("\n === Document splitting & vectorization ===\n")
                handle_document_loading(new_pdfs, document_processor, processed_pdfs, debug)
        else:
            st.info("Vector store already up to date")