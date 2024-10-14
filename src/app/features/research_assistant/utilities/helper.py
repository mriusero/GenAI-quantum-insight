import os
import pickle
from typing import List

import streamlit as st

PROCESSED_PDFS_FILE = "src/app/features/research_assistant/checkpoints/processed_pdfs.pkl"
VECTOR_STORE_FILE = "src/app/features/research_assistant/checkpoints/vector_store"

def load_processed_pdfs() -> List[str]:
    """Load processed PDFs from a pickle file, handle errors gracefully."""
    pdf_links = []
    if os.path.exists(PROCESSED_PDFS_FILE):
        try:
            with open(PROCESSED_PDFS_FILE, "rb") as f:
                pdf_links = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            st.error(f"Error loading processed PDFs: {e}")
    return pdf_links

def save_processed_pdfs(processed_pdfs: List[str]) -> None:
    """Save the list of processed PDFs to a pickle file, with error handling."""
    try:
        with open(PROCESSED_PDFS_FILE, "wb") as f:
            pickle.dump(processed_pdfs, f)
        st.success("Batch saved successfully.")
    except Exception as e:
        st.error(f"Error saving batch: {e}")


def display_files(new_pdfs: list, already_processed: list) -> None:
    """Display the already processed and new PDFs side by side in two columns."""

    col1, col2 = st.columns(2)

    with col1:
        if already_processed:
            st.markdown("### Files in Vector Store")
            st.write("*Documents vectorisés et indexés dans le vector store de chromadb.*")
            st.markdown("___")

            for i, pdf in enumerate(already_processed[:5], 1):
                st.write(f"{i}. {pdf}")

            if len(already_processed) > 10:
                st.write("...")

            for i, pdf in enumerate(already_processed[-5:], len(already_processed) - 4):
                st.write(f"{i}. {pdf}")
        else:
            st.subheader("Files in Vector Store")
            st.write("Vector store is empty.")

    with col2:
        if new_pdfs:
            st.markdown("### New files to vectorize")
            st.write("*Nouveaux documents à indexer dans le vector store de chromadb.*")
            st.markdown("___")

            for i, pdf in enumerate(new_pdfs[:5], 1):
                st.write(f"{i}. {pdf}")

            if len(new_pdfs) > 10:
                st.write("...")

            for i, pdf in enumerate(new_pdfs[-5:], len(new_pdfs) - 4):
                st.write(f"{i}. {pdf}")
        else:
            st.subheader("New files to vectorize")
            st.write("Vector store is already up to date.")