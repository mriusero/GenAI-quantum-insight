import os
import pickle
from typing import List
import streamlit as st

PROCESSED_PDFS_FILE = "src/app/features/research_assistant/checkpoints/processed_pdfs.pkl"
VECTOR_STORE_FILE = "src/app/features/research_assistant/checkpoints/vector_store"

def load_processed_pdfs() -> List[str]:
    """Load processed PDFs from a pickle file, handle errors gracefully."""
    if os.path.exists(PROCESSED_PDFS_FILE):
        try:
            with open(PROCESSED_PDFS_FILE, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            st.error(f"Error loading processed PDFs: {e}")
    return []

def save_processed_pdfs(processed_pdfs: List[str]) -> None:
    """Save the list of processed PDFs to a pickle file, with error handling."""
    try:
        with open(PROCESSED_PDFS_FILE, "wb") as f:
            pickle.dump(processed_pdfs, f)
        st.success("Batch saved successfully.")
    except Exception as e:
        st.error(f"Error saving batch: {e}")
