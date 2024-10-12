# cache.py

import os
import pickle
import streamlit as st
from typing import List

def load_processed_pdfs(file_path: str) -> List[str]:
    """Load processed PDFs from a pickle file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            st.error(f"Error loading processed PDFs: {e}")
    return []

def save_processed_pdfs(processed_pdfs: List[str], file_path: str) -> None:
    """Save processed PDFs to a pickle file."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(processed_pdfs, f)
        st.write("Batch saved successfully.")
    except Exception as e:
        st.error(f"Error saving batch: {e}")
