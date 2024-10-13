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
    """Display the already processed and new PDFs."""

    if already_processed:
        print("\n=== Files in Vector Store ===")

        for i, pdf in enumerate(already_processed[:5], 1):  # Afficher les 5 premiers fichiers
            print(f"{i}. {pdf}")

        if len(already_processed) > 10:  # Si la liste contient plus de 10 éléments, ajouter une indication "..."
            print("...")

        for i, pdf in enumerate(already_processed[-5:], len(already_processed) - 4):  # Afficher les 5 derniers fichiers
            print(f"{i}. {pdf}")
    else:
        print("\nVector store is empty.")

    print("_"*50)

    if new_pdfs:
        print("\n=== New files to vectorize ===")
        for i, pdf in enumerate(new_pdfs[:5], 1):
            print(f"{i}. {pdf}")

        if len(new_pdfs) > 10:
            print("...")

        for i, pdf in enumerate(new_pdfs[-5:], len(new_pdfs) - 4):
            print(f"{i}. {pdf}")
    else:
        print("\nVector store is already up to date.")