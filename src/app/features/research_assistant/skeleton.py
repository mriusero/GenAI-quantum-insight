from typing import Tuple

import streamlit as st

from .processing.preprocessor import DocumentProcessor
from .qa_system import QASystem, user_interface
from .utilities.helper import VECTOR_STORE_FILE

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def models_loading(hg_api_key: str, debug: bool = False) -> Tuple[DocumentProcessor, QASystem]:
    """Load and cache document processor and QA system models."""
    print("\n === Hugging Face logging ===\n")
    return (
        DocumentProcessor(
            debug=debug,
            embedding_model=EMBEDDING_MODEL
        ),
        QASystem(
            api_key=hg_api_key,
            debug=debug,
            model_name=MODEL_NAME,
            embedding_model=EMBEDDING_MODEL,
            vector_store_file=VECTOR_STORE_FILE
        )
    )

def run_assistance(qa_system: QASystem, debug: bool = False) -> None:
    """Main logic for the research assistant app as User Interface and QA system."""
    user_interface(qa_system)