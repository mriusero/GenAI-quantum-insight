import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from .document_processing import DocumentProcessor
from .vector_store_manager import VectorStoreManager
from .qa_system import QASystem

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def run_assistance(hg_api_key, debug=False):
    """Run the research assistant app."""

    st.title("Research Assistant")

    document_processing = DocumentProcessor(debug=debug)                             # Initialize document processor
    qa_system = QASystem(api_key=hg_api_key, debug=debug, model_name=MODEL_NAME,
                         embedding_model=EMBEDDING_MODEL)                            # Initialize QA system
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)                   # Embedding model setup
    vector_store_manager = VectorStoreManager(embeddings, debug=debug)               # Vector store manager setup

    pdf_urls = [
        "http://arxiv.org/pdf/2405.20113v2",
        "http://arxiv.org/pdf/2410.07099v1"
    ]  # Sample PDF URLs

    if st.button("Load Documents"):
        if pdf_urls:
            documents = document_processing.load_and_split(pdf_urls)
            vector_store_manager.create_vector_store(documents)
            st.write("Documents loaded and indexed.")

            if debug:
                st.write(f"Loaded {len(documents)} documents.")
        else:
            st.error("Please provide valid PDF URLs.")

    user_question = st.text_input("Ask a question:")
    if st.button("Submit"):
        try:
            vector_store = vector_store_manager.get_vector_store()
            response = qa_system.ask_question(user_question, vector_store)
            st.write(f"Answer: {response}")
        except ValueError as e:
            st.error(str(e))
