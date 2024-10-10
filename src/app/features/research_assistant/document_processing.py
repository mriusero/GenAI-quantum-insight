# document_processing.py
import streamlit as st
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100, debug=False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug

    def load_and_split(self, pdf_urls):
        """Load and split PDF documents into chunks."""
        documents = []
        for url in pdf_urls:
            loader = PyPDFLoader(url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(docs)
            documents.extend(chunks)
            logging.info(f"Document from {url} split into {len(chunks)} chunks.")

            if self.debug:
                st.write(chunks)

        return documents