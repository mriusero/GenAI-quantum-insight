import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


class ResearchAssistant:
    def __init__(self, api_key, model_name="meta-llama/Llama-2-7b-chat-hf",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):

        self.llm = HuggingFaceEndpoint(repo_id=model_name, temperature=0,
                                       huggingfacehub_api_token=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vector_store = None

    def load_documents(self, pdf_urls):
        """Load and split PDF documents into chunks."""
        documents = []
        for url in pdf_urls:
            loader = PyPDFLoader(url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            documents.extend(chunks)
        return documents

    def create_vector_store(self, documents):
        """Create a VectorStore with ChromaDB from the given documents."""
        self.vector_store = Chroma.from_documents(documents, self.embeddings)

    def ask_question(self, question):
        """Ask a question using Retrieval-Augmented Generation (RAG)."""
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please load documents first.")

        retriever = self.vector_store.as_retriever()        # Creating the question-answer chain
        qa_chain = ConversationalRetrievalChain.from_llm(self.llm, retriever, memory=self.memory)

        result = qa_chain({"question": question})   # Retrieve the answer
        return result['answer']

    def run_assistance(self, pdf_urls):
        """Launch the Streamlit application for questions."""
        st.title("Quantum Computing Research Assistant")

        if st.button("Load and Index Documents"):    # Load and process documents
            documents = self.load_documents(pdf_urls)
            self.create_vector_store(documents)
            st.write("Documents successfully loaded and indexed in VectorStore.")

        user_question = st.text_input("Ask a question about quantum computing:") # Ask a question
        if st.button("Submit Question"):
            response = self.ask_question(user_question)
            st.write("Answer:", response)  # Answer:

