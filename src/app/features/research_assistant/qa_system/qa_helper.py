import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class QA_helper:
    def __init__(self, embedding_model, debug=False, vector_store_file="VECTOR_STORE_FILE"):
        """
        Initializes the QA_helper class with a ChromaDB client and an embedding model.
        """
        self.vector_store_file = vector_store_file
        self.client = chromadb.PersistentClient(path=self.vector_store_file)
        self.collection = self.client.get_or_create_collection(name="arxiv_papers_collection")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.debug = debug
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    def retrieve_documents(self, collection_name: str, query: str, top_k: int = 3):
        """
        Retrieves relevant documents from a ChromaDB collection.
        """
        collection = self.client.get_collection(collection_name)
        query_embedding = self.embedding_model.encode([query])
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        if self.debug:
            st.write("## Question Embedding:\n", query_embedding)
            st.write("## Retrieval Query Result:\n", results)

        return results

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text.
        """
        tokens = self.tokenizer(text)["input_ids"]
        return len(tokens)

    @staticmethod
    def prolonge_answer(conversation_memory, feedback: str) -> str:
        """
        Creates a prompt based on the conversation history.
        """
        prompt = f"Complete the following text: {feedback}\n\n"
        for entry in conversation_memory:
            prompt += f"User: {entry['question']}  \nAssistant: {entry['response']}\n\n"
        return prompt.strip()
