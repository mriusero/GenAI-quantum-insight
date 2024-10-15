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

        return results


    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text.
        """
        tokens = self.tokenizer(text)["input_ids"]
        return len(tokens)

    @staticmethod
    def add_context(usr_level: str) -> str:
        """
        Creates a prompt based on the conversation history.
        """
        level_adaptation = {
            "Beginner": "Provide a straightforward and easy-to-understand explanation for the following question, avoiding technical terms or complex concepts:",
            "Intermediate": "Provide a clear and moderately detailed explanation for the following question, including some technical terms but keeping the explanation accessible to someone with a basic understanding of the topic:",
            "Advanced": "Provide a comprehensive and detailed answer for the following question, incorporating technical terms and concepts, and offering a deeper exploration of the topic, suitable for someone with substantial knowledge in the area:",
            "Expert": "Provide a highly technical, in-depth, and nuanced answer for the following question, using advanced terminology and concepts, aimed at someone with expert-level understanding of the subject matter:"
        }
        return level_adaptation.get(usr_level, level_adaptation[f"{usr_level}"])

    @staticmethod
    def prolonge_answer(usr_level: str) -> str:
        """
        Creates a prompt based on the conversation history.
        """
        level_adaptation = {
            "Beginner": "Please provide a simple and clear continuation of the following text.",
            "Intermediate": "Please provide a moderately detailed continuation of the following text.",
            "Advanced": "Please provide a thorough and detailed continuation of the following text.",
            "Expert": "Please provide an in-depth and highly technical continuation of the following text."
        }
        return level_adaptation.get(usr_level, level_adaptation[f"{usr_level}"])