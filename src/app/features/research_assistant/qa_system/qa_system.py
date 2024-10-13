import chromadb
import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class QASystem:
    def __init__(self, api_key: str,
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_file: str = "VECTOR_STORE_FILE",
                 debug: bool = False):
        """Initialize the Q&A system with LLM, embeddings, memory, and vector store."""
        self.api_key = api_key
        self.llm = HuggingFaceEndpoint(repo_id=model_name, temperature=0, huggingfacehub_api_token=api_key)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.debug = debug
        self.vector_store_file = vector_store_file
        self.client = chromadb.PersistentClient(path=self.vector_store_file)
        self.collection = self.client.get_or_create_collection(name="arxiv_papers_collection")
        self.base_url = f"https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

        # Initialiser le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    def initialize_ui(self):
        """Set up UI for the Q&A system using Streamlit."""
        usr_question = st.text_input("Ask a question:")
        if st.button("Submit"):
            with st.spinner('Processing...'):
                response = self.ask_question(usr_question)
                st.markdown(f"## Answer:\n\n  {response}\n")

    def retrieve_documents(self, collection_name: str, query: str, top_k: int = 3):
        """Retrieve relevant documents from the ChromaDB collection."""
        collection = self.client.get_collection(collection_name)
        query_embedding = self.embedding_model.encode([query])
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        if self.debug:
            st.write("## Question Embedding:\n", query_embedding)
            st.write("## Query Result:\n", results)

        return results

    def generate_augmented_response(self, query: str, retrieved_docs: dict) -> str:
        """Generate a response using the retrieved documents and Hugging Face API (Llama-2)."""
        document = f"Informations about {query}\n\n"
        for i, metadata in enumerate(retrieved_docs['metadatas'][0][:3]):
            title = metadata.get('title', 'Titre inconnu')
            summary = metadata.get('summary', 'Résumé non disponible')
            text = metadata.get('text', 'Text non disponible')
            document += f"\nDocument {i + 1}:\n- Title: {title}\n- Summary: {summary}\n- Text: {text}\n"
        return document

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text using the tokenizer."""
        return len(self.tokenizer.encode(text))

    def ask_question(self, usr_question: str, usr_level: str = "Intermediate") -> str:
        """Process the user's question using retrieval-augmented generation (RAG)."""
        try:
            retrieved_docs = self.retrieve_documents("arxiv_papers_collection", usr_question)
            document = self.generate_augmented_response(usr_question, retrieved_docs)

            total_tokens = self.count_tokens(usr_question) + self.count_tokens(document)
            if total_tokens > 4096:
                return "La combinaison de la question et des documents récupérés dépasse la limite de 4096 tokens."

            prompt = usr_question
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "inputs": prompt,
                "documents": document,
                "options": {
                    "use_cache": True,
                    "temperature": 0,
                    "max_length": 500,
                }
            }
            if self.debug:
                st.write("## Request Data:\n", data)
                st.write(f"#### Total tokens in input:\n`{total_tokens}`")

            response = requests.post(self.base_url, headers=headers, json=data)
            if response.status_code == 200:
                response_data = response.json()
                return response_data[0].get("generated_text", "Aucun texte généré.") if isinstance(response_data, list) and response_data else "Aucune donnée générée trouvée."
            return f"Erreur: {response.status_code}"

        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")
            return "Une erreur est survenue lors du traitement de votre question."