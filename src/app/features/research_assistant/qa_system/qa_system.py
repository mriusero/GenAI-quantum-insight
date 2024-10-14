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
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.conversation_memory = []

    def initialize_ui(self):
        """Set up UI for the Q&A system using Streamlit."""
        st.title("Système de Questions-Réponses")
        st.write("Posez votre question ci-dessous.")

        usr_question = st.text_input("Votre question:")
        if st.button("Soumettre"):
            if usr_question:
                with st.spinner('Traitement...'):
                    response = self.ask_question(usr_question)
                    st.markdown(f"## Réponse:\n\n  {response}\n")
            else:
                st.warning("Veuillez poser une question.")

        if self.conversation_memory:
            st.write("### Historique de la conversation:")
            for entry in self.conversation_memory:
                st.write(f"**Vous:** {entry['question']}")
                st.write(f"**Assistant:** {entry['response']}")

        if st.button("Réinitialiser la conversation"):
            self.conversation_memory = []

        #usr_question = st.text_input("Ask a question:")
        #if st.button("Submit"):
        #    with st.spinner('Processing...'):
        #        response = self.ask_question(usr_question)
        #        st.markdown(f"## Answer:\n\n  {response}\n")

    def retrieve_documents(self, collection_name: str, query: str, top_k: int = 3):
        """Retrieve relevant documents from the ChromaDB collection."""
        collection = self.client.get_collection(collection_name)
        query_embedding = self.embedding_model.encode([query])
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        if self.debug:
            st.write("## Question Embedding:\n", query_embedding)
            st.write("## Query Result:\n", results)

        return results

    def generate_augmented_response(self, query: str, retrieved_docs: dict) -> dict:
        """Generate a structured response object using the retrieved documents."""
        prompt = self.prolonge_answer(query)
        response_docs = []
        for i, metadata in enumerate(retrieved_docs.get('metadatas', [[]])[0][:2]):
            doc_info = {
                "document_number": i + 1,
                "title": metadata.get('title', 'Titre inconnu'),
                "author": metadata.get('author', 'Auteur inconnu'),
                "published": metadata.get('published', 'Date de publication non disponible'),
                "summary": metadata.get('summary', 'Résumé non disponible')[:300],
                "text": metadata.get('text', 'Texte non disponible'),
                "pdf_link": metadata.get('pdf_link', 'Lien PDF non disponible')
            }
            response_docs.append(doc_info)

        response = {                        # Créer l'objet de réponse
            "inputs": prompt,
            "documents": response_docs,
            "options": {
                "use_cache": True,
                "temperature": 0.1,
                "max_new_tokens": 400,              # Nombre maximum de nouveaux tokens à générer
                "top_k": 50,                        # Nombre de tokens les plus probables à garder pour le filtrage top-k
                "top_p": 0.9,                       # Valeur de filtrage pour la stratégie de sampling "nucleus" (top-p)
                "frequency_penalty": 0.0,           # Pénalité pour la fréquence des tokens générés
                "repetition_penalty": 1.0,          # Pénalité pour les répétitions de tokens
                "stop": [".\n"],                    # Liste de séquences à considérer comme arrêts dans la génération
                "seed": None,                       # Graines pour le sampling aléatoire (None pour un comportement aléatoire)
                "do_sample": True,                  # Active l'échantillonnage des logits
                "best_of": 1,                       # Nombre de séquences générées pour choisir la meilleure
                "return_full_text": False,           # Indique si le texte d'entrée doit être préfixé au texte généré
            }
        }
        return response

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text using the tokenizer."""
        return len(self.tokenizer.encode(text))

    def query(self, headers, data):
        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code == 200:
            data = response.json()
            st.write(data)
            return data[0].get("generated_text", "Aucune réponse trouvée.")
        else:
            return f"Erreur: {response.status_code}"

    def prolonge_answer(self, feedback: str) -> str:
        """Create a prompt for the model, including memory of past interactions."""
        prompt = f"Complete the following text and structure your answer in markdown with title, subtitles, front tips, code... : {feedback}\n\n"
        for entry in self.conversation_memory:
            prompt += f"User:{entry['question']}  \nAssistant: {entry['response']}\n\n"
        return prompt.strip()

    def ask_question(self, usr_question: str, usr_level: str = "Intermediate") -> str:
        """Process the user's question using retrieval-augmented generation (RAG)."""
        try:
            retrieved_docs = self.retrieve_documents("arxiv_papers_collection", usr_question)
            data = self.generate_augmented_response(usr_question, retrieved_docs)
            total_tokens = self.count_tokens(usr_question)

            for doc in data['documents']:
                total_tokens += self.count_tokens(doc['title'])
                total_tokens += self.count_tokens(doc['author'])
                total_tokens += self.count_tokens(doc['published'])
                total_tokens += self.count_tokens(doc['summary'])
                total_tokens += self.count_tokens(doc['text'])
                total_tokens += self.count_tokens(doc['pdf_link'])

            print(f"Total token in query: {total_tokens}")

            if total_tokens > 4096:
                return st.error("La requête dépasse la limite de 4096 tokens.")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if self.debug:
                st.write("## Request Data:\n", data)
                st.write(f"#### Total tokens in input:\n{total_tokens}")

            response_text = self.query(headers, data)
            self.conversation_memory.append({"question": usr_question, "response": response_text})
            return response_text

        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")
            return "Une erreur est survenue lors du traitement de votre question."
