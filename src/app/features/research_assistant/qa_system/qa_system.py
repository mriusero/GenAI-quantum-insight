from typing import Dict, List, Any

import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

from .qa_helper import QA_helper


class QASystem:
    def __init__(self,
                 api_key: str,
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_file: str = "VECTOR_STORE_FILE",
                 debug: bool = False) -> None:
        """
        Initialize the Q&A system with LLM, embeddings, memory, and vector store.
        """
        self.api_key: str = api_key
        self.llm = HuggingFaceEndpoint(repo_id=model_name, huggingfacehub_api_token=api_key)
        self.debug: bool = debug
        self.base_url: str = f"https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        self.conversation_memory: List[Dict[str, str]] = []
        self.helper = QA_helper(embedding_model=embedding_model, debug=self.debug, vector_store_file=vector_store_file)


    def query(self, headers: Dict[str, str], data: Dict[str, Any]) -> str:
        """
        Send a query to the Hugging Face API and retrieve the response.
        """
        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code == 200:
            response = response.json()
            if self.debug:
                st.write(f"### Model feedback:\n")
                st.json(response)
            return response[0].get("generated_text", "Aucune réponse trouvée.")
        else:
            return f"Erreur: {response.status_code}"


    def generate_augmented_response(self,
                                    query: str,
                                    retrieved_docs: Dict[str, Any],
                                    usr_level: str = "Intermediate",
                                    temperature: float = 0.0,
                                    max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate a structured response object using the retrieved documents.
        """
        context: str = self.helper.add_context(usr_level)
        query: str = f"{query}"
        query += "\nsystem:"

        response_docs = [{
            "document_number": i + 1,
            "title": metadata.get('title', 'Titre inconnu'),
            "author": metadata.get('author', 'Auteur inconnu'),
            "published": metadata.get('published', 'Date de publication non disponible'),
            "summary": metadata.get('summary', 'Résumé non disponible')[:300],
            "pdf_link": metadata.get('pdf_link', 'Lien PDF non disponible'),
            "text": metadata.get('text', 'Texte non disponible')
        } for i, metadata in enumerate(retrieved_docs.get('metadatas', [[]])[0][:5])]

        input: Dict[str, Any] = {
            "inputs": f"{context}\n{query}",
            "documents": response_docs,
            "options": {
                "use_cache": False,
                "temperature": float(temperature),
                "max_new_tokens": int(max_tokens),  # Nombre maximum de nouveaux tokens à générer
                "top_k": 50,  # Nombre de tokens les plus probables à garder pour le filtrage top-k
                "top_p": 0.9,  # Valeur de filtrage pour la stratégie de sampling "nucleus" (top-p)
                "frequency_penalty": 0.0,  # Pénalité pour la fréquence des tokens générés
                "repetition_penalty": 1.0,  # Pénalité pour les répétitions de tokens
                "stop": [".", "\n"],  # Liste de séquences à considérer comme arrêts dans la génération
                "seed": None,  # Graines pour le sampling aléatoire (None pour un comportement aléatoire)
                "do_sample": True,  # Active l'échantillonnage des logits
                "best_of": 1,  # Nombre de séquences générées pour choisir la meilleure
                "return_full_text": False,  # Indique si le texte d'entrée doit être préfixé au texte généré
            }
        }
        if self.debug:
            st.write(f"### Prompt context:\n{context}")
            st.write(f"### User query:\n {query}")
            st.write(f"### Retrieval Query Result:\n")
            st.json(retrieved_docs)
            st.write(f"### Response documents (Retrieval result reshaped):\n")
            st.json(response_docs)
            st.write(f"### Model input:\n")
            st.json(input)
        return input


    def extend_response(self, initial_feedback: str, input: Dict[str, Any], headers: Dict[str, str]) -> str:
        """
        Extend the response by querying until the answer stabilizes or the max iterations are reached.
        """
        initial_input = input["inputs"]    # Update the input with the previous exchanges
        attachments = input["documents"]
        options = input["options"]
        new_input = {
            "inputs": initial_feedback,
            "documents": attachments,
            "options": options
        }

        max_iterations = 7    # Init for response extension
        prev_feedback = ""
        response_placeholder = st.empty()
        final_answer = initial_feedback.replace(initial_input, "").replace("Answer:", "").strip()

        for _ in range(max_iterations):                         # Loop to extend the response
            new_feedback = self.query(headers, new_input)
            if not new_feedback:
                print("WARNING: Received empty feedback from query.")
                break

            new_answer = new_feedback.replace(new_input["inputs"], "").replace("Answer:", "").strip()
            clean_new_answer = ''.join(new_answer.split()).lower()          # For comparison
            clean_prev_feedback = ''.join(prev_feedback.split()).lower()

            if clean_new_answer == clean_prev_feedback:    # Check if answer has stabilized
                print("\nINFO: -- Response appears complete or no longer improves.\n")
                final_answer += new_answer if new_answer else prev_feedback
                break

            final_answer += new_answer
            prev_feedback = new_answer
            new_input["inputs"] = new_feedback
            response_placeholder.markdown(f"<div class='system-bubble'>{final_answer}</div>", unsafe_allow_html=True)

        response_placeholder.empty()    # Clear the real-time response placeholder
        return final_answer.strip()


    def ask_question(self, usr_question: str, usr_level: str = "Intermediate",
                     temperature: float = 0.0, max_tokens: int = 500) -> str:
        """
        Process the user's question using retrieval-augmented generation (RAG).
        """
        st.session_state.total_tokens = 0
        st.session_state.tokens_count.empty()
        try:
            conversation_history = "\n".join(
                [f"user: {msg['content']}" if msg["role"] == "user" else f"system: {msg['content']}"
                 for msg in st.session_state.messages]
            )
            query_with_history = f"{conversation_history}\nuser:\n{usr_question}\n_____________________________\n"
            retrieved_docs = self.helper.retrieve_documents("arxiv_papers_collection", usr_question)
            input = self.generate_augmented_response(query=query_with_history,
                                                     retrieved_docs=retrieved_docs,
                                                     usr_level=usr_level,
                                                     temperature=temperature,
                                                     max_tokens=max_tokens)

            # -- Tokens input count
            st.session_state.total_tokens = self.helper.calculate_tokens(input=input)
            with st.session_state.tokens_count:
                st.write(f"`{st.session_state.get('total_tokens', 0)}` tokens")

            progress_value = min(st.session_state.total_tokens / 4096, 1.0)
            st.session_state.tokens_bar.progress(progress_value)

            if st.session_state.total_tokens > 2000:
                #st.info("Context will exceed the 4096 tokens limit, conversation memory trimmed.\n"
                self.helper.trim_conversation_history()
            else:

                print(f"Context & User question:\n--------------------------\n {input['inputs']}\n")   # Display the context, user question, and parameters
                print(f"Parameters:\n---------------------------------------\n"
                      f"- Temperature: {temperature}\n"
                      f"- Max tokens per answer: {max_tokens}\n"
                      f"- Explanation: {usr_level}\n"
                      f"- Total tokens: {st.session_state.total_tokens}\n")

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                feedback = self.query(headers, input)                                                       # Send the query to the model
                answer = feedback.replace(input["inputs"], "").replace("Answer:", "")    # Extract the answer from the feedback
                final_answer = self.extend_response(feedback, input, headers)                               # Extend the answer if needed

                print(f"\nInitial answer:\n----------------------------------------------------\n{answer}")
                print(f"\nReturning final_answer:\n----------------------------------------------------\n{final_answer}")

                st.session_state.response_placeholder = None

                # -- Tokens output count
                st.session_state.tokens_count.empty()

                st.session_state.total_tokens += self.helper.count_tokens(final_answer)
                with st.session_state.tokens_count:
                    st.write(f"`{st.session_state.get('total_tokens', 0)}` tokens")

                progress_value = min(st.session_state.total_tokens / 4096, 1.0)
                st.session_state.tokens_bar.progress(progress_value)

                if st.session_state.total_tokens > 2000:                                                 # Check if the response exceeds the token limit
                    #st.info("Request exceeds the 4096 tokens limit, conversation memory was trimmed.\n"
                    #        "Note: you can also save or download a discussion before exceeding the context window size limit. (see tokens numbers above)")
                    self.helper.trim_conversation_history()

                return final_answer if final_answer else "No valid answer found"

        except Exception as e:
            st.error(f"Erreur système : {str(e)}")
            return "An error occurred while processing your question."
