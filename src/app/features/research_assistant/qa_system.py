import streamlit as st
import chromadb
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from transformers import pipeline

class QASystem:
    def __init__(self, api_key, debug=False,
                 model_name="facebook/bart-large-cnn",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = HuggingFaceEndpoint(repo_id=model_name, temperature=0, huggingfacehub_api_token=api_key)  # Initialize LLM
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)  # Initialize embeddings
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Initialize memory
        self.debug = debug  # Debug flag

    def initialize_ui(self, VECTOR_STORE_FILE):
        """Initialize the user interface for the Q&A system."""  # Initialize UI for Q&A
        usr_level = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])
        usr_question = st.text_input("Ask a question:")

        if st.button("Submit"):
            response = self.ask_question(usr_question, VECTOR_STORE_FILE, usr_level)
            st.markdown(response)  # Display the response to the user

    def get_text_by_id(self, id, query_result):
        """Get the text associated with a document ID from the query result."""  # Retrieve text by document ID
        try:
            ids = query_result["ids"][0]  # List of IDs
            index = ids.index(id)  # Find the index of the ID

            # Retrieve corresponding metadata
            metadata = query_result["metadatas"][0][index]
            formatted_text = (
                f"**Title:** {metadata.get('title', 'No Title')}\n\n"
                f"**Author:** {metadata.get('author', 'Unknown Author')}\n\n"
                f"**Published:** {metadata.get('published', 'No Publication Date')}\n\n"
                f"**Summary:** {metadata.get('summary', 'No Summary Available')}\n\n"
                f"**Text:** {metadata.get('text', 'No Text Available')}\n\n"
                f"[PDF Link]({metadata.get('pdf_link', '')})" if metadata.get("pdf_link") else ""
            )
            return formatted_text

        except Exception as e:
            st.error(f"An error occurred while retrieving the text for ID {id}: {e}")  # Error handling
            return "Sorry, I couldn't retrieve the associated text."

    def ask_question(self, usr_question, VECTOR_STORE_FILE, usr_level="Intermediate"):
        """Ask a question using retrieval-augmented generation (RAG)."""  # RAG for Q&A
        client = chromadb.PersistentClient(path=VECTOR_STORE_FILE)
        collection = client.get_or_create_collection(name="arxiv_papers_collection")

        question_embedding = self.embedding_model.embed_query(usr_question)
        query_result = collection.query(query_embeddings=[question_embedding], n_results=5)

        if self.debug:
            st.write(query_result)

        if "ids" in query_result and query_result["ids"]:
            ids, distances = query_result["ids"][0], query_result["distances"][0]  # Extract IDs and distances

            responses = [self.get_text_by_id(id, query_result) for id in ids]  # Get text for all IDs
            combined_text = "\n".join(responses)  # Combine all responses

            if not combined_text.strip():
                st.error("Aucun contenu pertinent trouvé à résumer.")  # No content found
                return "Sorry, I couldn't find an answer to your question."

            combined_text = combined_text[:1024]  # Truncate to max input length
            summary = self._summarize_text(combined_text)  # Summarize text
            translation = self._translate_text(summary)  # Translate summary

            return f"## Response for {usr_level}:\n\n{translation}\n\n"  # Format the response
        return "Sorry, I couldn't find an answer to your question."  # No results found

    def _summarize_text(self, text):
        """Summarize the provided text."""  # Text summarization
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        try:
            return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            st.error(f"Error during summarization: {e}")  # Error handling
            return "Sorry, I couldn't summarize the text."

    def _translate_text(self, text):
        """Translate the provided text to French."""  # Text translation
        translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        try:
            return translator(text, max_length=400)[0]['translation_text']  # Increase max_length for translation
        except Exception as e:
            st.error(f"Error during translation: {e}")  # Error handling
            return "Sorry, I couldn't translate the summary."
