import chromadb
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from transformers import pipeline

class QASystem:
    def __init__(self, api_key, debug=False,
                 model_name="facebook/bart-large-cnn",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_file="VECTOR_STORE_FILE"):
        """Initialize the Q&A system with LLM, embeddings, memory, and vector store."""
        self.llm = HuggingFaceEndpoint(repo_id=model_name, temperature=0, huggingfacehub_api_token=api_key)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.debug = debug
        self.vector_store_file = vector_store_file

    def initialize_ui(self):
        """Set up UI for the Q&A system."""
        usr_level = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])
        usr_question = st.text_input("Ask a question:")
        if st.button("Submit"):
            st.markdown(self.ask_question(usr_question, usr_level))

    def get_text_by_id(self, id, query_result):
        """Retrieve and format metadata from the query result by document ID."""
        try:
            index = query_result["ids"][0].index(id)
            metadata = query_result["metadatas"][0][index]
            return (
                f"**Title:** {metadata.get('title', 'No Title')}\n"
                f"**Author:** {metadata.get('author', 'Unknown Author')}\n"
                f"**Published:** {metadata.get('published', 'No Publication Date')}\n"
                f"**Summary:** {metadata.get('summary', 'No Summary Available')}\n"
                f"**Text:** {metadata.get('text', 'No Text Available')}\n"
                f"[PDF Link]({metadata.get('pdf_link', '')})" if metadata.get("pdf_link") else ""
            )
        except Exception as e:
            st.error(f"Error retrieving text for ID {id}: {e}")
            return "Sorry, I couldn't retrieve the associated text."

    def ask_question(self, usr_question, usr_level="Intermediate"):
        """Process the question using retrieval-augmented generation (RAG)."""
        client = chromadb.PersistentClient(path=self.vector_store_file)
        collection = client.get_or_create_collection(name="arxiv_papers_collection")

        question_embedding = self.embedding_model.embed_query(usr_question)
        query_result = collection.query(query_embeddings=[question_embedding], n_results=5)

        if self.debug:
            st.write(query_result)

        if "ids" in query_result and query_result["ids"]:
            combined_text = "\n".join(self.get_text_by_id(id, query_result) for id in query_result["ids"][0])
            if combined_text.strip():
                combined_text = combined_text[:1024]
                summary = self._summarize_text(combined_text)
                translation = self._translate_text(summary)
                return f"## Response for {usr_level}:\n\n{translation}"
            st.error("No relevant content found to summarize.")
        return "Sorry, I couldn't find an answer to your question."

    def _summarize_text(self, text):
        """Summarize the input text using a Hugging Face model."""
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        try:
            return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return "Sorry, I couldn't summarize the text."

    def _translate_text(self, text):
        """Translate the text to French using a Hugging Face model."""
        translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        try:
            return translator(text, max_length=400)[0]['translation_text']
        except Exception as e:
            st.error(f"Error during translation: {e}")
            return "Sorry, I couldn't translate the summary."
