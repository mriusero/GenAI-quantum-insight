import chromadb
import streamlit as st
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

    INSTRUCTION = """    
Instructions:

    You are a scientific expert. Follow these guidelines to provide clear, structured, and engaging explanations:

        1. **Adapt to the question**:
           - Match your tone and explanation depth to the user's expertise.
           - Use formal or conversational language based on the question style.

        2. **Always** format your response in **Markdown**:
           - Use title, headers and subheaders to organize your content
           - Use **bullet points** for lists of ideas or steps.
           - Emphasize important terms using **bold** or *italic* text.
           - Separate different topics into paragraphs or distinct sections.
           - Pay attention to formatting markdown correctly with specials characters for a correct display.

        3. **Simplify complex ideas**:
           - Divide complex concepts into smaller, understandable parts.
           - Use real-world examples or analogies to make abstract ideas concrete.
           - Generate the content as a articles, cheat sheet, codes examples or scientific papers synthesis.

        4. **Code & technical content**:
           - When including code, enclose it in ```python **code blocks** ``` for better readability.
           - Incorporate technical terms, but explain them in an accessible way.

        5. **Citations**:
           - Include credible sources from scientist, and cite them clearly.

        6. **User-centered response**:
           - Keep the user’s perspective in mind. Provide explanations that are clear, informative, and moderately detailed.
           - Don't imagine a user question, always answer the latest query.
           - In case of ambiguity, ask for clarification or provide multiple interpretations.
           
    Now, """

    LEVEL_ADAPTATION = {
            "Beginner": "provide an easy-to-understand explanation for the following query, avoiding technical terms or complex concepts:\n",
            "Intermediate": "provide a clear and moderately detailed explanation for the following query, including some technical terms but keeping the explanation accessible:\n",
            "Advanced": "provide a comprehensive and detailed answer for the following query, incorporating technical terms and concepts, and offering a deeper exploration of the topic:\n",
            "Expert": "provide a highly technical, in-depth, and nuanced answer for the following query, using advanced terminology and concepts:\n"
    }

    def retrieve_documents(self, collection_name: str, query: str, top_k: int = 5):
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

    #(text)["input_ids"]

    def calculate_tokens(self, input):
        """
        Calculates the total number of tokens in the input data.
        """
        total = 0
        total += self.count_tokens(input["inputs"])
        return total

    def trim_conversation_history(self, max_length=4):
        """
        Trim the conversation history to avoid excessive memory usage.
        """
        if len(st.session_state.messages) > max_length:
            st.session_state.messages = st.session_state.messages[-max_length:]

        conversation_history = "\n".join(
            [f"User: {msg['content']}" if msg["role"] == "user" else f"System: {msg['content']}"
             for msg in st.session_state.messages]
        )
        history = f"{conversation_history}"

        st.session_state.tokens_count.empty()

        st.session_state.total_tokens += self.count_tokens(history)  # Update total tokens count
        with st.session_state.tokens_count:
            st.write(f"`{st.session_state.get('total_tokens', 0)}` tokens")
        progress_value = min(st.session_state.total_tokens / 4096, 1.0)
        st.session_state.tokens_bar.progress(progress_value)


    @staticmethod
    def add_context(usr_level: str) -> str:
        """Creates a prompt based on the conversation history."""
        level_instruction = QA_helper.LEVEL_ADAPTATION.get(usr_level, QA_helper.LEVEL_ADAPTATION["Intermediate"])
        return QA_helper.INSTRUCTION + level_instruction
