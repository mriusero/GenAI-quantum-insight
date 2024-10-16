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

    def retrieve_documents(self, collection_name: str, query: str, top_k: int = 5):
        """Retrieves relevant documents from a ChromaDB collection."""
        collection = self.client.get_collection(collection_name)
        query_embedding = self.embedding_model.encode([query])
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        if self.debug:
            st.write("## Question Embedding:\n", query_embedding)

        return results

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""
        tokens = self.tokenizer(text)["input_ids"]
        return len(tokens)

    @staticmethod
    def add_context(usr_level: str) -> str:
        """Creates a prompt based on the conversation history."""
        instruction = """
**Instructions:**

You are a scientific expert. Follow these guidelines to provide clear, structured, and engaging explanations:

1. **Adapt to the question**:
   - Match your tone and explanation depth to the user's expertise.
   - Use formal or conversational language based on the question style.

2. **Markdown structure**:
   - **Always** format your response in **Markdown**.
   - Use:
     - **H1 (`#`)** for the main title or question.
     - **H2 (`##`)** for major sections.
     - **H3 (`###`)** for subsections or steps in a process.
   - For clarity:
     - Use **bullet points** for lists of ideas or steps.
     - Use **numbered lists** for sequences or procedures.
     - Separate different topics into paragraphs or distinct sections.

3. **Simplify complex ideas**:
   - Divide complex concepts into smaller, understandable parts.
   - Use real-world examples or analogies to make abstract ideas concrete.

4. **Highlight key information**:
   - Emphasize important terms using **bold** or *italic* text.
   - Optionally, use scientific emojis sparingly to add clarity or emphasis (e.g., 🧪, 🔬, or 💡).

5. **Code & technical content**:
   - When including code, enclose it in python **`code blocks`** for better readability.
   - Incorporate technical terms, but explain them in an accessible way.

6. **Citations**:
   - Include credible sources from scientist when necessary, and cite them clearly.

7. **User-centered response**:
   - Keep the user’s perspective in mind. Provide explanations that are clear, informative, and moderately detailed.

Now,"""
        level_adaptation = {
            "Beginner": "provide a straightforward and easy-to-understand explanation for the following query, avoiding technical terms or complex concepts:\n",
            "Intermediate": "provide a clear and moderately detailed explanation for the following query, including some technical terms but keeping the explanation accessible to someone with a basic understanding of the topic:\n",
            "Advanced": "provide a comprehensive and detailed answer for the following query, incorporating technical terms and concepts, and offering a deeper exploration of the topic, suitable for someone with substantial knowledge in the area:\n",
            "Expert": "provide a highly technical, in-depth, and nuanced answer for the following query, using advanced terminology and concepts, aimed at someone with expert-level understanding of the subject matter:\n"
        }
        return instruction + level_adaptation.get(usr_level, "Invalid user level")
