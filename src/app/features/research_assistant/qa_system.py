# qa_system.py

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class QASystem:
    def __init__(self, api_key, debug=False,
                 model_name="meta-llama/Llama-2-7b-chat-hf",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = HuggingFaceEndpoint(repo_id=model_name, temperature=0, huggingfacehub_api_token=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.debug = debug

    def ask_question(self, question, vector_store):
        """Ask a question using Retrieval-Augmented Generation (RAG)."""
        retriever = vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(self.llm, retriever, memory=self.memory)
        result = qa_chain({"question": question})
        if self.debug:
            print(result)
        return result['answer']
