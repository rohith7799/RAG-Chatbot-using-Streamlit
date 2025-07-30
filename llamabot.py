import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 RAG-Powered PDF Chatbot")

# ----------------------- SESSION STATE INIT -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ----------------------- SIDEBAR - PDF UPLOAD -----------------------
with st.sidebar:
    st.header("📄 Upload PDF(s)")
    pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    documents = []
    if pdf_files:
        for uploaded_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()
            documents.extend(pages)
            os.remove(tmp_path)

        st.success(f"✅ Loaded {len(documents)} pages.")

        # Text Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Embedding
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

        # FAISS Vector Store
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        st.session_state.vector_db = vector_store


# ----------------------- RAG CHAIN SETUP -----------------------
def setup_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 0.8})
    llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

    prompt_template = """
    You are an assistant for answering questions based on documents.
    Use ONLY the context below to answer the question.
    If you don't know the answer, say you don't know.

    ### Question:
    {question}

    ### Context:
    {context}

    ### Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | combine_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ----------------------- MAIN CHAT INTERFACE -----------------------
col1, col2 = st.columns([1, 2])

with col2:
    st.subheader("💬 Chat with your PDFs")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the PDFs...")

    if user_input:
        if not st.session_state.vector_db:
            st.warning("⚠️ Please upload PDF(s) first.")
        else:
            # Append user message
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            rag_chain = setup_rag_chain(st.session_state.vector_db)
            with st.spinner("🤖 Thinking..."):
                response = rag_chain.invoke(user_input)

            # Append assistant message
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
