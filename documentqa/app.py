import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.vectorstores import FAISS
from langchain_classic.document_loaders import PyPDFDirectoryLoader
import time


def vector_embedding():
    if "retriever_chain" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
        st.session_state.loader = PyPDFDirectoryLoader("./documentqa/tam")
        st.session_state.documents = st.session_state.loader.load()
        print("Number of documents:", len(st.session_state.documents))
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.documents)
        print("Number of final documents:", len(st.session_state.final_docs))
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
        st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
        st.session_state.retriever = st.session_state.vector_store.as_retriever()
        st.session_state.retriever_chain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)


load_dotenv()

## langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST")

st.title("Pdf qa with Ollama")


llm = ChatOllama(
    model="qwen3.8:27b",
    base_url=os.environ['OLLAMA_HOST']
    )

prompt = ChatPromptTemplate.from_template(
"""
Answer questions based on provided context only.
Please provide the most accurate response base on the question.

<context>
{context}
</context>

Question: {input}
"""
)

question = st.text_input("Enter yor question about tam")

if st.button("Document embedding"):
    vector_embedding()
    st.write("Vectorstore in ready")

if question:
    start = time.process_time()
    response = st.session_state.retriever_chain.invoke({'input' : question})
    st.write(response['answer'])

     # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")