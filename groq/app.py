import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()


## load groq api key
grok_api_key = os.environ['GROQ_API_KEY']

## Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector_store = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("Chat Groq demo")
# llm = ChatGroq(model="openai/gpt-oss-20b")

# llm = ChatOpenAI(
#     model="Qwen/Qwen3.5-4B",
#     base_url="http://185.103.84.38:8000/v1"
# )

llm = ChatOllama(model="qwen3.5:4b")

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions only based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector_store.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Wrtie your question!")

if prompt:
    start = time.process_time()
    response = retriever_chain.invoke({"input": prompt})
    print("Response time: ", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("=====================")