from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant. Please response to the user queries."),
        ("user", "Question:{question}")
    ]
)


## steamlit framework

st.title("Langchain chat bot with qwen 3.5")
input_text=st.text_input("Ask your question")

## ollama qwen3.5 model

llm=ChatOllama(model="qwen3.5",
               base_url=os.environ["OLLAMA_HOST"] )
output_parser=StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))