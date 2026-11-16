from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant. Please response to the user queries."),
        ("user", "Question:{question}")
    ]
)


## steamlit framework

st.title("Langchain chat bot with api")
input_text=st.text_input("Ask your question")

## Ai model

llm=ChatOpenAI(model="gpt-4.1-mini")
output_parser=StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))