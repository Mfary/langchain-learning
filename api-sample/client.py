import requests
import streamlit as st


def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke", 
        json={'input':{'topic': input_text}})

    return response.json()['output']['content']

    
def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", 
        json={'input':{'topic': input_text}})

    return response.json()['output']['content']

st.title("Langchain demo with api")
input_text_openai = st.text_input("Essay Topic:")
input_text_ollama = st.text_input("Poem Topic:")

if input_text_openai:
    st.write(get_openai_response(input_text_openai))
    
if input_text_ollama:
    st.write(get_ollama_response(input_text_ollama))