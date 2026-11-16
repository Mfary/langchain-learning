from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Chat server",
    version="1.0",
    description="Api sample for langchain"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model_openai=ChatOpenAI()
model_ollama=ChatOllama(model="qwen3.5")

prompt_essay = ChatPromptTemplate.from_template("wrtie an essay about {topic} with 1000 words")
prompt_poem = ChatPromptTemplate.from_template("wrtie an poem about {topic} with 1000 words")


add_routes(
    app,
    prompt_essay|model_openai,
    path="/essay"
)

add_routes(
    app,
    prompt_poem|model_ollama,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)