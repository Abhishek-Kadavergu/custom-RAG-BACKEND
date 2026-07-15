# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAGPipeline
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")





rag = RAGPipeline(api_key=GROQ_API_KEY)



class QueryRequest(BaseModel):
    question: str


# Root route
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


# Query route
@app.post("/ask")
def ask_question(request: QueryRequest):
    response = rag.ask(request.question)
    return {"answer": response}
