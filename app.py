# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from customer_support import run_customer_support

# load .env
load_dotenv()

class QueryIn(BaseModel):
    query: str

app = FastAPI(
    title="LangGraph Gemini Support API",
    version="1.0"
)

@app.post("/support")
def support_endpoint(body: QueryIn):
    """POST /support -> {category, sentiment, response}"""
    return run_customer_support(body.query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
