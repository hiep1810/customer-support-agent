# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from .customer_support import (
    run_customer_support,
    categorize_query,
    analyze_sentiment_query,
    CategoryOut,
    SentimentOut,
)

# load .env
load_dotenv()

class QueryIn(BaseModel):
    query: str

app = FastAPI(
    title="LangGraph Gemini Support API",
    version="1.0"
)

@app.post("/categorize", response_model=CategoryOut)
def categorize_endpoint(body: QueryIn):
    """POST /categorize -> {category}"""
    return categorize_query(body.query)


@app.post("/sentiment", response_model=SentimentOut)
def sentiment_endpoint(body: QueryIn):
    """POST /sentiment -> {sentiment}"""
    return analyze_sentiment_query(body.query)


@app.post("/support")
def support_endpoint(body: QueryIn):
    """POST /support -> {category, sentiment, response}"""
    return run_customer_support(body.query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
