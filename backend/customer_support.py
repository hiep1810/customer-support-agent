# customer_support.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel

# Load Gemini key
load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Missing GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise EnvironmentError("Missing GEMINI_MODEL")

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# ---------- Schemas
class CategoryOut(BaseModel):
    category: Literal["Technical", "Billing", "General"]

class SentimentOut(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative"]

# Optional: nudge Gemini to JSON mode for extra safety (supported in recent LC)
json_mode = {"response_mime_type": "application/json"}

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    generation_config=json_mode,  # safe to keep; remove if your version errors
    google_api_key=GEMINI_API_KEY,   # <<< important
)

# ---------- 1) Categorization node (strict)
def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "You are a classifier. "
        "Return JSON with a single field 'category' whose value is exactly one of: "
        "Technical, Billing, General. No explanations.\n\nQuery: {query}"
    )
    chain = prompt | model.with_structured_output(CategoryOut)
    out: CategoryOut = chain.invoke({"query": state["query"]})
    return {"category": out.category}

# ---------- 2) Sentiment analysis node (strict)
def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "You are a classifier. "
        "Return JSON with a single field 'sentiment' whose value is exactly one of: "
        "Positive, Neutral, Negative. No explanations.\n\nQuery: {query}"
    )
    chain = prompt | model.with_structured_output(SentimentOut)
    out: SentimentOut = chain.invoke({"query": state["query"]})
    return {"sentiment": out.sentiment}

# ---------- 3) Response generator (kept free-form, but still controlled)
def generate_response(state: State, template: str) -> State:
    """
    Generates a response using the given prompt template.
    """
    prompt = ChatPromptTemplate.from_template(
        template + "\n\nRespond in the user's language and keep it concise."
    )
    chain = prompt | model
    resp = chain.invoke({"query": state["query"]}).content.strip()
    return {"response": resp}

def handle_technical(state: State) -> State:
    """Provide a response for technical support queries."""
    return generate_response(state, "Provide a technical support response: {query}")


def handle_billing(state: State) -> State:
    """Provide a response for billing-related queries."""
    return generate_response(state, "Provide a billing support response: {query}")


def handle_general(state: State) -> State:
    """Provide a response for general support queries."""
    return generate_response(state, "Provide a general support response: {query}")

# 4) Escalation node

def escalate(state: State) -> State:
    return {"response": "Escalated to a human agent due to negative sentiment."}

# 5) Routing logic
def route_query(state: State) -> str:
    if state["sentiment"] == "Negative":
        return "escalate"
    return {
        "Technical": "handle_technical",
        "Billing":  "handle_billing",
    }.get(state["category"], "handle_general")

# --- Build the StateGraph ---
workflow = StateGraph(State)
for name, fn in [
    ("categorize", categorize),
    ("analyze_sentiment", analyze_sentiment),
    ("handle_technical", handle_technical),
    ("handle_billing", handle_billing),
    ("handle_general", handle_general),
    ("escalate", escalate),
]:
    workflow.add_node(name, fn)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing":  "handle_billing",
        "handle_general":  "handle_general",
        "escalate":         "escalate",
    }
)
for end_node in ["handle_technical", "handle_billing", "handle_general", "escalate"]:
    workflow.add_edge(end_node, END)

workflow.set_entry_point("categorize")
app = workflow.compile()

# Utility to expose via API

def run_customer_support(query: str) -> dict:
    res = app.invoke({"query": query})
    return {k: res[k] for k in ("category", "sentiment", "response")}


def categorize_query(query: str) -> dict:
    """Classify the query into a support category."""
    return categorize({"query": query})


def analyze_sentiment_query(query: str) -> dict:
    """Analyze the sentiment of the query."""
    return analyze_sentiment({"query": query})
