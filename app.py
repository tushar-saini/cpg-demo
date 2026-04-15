from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import build_graph
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
import os
import time

load_dotenv()  # loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = 'gpt-4o-mini'

# -----------------------
# Request Schema
# -----------------------
class QueryRequest(BaseModel):
    question: str
    country: str
    language: str


# -----------------------
# App Init
# -----------------------
app = FastAPI(title="Customer Content QA API")


def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0
    )


# Build graph
llm = get_llm()
graph = build_graph(llm)


# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# Main Endpoint
# -----------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        start = time.time()
        state = {
            "question": request.question,
            "country": request.country,
            "language": request.language
        }

        result = graph.invoke(state)
        if type(result.get("answer") ) == str:
            answer = result.get("answer")
        else:
            answer = result.get("answer").content
        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "answer": answer,
            "language_used": "TO DO",
            "citations": result.get("citations", []),
            "trace": {"latency_ms": latency_ms, "model": MODEL_NAME}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))