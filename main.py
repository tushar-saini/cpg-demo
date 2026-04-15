# main.py

from agent.graph import build_graph
from chromadb.utils import embedding_functions
import chromadb

# --- LLM (replace with your actual one)
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOllama

from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")

def get_llm():
    # return ChatOllama(
    #     model="llama3",
    #     temperature=0
    # )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def main():
    llm = get_llm()

    graph = build_graph(llm)

    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break

        state = {
            "question": question,
            "country": "A",
            "language": "en",
            "retry_count": 0 
        }

        result = graph.invoke(state)

        print("\nAnswer:", result.get("answer"))
        print("\nCitations:", result.get("citations"))
        print("\nValid:", result.get("is_valid"))


if __name__ == "__main__":
    main()