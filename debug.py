# debug_graph.py

from agent.graph import build_graph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()  # loads variables from .env

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

graph = build_graph(llm)

# Get graph structure
print(graph.get_graph().draw_mermaid())