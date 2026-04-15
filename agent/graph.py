from langgraph.graph import StateGraph
from agent.state import GraphState  

from agent.nodes import (
    validate_input,
    check_language_support,
    retrieve_docs,
    filter_relevant_docs,
    generate_answer,
    extract_citations,
    validate_citations
)

def route_language(state):
    return "unsupported" if state.get("unsupported") else "supported"

def check_valid(state):
    return "retry" if not state.get("is_valid", False) else "end"


def build_graph(llm):
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("validate", validate_input)
    graph.add_node("check_language", check_language_support)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("gracefully", retrieve_docs)
    graph.add_node("filter", filter_relevant_docs)
    graph.add_node("generate", lambda s: generate_answer(s, llm))
    graph.add_node("citations", extract_citations)
    graph.add_node("validate_citations", validate_citations)

    # Flow
    graph.set_entry_point("validate")

    graph.add_edge("validate", "check_language")

    graph.add_conditional_edges(
        "check_language",
        route_language,
        {
            "unsupported": "__end__",
            "supported": "retrieve"
        }
    )    
    
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "generate")
    graph.add_edge("generate", "citations")
    graph.add_edge("citations", "validate_citations")

    # Retry loop
    graph.add_conditional_edges(
        "validate_citations",
        check_valid,
        {
            "retry": "generate",
            "end": "__end__"
        }
    )

    return graph.compile()