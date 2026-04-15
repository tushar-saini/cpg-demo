from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    # --- Input ---
    question: str
    country: str
    language: str

    # --- Retrieval ---
    retrieved_docs: List[Dict]
    filtered_docs: List[Dict]

    # --- Generation ---
    answer: str

    # --- Citations ---
    citations: List[Dict]

    # --- Validation ---
    is_valid: bool