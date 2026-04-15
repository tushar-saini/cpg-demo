# app/agent/nodes.py
from agent.state import GraphState
from retrieval.query import query_docs
import re

SUPPORTED_LANGUAGES = {
    "A": ["en", "hi"],
    "B": ["en", "es"],
    "C": ["en", "fr_CA"],
    "D": ["en"]
}

def split_sentences(text):
    # simple sentence splitter
    return re.split(r'(?<=[.!?])\s+', text)


def find_best_excerpt(answer, chunk_text):
    sentences = split_sentences(chunk_text)

    best_sentence = None
    best_score = 0

    for sent in sentences:
        # simple overlap score
        overlap = len(set(sent.lower().split()) & set(answer.lower().split()))
        if overlap > best_score:
            best_score = overlap
            best_sentence = sent

    return best_sentence if best_sentence else chunk_text

    
def validate_input(state: GraphState):
    if not state["question"]:
        raise ValueError("Question is empty")

    return state

def check_language_support(state):
    country = state["country"]
    language = state["language"]

    supported = SUPPORTED_LANGUAGES.get(country, [])

    if language not in supported:
        return {
            **state,
            "answer": f"Language '{language}' is not supported for Country {country}. Supported languages are: {', '.join(supported)}.",
            "citations": [],
            "is_valid": False,
            "unsupported": True   # 🔥 flag for graph routing
        }

    return {
        **state,
        "unsupported": False
    }

def retrieve_docs(state: GraphState):
    docs = query_docs(
        question=state["question"],
        country=state["country"],
        language=state["language"],
        k=3
    )

    return {
        **state,
        "retrieved_docs": docs
    }

def filter_relevant_docs(state: GraphState):
    docs = state["retrieved_docs"]

    # Step 1: distance filtering
    docs = [d for d in docs if d["distance"] < 0.7]

    # Step 2: simple safeguard
    if not docs:
        return {
            **state,
            "filtered_docs": []
        }

    return {
        **state,
        "filtered_docs": docs
    }


def generate_answer(state: GraphState, llm):
    docs = state["filtered_docs"]

    if not docs:
        return {
            **state,
            "answer": "No relevant information found."
        }

    context = "\n".join([d["text"] for d in docs])

    retry_count = state.get("retry_count", 0)

    prompt = f"""
        You are a strict assistant.

        Answer ONLY using the provided context.
        If the answer is not present in the context, respond with "Not found".

        Language Rules:
        - Detect the language of the user's question.
        - Respond in the SAME language as the question.
        - Do NOT translate the answer into another language unless explicitly asked.
        - Do NOT include English translations if the question is in a non-English language.

        Answering Rules:
        - Use only the provided context.
        - Do NOT hallucinate or add external information.
        - Include all relevant conditions and exceptions from the context.
        - Keep the answer concise and precise.
        
        Context:
        {context}
        
        Question:
        {state['question']}
        
        Rules:
        - Do NOT hallucinate
        - Include all relevant conditions and exceptions
    """

    answer = llm.invoke(prompt)

    return {
        **state,
        "answer": answer,
        "retry_count": retry_count 
    }

def extract_citations(state):
    docs = state.get("filtered_docs", [])
    answer = state.get("answer", "")

    citations = []

    for d in docs:
        chunk_text = d["text"]

        excerpt = find_best_excerpt(answer.content, chunk_text)

        citations.append({
            "content_id": d["metadata"]["content_id"],
            "type": d["metadata"]["type"],
            "excerpt": excerpt.strip(),
            "match_score": d['distance']  # optional (can remove later)
        })

    return {
        **state,
        "citations": citations
    }

def validate_citations(state, llm):
    """
    Validates that:
    1. Citations exist
    2. Citations belong to retrieved documents
    3. Answer is grounded in context (LLM-based check)
    """

    answer = state.get("answer", "")
    docs = state.get("filtered_docs", [])
    citations = state.get("citations", [])

    # -----------------------------
    # Step 1: Basic sanity checks
    # -----------------------------
    if not answer or not docs:
        return {**state, "is_valid": False, "retry_count": state.get("retry_count", 0) + 1}

    # -----------------------------
    # Step 2: Citation presence check
    # -----------------------------
    if not citations:
        return {**state, "is_valid": False, "retry_count": state.get("retry_count", 0) + 1}

    # -----------------------------
    # Step 3: Build context
    # -----------------------------
    context = "\n\n".join([
        f"[{d['metadata']['content_id']}] {d['text']}"
        for d in docs
    ])

    # -----------------------------
    # Step 4: LLM grounding check
    # -----------------------------
    prompt = f"""
        You are a strict evaluator.

        Answer:
        {answer}

        Context:
        {context}

        Task:
        - Verify if the answer is fully supported by the context.
        - Every claim in the answer must be grounded in the context.
        - If any part is unsupported or hallucinated, mark INVALID.

        Respond ONLY with:
        VALID or INVALID
    """

    try:
        response = llm.invoke(prompt)
        verdict = response.content.strip().upper()

        is_valid = verdict == "VALID"

    except Exception:
        # fail-safe: mark invalid if LLM check fails
        is_valid = False

    # -----------------------------
    # Step 6: Return updated state
    # -----------------------------
    if is_valid:
        return {
            **state,
            "is_valid": is_valid
        }
    else:
        return {
            **state,
            "is_valid": is_valid,
            "retry_count": state.get("retry_count", 0) + 1
        }