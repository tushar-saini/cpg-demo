# evaluation/eval.py

from agent.graph import build_graph
from langchain_openai import ChatOpenAI


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


TEST_CASES = [
    {
        "question": "What is return policy?",
        "country": "A",
        "language": "en"
    },
    {
        "question": "What is return policy?",
        "country": "B",
        "language": "en"
    },
    {
        "question": "How do upgrade my account from preium to VIP?",
        "country": "A",
        "language": "en"
    },
    {
        "question": "La clôture du compte prend effet immédiatement.",
        "country": "C",
        "language": "fr_CA"
    }
    
]


def run_eval():
    llm = get_llm()
    graph = build_graph(llm)

    for i, test in enumerate(TEST_CASES):
        result = graph.invoke(test)

        print(f"\nTest Case {i+1}")
        print("Question:", test["question"])
        print("Answer:", result.get("answer").content)
        print("Valid:", result.get("is_valid"))
        print("Citations:", len(result.get("citations", [])))

