"""
Module 2 Bridge Project - LangGraph STARTER
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search_tools import NO_RESULTS, SEARCH_UNAVAILABLE, web_search  # noqa: E402

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")
base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found.\n"
        "Create a file named .env in the project folder containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
        "then run this script again."
    )

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url=base_url,
    api_key=api_key,
)

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class DeskState(TypedDict):
    question: str      
    query: str         
    evidence: str      
    verdict: str       
    reasoning: str     
    output: str        
    route_taken: str   


def plan_query(state: DeskState) -> DeskState:
    prompt = f"""
    You are a fact-checker. Convert the following claim into ONE short, highly effective web search query to verify it.
    Return ONLY the exact search query and absolutely nothing else.
    
    Claim: {state['question']}
    """
    response = llm.invoke(prompt)
    state["query"] = response.content.strip().replace('"', '').replace("'", "")
    return state


def run_search(state: DeskState) -> DeskState:
    state["evidence"] = web_search(state["query"])
    return state


def assess(state: DeskState) -> DeskState:
    if state["evidence"] == SEARCH_UNAVAILABLE:
        state["verdict"] = "NOT_ENOUGH"
        state["reasoning"] = "Search service could not be reached."
        return state
        
    if state["evidence"] == NO_RESULTS:
        state["verdict"] = "NOT_ENOUGH"
        state["reasoning"] = "Search ran successfully but found absolutely nothing."
        return state

    prompt = f"""
    You are a fact-checker assessing search results. 
    Does the evidence provide enough solid information to verify or answer the question?
    
    Question/Claim: {state['question']}
    Evidence: {state['evidence']}
    
    Provide 1-3 short bullets explaining your reasoning, then write exactly one word on the final line: ENOUGH or NOT_ENOUGH.
    """
    
    response = llm.invoke(prompt)
    state["reasoning"] = response.content.strip()
    state["verdict"] = read_verdict(state["reasoning"])
    return state


def read_verdict(raw: str) -> str:
    lines = [line.strip() for line in raw.split('\n') if line.strip()]
    if not lines:
        return "NOT_ENOUGH" 
        
    last_line = lines[-1].upper()
    
    if "NOT_ENOUGH" in last_line:
        return "NOT_ENOUGH"
    if "ENOUGH" in last_line:
        return "ENOUGH"
        
    return "NOT_ENOUGH"


def choose_next(state: DeskState) -> Literal["write_answer", "report_gap"]:
    if state["verdict"] == "ENOUGH":
        return "write_answer"
    return "report_gap"


def write_answer(state: DeskState) -> DeskState:
    prompt = f"""
    Answer the following question using ONLY the provided evidence.
    Rules:
    - Every claim must come from the evidence, invent nothing.
    - Quote at least one source URL.
    - If part of the question is not covered, say so.
    - Keep the answer under 180 words.

    Question: {state['question']}
    Evidence: {state['evidence']}
    """
    response = llm.invoke(prompt)
    state["output"] = response.content.strip()
    state["route_taken"] = "ANSWERED"
    return state


def report_gap(state: DeskState) -> DeskState:
    prompt = f"""
    You could not verify the user's question with the provided evidence. 
    You must NOT answer the question.
    
    Produce exactly this format:
    - One sentence saying plainly this could not be verified.
    - One or two bullets on what is missing from the evidence.
    - A final line in exactly this form: NEXT SEARCH: <the one query you would run next>

    Question: {state['question']}
    Evidence: {state['evidence']}
    """
    response = llm.invoke(prompt)
    state["output"] = response.content.strip()
    state["route_taken"] = "GAP REPORTED"
    return state


def build_graph():
    graph = StateGraph(DeskState)

    graph.add_node("plan_query", plan_query)
    graph.add_node("run_search", run_search)
    graph.add_node("assess", assess)
    graph.add_node("write_answer", write_answer)
    graph.add_node("report_gap", report_gap)

    graph.set_entry_point("plan_query")
    graph.add_edge("plan_query", "run_search")
    graph.add_edge("run_search", "assess")

    graph.add_conditional_edges(
        "assess",              
        choose_next,           
        {                      
            "write_answer": "write_answer",
            "report_gap":   "report_gap",
        },
    )

    graph.add_edge("write_answer", END)
    graph.add_edge("report_gap", END)

    return graph.compile()


def run(question: str) -> DeskState:
    return build_graph().invoke({
        "question": question,
        "query": "",
        "evidence": "",
        "verdict": "",
        "reasoning": "",
        "output": "",
        "route_taken": "",
    })


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or \
        "Anthropic was founded by former OpenAI employees"

    result = run(question)

    print("=" * 70)
    print(f"QUESTION : {result['question']}")
    print("=" * 70)
    print(f"\n[1] SEARCH QUERY\n{result['query']}")
    print(f"\n[2] EVIDENCE ({len(result['evidence'])} chars)")
    print(result["evidence"][:700])
    print(f"\n[3] ASSESSMENT\n{result['reasoning']}")
    print("\n" + "=" * 70)
    print(f"VERDICT : {result['verdict']}")
    print(f"ROUTE   : {result['route_taken'].upper()}")
    print("=" * 70)
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
