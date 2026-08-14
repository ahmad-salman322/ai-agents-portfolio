"""
Module 2 Phase 1 Mini-Project — LangGraph
Build the same three-task Study Guide Agent in LangGraph and CrewAI.
"""

from __future__ import annotations
import os
import sys
from typing import TypedDict
from dotenv import find_dotenv, load_dotenv
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

# Ensure we find the .env file no matter where we run this script from
load_dotenv()                          
load_dotenv(find_dotenv(usecwd=True))  

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Please check your .env file.")

# Set up the model
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Define the data structure that gets passed between tasks
class StudyGuideState(TypedDict):
    topic: str
    explanation: str
    example: str
    quiz: str

def explain_topic(state: StudyGuideState) -> StudyGuideState:
    """Task 1: Simplify and explain the topic"""
    prompt = f"Explain the topic '{state['topic']}' in 2-3 plain-language sentences for a beginner. Do not invent statistics."
    response = llm.invoke(prompt)
    
    # Keep the old state data, just append the new explanation
    return {**state, "explanation": response.content}

def create_example(state: StudyGuideState) -> StudyGuideState:
    """Task 2: Provide a real-world example and clear up a common misconception"""
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Explanation: {state['explanation']}\n"
        f"Based on the explanation, provide one practical example and one common misconception. Distinguish clearly between them."
    )
    response = llm.invoke(prompt)
    
    return {**state, "example": response.content}

def create_quiz(state: StudyGuideState) -> StudyGuideState:
    """Task 3: Generate a quick 3-question quiz"""
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Explanation: {state['explanation']}\n"
        f"Example: {state['example']}\n"
        f"Create exactly 3 quiz questions based on this material, followed by an answer key."
    )
    response = llm.invoke(prompt)
    
    return {**state, "quiz": response.content}

def build_graph():
    graph = StateGraph(StudyGuideState)

    # Register all our functions as Nodes
    graph.add_node("explain_topic", explain_topic)
    graph.add_node("create_example", create_example)
    graph.add_node("create_quiz", create_quiz)

    # Map out the flow step-by-step
    graph.set_entry_point("explain_topic")
    graph.add_edge("explain_topic", "create_example")
    graph.add_edge("create_example", "create_quiz")
    graph.add_edge("create_quiz", END)

    return graph.compile()

def run_study_guide(topic: str) -> StudyGuideState:
    app = build_graph()
    
    # Initialize an empty state backpack to fill during the run
    initial_state: StudyGuideState = {
        "topic": topic,
        "explanation": "",
        "example": "",
        "quiz": "",
    }
    return app.invoke(initial_state)

if __name__ == "__main__":
    # Grab the topic from the terminal, or fallback to a default one
    topic = " ".join(sys.argv[1:]).strip() or "Model Context Protocol"
    result = run_study_guide(topic)

    # Print the final output cleanly in Markdown format
    print(f"# Study Guide: {result['topic']}\n")
    print("## Explanation\n", result["explanation"])
    print("\n## Example and misconception\n", result["example"])
    print("\n## Quiz\n", result["quiz"])
