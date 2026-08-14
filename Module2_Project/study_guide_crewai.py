"""
Module 2 Phase 1 Mini-Project — CrewAI
Build one Agent with three sequential Tasks.
"""

from __future__ import annotations

import os
import sys
from dotenv import find_dotenv, load_dotenv
from crewai import Agent, Crew, LLM, Process, Task


load_dotenv()                          
load_dotenv(find_dotenv(usecwd=True))  

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Please check your .env file.")

# Set up the model
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)

def build_crew() -> Crew:
    # Build a "Patient Teacher" agent to create solid educational material
    teacher = Agent(
        role="Patient Study Guide Teacher",
        goal="Create accurate, understandable learning materials for students.",
        backstory="You are an expert educator. You break down complex topics simply, and you NEVER invent facts or statistics.",
        llm=llm,          
        verbose=True,
    )

    # Task 1: Explanation
    explain_task = Task(
        description="Explain {topic} in 2-3 plain-language sentences. Do not invent statistics.",
        expected_output="A concise 2-3 sentence plain-language explanation.",
        agent=teacher,
    )

    # Task 2: Example
    # The magic here is passing the first task in the context so it builds on it
    example_task = Task(
        description="Create a practical example and a common misconception for {topic}.",
        expected_output="One practical example and one common misconception, clearly distinguished.",
        agent=teacher,
        context=[explain_task], # Pass context from the previous task
    )

    # Task 3: Quiz and Final Assembly
    # Give it all previous work so it can compile the complete study guide
    quiz_task = Task(
        description=(
            "Assemble the complete study guide for {topic}. Include the "
            "explanation, the practical example and misconception, then exactly "
            "three questions followed by a matching answer key."
        ),
        expected_output=(
            "A complete guide formatted with headers: '## Explanation', '## Example and misconception', "
            "and '## Quiz'."
        ),
        agent=teacher,
        context=[explain_task, example_task], # Pass everything done so far
    )

    # Assign tasks to the crew and run them sequentially
    return Crew(
        agents=[teacher],
        tasks=[explain_task, example_task, quiz_task],
        process=Process.sequential,
        verbose=True,
        tracing=False,
    )

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]).strip() or "Model Context Protocol"
    crew = build_crew()
    
    result = crew.kickoff(inputs={"topic": topic})
    print("\n\n" + "="*50 + "\nFINAL STUDY GUIDE\n" + "="*50)
    print(result)
