"""
Module 2 Bridge Project - CrewAI STARTER (hierarchical)

Same job as the LangGraph version. Build that one FIRST, then come back here.

WHAT IS DIFFERENT
    In LangGraph you drew the branch yourself, in one line you could point
    at. A hierarchical crew has no such line. Instead you hire a MANAGER and
    let it decide who works, in what order, and when the job is done.

    You are trading a decision you can READ for a decision you DELEGATE.

    Run both versions on the nonsense claim afterwards and watch carefully.
    In LangGraph, refusing is a code path - report_gap physically cannot
    produce an answer. Here, refusing is an instruction in a backstory. You
    are asking the manager nicely.

    Whether it listens is the most interesting result in this project, and
    it goes in your README either way.

Run:
    python research_crewai_starter.py "Anthropic was founded by ex-OpenAI staff"
    python research_crewai_starter.py "the flurbotron 9000 was released in 2019"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# CrewAI asks about execution traces on the first run in a new folder and
# blocks for 20 seconds waiting for an answer. This stops that.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from dotenv import find_dotenv, load_dotenv  # noqa: E402

try:
    from crewai.events.listeners.tracing.utils import mark_first_execution_done

    mark_first_execution_done()
except Exception:      # different CrewAI version - the prompt times out anyway
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search_tools import web_search  # noqa: E402

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found.\n"
        "Create a file named .env in the project folder containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
        "then run this script again."
    )


# TODO 1 - build the model client. Same model as your LangGraph version, so
       #   the comparison is honest. Note the "openrouter/" prefix.

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)


# ---------------------------------------------------------------------------
# THE TOOL
# ---------------------------------------------------------------------------
# In LangGraph the tool was a plain function call inside a node - YOU decided
# when it ran. Here you hand the tool to an agent and the agent decides.
#
# TODO 2 - finish the docstring below.
#
#          The docstring is NOT a comment. It is the description the model
#          reads when deciding whether to use this tool, and what to pass it.
#          Write it for the model, not for yourself.
#
#          It should say what the tool searches, that it returns findings with
#          source URLs, and that it can return the exact text NO_RESULTS or
#          SEARCH_UNAVAILABLE.
@tool("Web Search")
def search_web(query: str) -> str:
    """
    Use this tool to search the web for information. 
    It takes a short search query and returns a plain-text block of findings along with their source URLs. 
    If the search succeeds but finds nothing useful, it returns the exact text 'NO_RESULTS'. 
    If the search service is unreachable, it returns the exact text 'SEARCH_UNAVAILABLE'.
    """
    return web_search(query)


def build_crew() -> Crew:
    # TODO 3 - the researcher.
    researcher = Agent(
        role="Senior Fact-Checking Researcher",
        goal="Accurately gather verifiable evidence from the web regarding the user's claim.",
        backstory=(
            "You are a meticulous researcher. You rely entirely on the exact findings returned by the Web Search tool. "
            "You always report what the search actually returned and strictly preserve the source URLs. "
            "If the tool returns NO_RESULTS or SEARCH_UNAVAILABLE, you say so plainly and you absolutely NEVER "
            "fill the silence from your own memory or invent facts."
        ),
        tools=[search_web],
        llm=llm,
        verbose=True,
        max_iter=4,
    )

    # TODO 4 - the writer.
    writer = Agent(
        role="Technical Fact-Checking Writer",
        goal="Draft precise, honest, and evidence-backed answers based strictly on the provided research.",
        backstory=(
            "You are a strict technical writer. You only ever use the information given to you by the researcher. "
            "You strongly prefer to hand back 'we could not verify this' rather than writing something that reads well "
            "but might be wrong. You never invent information or make assumptions."
        ),
        llm=llm,
        verbose=True,
        max_iter=4,
    )

    # -----------------------------------------------------------------------
    # THE TASKS
    # -----------------------------------------------------------------------
    # Notice there is NO agent= on either task. In a hierarchical crew the
    # manager assigns the work. (In a sequential crew, leaving agent= off is
    # an error - try it once and read what CrewAI tells you.)

    # TODO 5 - the research task.
    #          Use {question}, which is filled in by kickoff(inputs=...).
    #          Tell it to use the Web Search tool, report exactly what came
    #          back including source URLs, and NOT to substitute its own
    #          knowledge if the search returned nothing.
    
    research_task = Task(
         description="Find solid evidence to verify the following claim: {question}. You must use the Web Search tool to find this information. Report exactly what came back including source URLs, and do NOT substitute your own knowledge if the search returned nothing.",
         expected_output="A detailed report of the search findings, including the exact evidence and source URLs. If no results are found, state plainly that the search returned nothing.",
    )

    # TODO 6 - the writing task.
    write_task = Task(
        description=(
            "Write the final output for the claim: {question} based ONLY on the evidence provided by the researcher. "
            "Begin your final response with either 'VERDICT: ANSWERED' or 'VERDICT: COULD NOT VERIFY'."
        ),
        expected_output=(
            "If the evidence supports an answer: A paragraph under 180 words, quoting at least one source URL. "
            "If the evidence does NOT support an answer: No answer at all. Instead, provide exactly one sentence saying it could not be verified, "
            "one or two bullet points on what is missing from the evidence, and a final line in exactly this form: 'NEXT SEARCH: <the one query you would run next>'."
        ),
    )

    # TODO 7 - a HIERARCHICAL crew.
    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.hierarchical,   # Changed to hierarchical
        manager_llm=llm,                # Added manager LLM
        verbose=True,
        tracing=False,
    )


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or \
        "Anthropic was founded by former OpenAI employees"

    result = str(build_crew().kickoff(inputs={"question": question}))

    print("\n" + "=" * 70)
    print(f"QUESTION : {question}")
    print("=" * 70)
    print(result)

    # TODO 8 - log which way the crew went.
    if "VERDICT: ANSWERED" in result:
        route = "ANSWERED"
    elif "VERDICT: COULD NOT VERIFY" in result:
        route = "GAP REPORTED"
    else:
        route = "UNKNOWN (The model did not follow the exact formatting instruction)"

    print("\n" + "=" * 70)
    print(f"ROUTE : {route}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
