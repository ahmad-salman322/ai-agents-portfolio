"""
Module 3 Lab — STARTER
ReAct + Planning agent in LangGraph with STRUCTURED OUTPUTS.

  >>> READ M3-Lab-Brief.md FIRST <<<
  It explains what you are building and why, in ten minutes.
  Then follow M3-Lab-Worksheet.md, which walks these TODOs in order.

WHAT YOU ARE BUILDING
  An agent that WORKS OUT numeric answers instead of guessing them:

      REASON  -> the model reads the question + everything found so far,
                 and decides ONE next move
      ACT     -> your Python runs the tool it asked for (no AI here)
      OBSERVE -> the result is written to state["scratchpad"]
      ... loop until it has enough to answer, or the budget runs out.

  Part 1 (Steps 1-5): that loop, as a LangGraph state graph.
  Part 2 (Steps 6-9): add a planner that breaks the goal into steps first.

THE ONE THING TO REMEMBER
  The model has NO memory between calls. It only knows what you put in the
  prompt. state["scratchpad"] IS the memory - if something isn't in there,
  the model cannot see it.

Fill in the TODOs. No manual JSON parsing — use with_structured_output().
Set your key in .env (OPENROUTER_API_KEY=sk-or-...).
"""

import os
from typing import TypedDict, Literal, Optional
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
# from langchain_openai import ChatOpenAI

load_dotenv()

MAX_STEPS = 8
MAX_REPLANS = 5

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],  # add `import os`
)


# ---------------------------------------------------------------------------
# Tool: calculator (reuse M1 — safe, returns error string instead of raising)
# ---------------------------------------------------------------------------
_ALLOWED = re.compile(r"^[0-9+\-*/().\s]+$")


def calculator(expr: str) -> str:
    if not _ALLOWED.match(expr or ""):
        return "ERROR: Invalid arithmetic expression."

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Schemas (structured outputs)
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """Step 2: the schema-validated action the model must return.

    NOTE: every field is explicitly typed. Do NOT use `args: dict` - strict
    structured-output mode requires additionalProperties:false on every object,
    and a bare dict cannot express that (the provider returns HTTP 400).
    """

    tool: Literal["calculator", "final_answer"]
    expr: Optional[str] = Field(
        default=None, description="Arithmetic expression, when tool='calculator'."
    )
    text: Optional[str] = Field(
        default=None, description="The answer text, when tool='final_answer'."
    )


class Plan(BaseModel):
    """Step 6: an ordered list of concrete, tool-executable steps."""

    steps: list[str]


class Replan(BaseModel):
    decision: Literal["finish", "continue"]
    answer: Optional[str] = Field(
        default=None, description="The final answer text, when decision='finish'."
    )
    steps: list[str] = Field(
        default_factory=list,
        description="The remaining steps still to run, when decision='continue'.",
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class State(TypedDict):
    question: str
    scratchpad: list
    action: Optional[dict]
    answer: Optional[str]
    plan: list
    past_steps: list
    steps_used: int  # Step 5: the MAX_STEPS budget counter
    replans: int  # Step 8: the MAX_REPLANS budget counter


# ---------------------------------------------------------------------------
# PART 1 — ReAct nodes
# ---------------------------------------------------------------------------

# def reason(state: State) -> State:
#     scratchpad_text = "\n".join(state.get("scratchpad", [])) if state.get("scratchpad") else "No observations yet."
#
#     prompt = f"""
#     Answer the question below. You must use the calculator tool for arithmetic.
#     Do not do arithmetic in your head.
#
#     Question: {state['question']}
#
#     Observations so far:
#     {scratchpad_text}
#     """
#
#     chosen_action = llm.with_structured_output(Action).invoke(prompt)
#
#     return {
#         "action": chosen_action.model_dump(),
#         "steps_used": state.get("steps_used", 0) + 1
#     }


def reason(state: State) -> State:

    observations = (
        "\n".join(state.get("scratchpad", []))
        if state.get("scratchpad")
        else "(nothing yet)"
    )

    prompt = f"""
    Answer the question below, one step at a time.
    You cannot do arithmetic yourself - use the calculator tool.
    
    RULES:
    - Read the observations FIRST.
    - Never repeat a calculation that is already there; reuse the result.
    - As soon as the observations contain everything you need, use final_answer.
    
    QUESTION: {state["question"]}
    
    OBSERVATIONS SO FAR:
    {observations}
    """

    print(f"\n--- Step {state.get('steps_used', 0) + 1} ---")
    print(f"Observations: {observations}")

    chosen_action = llm.with_structured_output(Action).invoke(prompt)

    print(f"Model decided to use tool: {chosen_action.tool}")
    if chosen_action.expr:
        print(f"Expression: {chosen_action.expr}")
    if chosen_action.text:
        print(f"Text: {chosen_action.text}")

    return {
        "action": chosen_action.model_dump(),
        "steps_used": state.get("steps_used", 0) + 1,
    }


def act(state: State) -> State:
    action = state["action"]

    if action["tool"] == "final_answer":
        return {"answer": action["text"]}

    elif action["tool"] == "calculator":
        expr = action["expr"]
        result = calculator(expr)
        new_observation = f"calculator({expr}) = {result}"
        current_scratchpad = state.get("scratchpad", [])
        updated_scratchpad = current_scratchpad + [new_observation]

        return {"scratchpad": updated_scratchpad}


def is_done(state: State) -> str:

    if state.get("answer") or state.get("steps_used", 0) >= MAX_STEPS:
        return "end"

    return "loop"


# ---------------------------------------------------------------------------
# PART 2 — Planning nodes
# ---------------------------------------------------------------------------
def planner(state: State) -> State:
    # TODO (Step 6): produce a Plan from the question; store steps in state["plan"].

    prompt = f"""
    Break the question below into an ordered list of steps.
    
    RULES:
    - Each step must be ONE concrete action: a single arithmetic operation the
      calculator can run, or a single comparison / statement of the answer.
    - Refer to an earlier result in words, for example "the total from step 1".
    - Never put two operations in one step.
    - Keep the plan as short as possible.
    
    QUESTION: {state["question"]}
    """

    produced_plan = llm.with_structured_output(Plan).invoke(prompt)

    print("\n--- PLAN ---")
    for i, step in enumerate(produced_plan.steps, 1):
        print(f"{i}. {step}")

    return {"plan": produced_plan.steps, "past_steps": []}


def executor(state: State) -> State:
    # TODO (Step 7): take next plan step, run it via ReAct mechanics (reason+act),
    # append (step, observation) to state["past_steps"].

    plan = state.get("plan", [])
    past = state.get("past_steps", [])

    if not plan:
        return {}

    current_step = plan[0]
    context = "\n".join([f"{s} -> {o}" for s, o in past]) if past else "(nothing yet)"

    prompt = f"""
    You are executing ONE step of a plan. Do that step only.
    
    RULES:
    - Use the calculator tool for any arithmetic.
    - The numbers you need may already be in the earlier results below.
      Reuse them, never recompute them, and never invent a number.
    - If the step needs no arithmetic, use final_answer with the short result.
    
    ORIGINAL QUESTION: {state["question"]}
    
    RESULTS OF EARLIER STEPS:
    {context}
    
    CURRENT STEP: {current_step}
    """

    print(f"\n--- Executing: {current_step} ---")
    print(f"Earlier results: {context}")

    chosen_action = llm.with_structured_output(Action).invoke(prompt)

    if chosen_action.tool == "calculator":
        expr = chosen_action.expr
        observation = f"calculator({expr}) = {calculator(expr)}"
    else:
        observation = chosen_action.text or "(no result)"

    print(f"Observation: {observation}")

    return {
        "plan": plan[1:],
        "past_steps": past + [(current_step, observation)],
        "scratchpad": state.get("scratchpad", []) + [observation],
        "steps_used": state.get("steps_used", 0) + 1,
    }


def replan(state: State) -> State:
    # TODO (Step 8): decide finish (set state["answer"]) or update remaining plan.

    replans = state.get("replans", 0) + 1
    past = state.get("past_steps", [])
    remaining = state.get("plan", [])
    progress = "\n".join([f"{s} -> {o}" for s, o in past]) if past else "(nothing yet)"
    remaining_text = "\n".join(remaining) if remaining else "(none left)"

    print(f"\n--- Replan {replans} ---")

    if replans > MAX_REPLANS or state.get("steps_used", 0) >= MAX_STEPS:
        print("Budget exhausted - stopping with what we have.")
        return {
            "replans": replans,
            "answer": f"Could not finish within the budget. Progress so far:\n{progress}",
        }

    prompt = f"""
    You are managing the plan below. Decide whether the work is finished.
    
    RULES:
    - Choose "finish" only if the results below already answer the whole
      question, and write the final answer using only those numbers.
    - Choose "continue" if work is left, and return the steps that are still
      to be done. You may rewrite them, but never repeat finished work.
    - Never invent a number that is not in the results.
    
    QUESTION: {state["question"]}
    
    RESULTS SO FAR:
    {progress}
    
    REMAINING STEPS:
    {remaining_text}
    """

    decision = llm.with_structured_output(Replan).invoke(prompt)

    print(f"Decision: {decision.decision}")

    if decision.decision == "finish" or not (decision.steps or remaining):
        return {
            "replans": replans,
            "answer": decision.answer
            or f"Could not produce an answer. Progress so far:\n{progress}",
        }

    return {"replans": replans, "plan": decision.steps or remaining}


def replan_done(state: State) -> str:
    # TODO (Step 8): "end" if answer else "loop"

    if state.get("answer"):
        return "end"

    return "loop"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_react_graph():
    """Part 1 graph: reason -> act -> (loop|END)."""
    g = StateGraph(State)
    g.add_node("reason", reason)
    g.add_node("act", act)
    g.set_entry_point("reason")
    g.add_edge("reason", "act")
    g.add_conditional_edges("act", is_done, {"end": END, "loop": "reason"})
    return g.compile()


def build_plan_execute_graph():
    """Part 2 graph: planner -> executor -> replan -> (loop|END)."""
    g = StateGraph(State)
    # TODO (Steps 6-8): add planner, executor, replan nodes; wire edges + conditional.
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("replan", replan)
    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "replan")
    g.add_conditional_edges("replan", replan_done, {"end": END, "loop": "executor"})
    return g.compile()


# if __name__ == "__main__":
#     q = "What is (23 * 19) + 100?"
#
#     print("Module 3 setup check")
#     print("-" * 40)
#
#     ok = True
#     try:
#         import langgraph
#         print("  [ok]   langgraph imported")
#     except Exception as e:
#         ok = False
#         print(f"  [FAIL] langgraph: {e}")
#
#     try:
#         from langchain_openai import ChatOpenAI  # noqa: F401
#         print("  [ok]   langchain_openai imported")
#     except Exception as e:
#         ok = False
#         print(f"  [FAIL] langchain_openai: {e}")
#
#     if os.environ.get("OPENROUTER_API_KEY"):
#         print("  [ok]   OPENROUTER_API_KEY found")
#     else:
#         ok = False
#         print("  [FAIL] OPENROUTER_API_KEY missing - check your .env file")
#
#     print("-" * 40)
#     if ok:
#         print("READY")
#         print("  1. read M3-Lab-Brief.md      (what you're building, and why)")
#         print("  2. then M3-Lab-Worksheet.md  (Step 1 onwards)")
#     else:
#         print("NOT READY - fix the [FAIL] lines above, then run this again.")
#
#
#
#
#
if __name__ == "__main__":
    q = (
        "A team has 3 sprints of 12, 19, and 8 story points. "
        "What's the average per sprint, and is it above 12?"
    )

    app = build_plan_execute_graph()

    result = app.invoke(
        {
            "question": q,
            "scratchpad": [],
            "action": None,
            "answer": None,
            "plan": [],
            "past_steps": [],
            "steps_used": 0,
            "replans": 0,
        }
    )

    print("ANSWER:", result.get("answer"))

    print(app.get_graph().draw_mermaid())
