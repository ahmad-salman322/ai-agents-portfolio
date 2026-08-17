# Module 2 Bridge Project: The Research Desk (LangGraph vs. CrewAI)

The same job, built twice.

A user hands the agent a claim. The agent turns it into a search query, searches the
web, judges whether the evidence is actually good enough, and then does **one of two
things**: writes a sourced answer, or refuses and reports the gap.

The interesting part is not the answer. It is the refusal — and *where the decision to
refuse physically lives* in each framework.

## Project Structure

| File | What it is |
| --- | --- |
| `research_langgraph_starter.py` | LangGraph version. Explicit state machine with a conditional edge. |
| `research_crewai_starter.py` | CrewAI version. Hierarchical crew — a manager LLM assigns the work. |
| `search_tools.py` | Shared web search tool (DuckDuckGo + Wikipedia, no API key needed). Pre-written. |
| `requirements.txt` | Dependencies. |

Both versions use the **same model** (`gpt-4o-mini` via OpenRouter) and the **same
search tool**, so the comparison is honest. The only variable is the framework.

## Setup

```bash
# 1. Activate the virtual environment (it lives at the portfolio root)
source ~/ai-agents-portfolio/.venv/bin/activate

# 2. Confirm you are on the right interpreter — this must print a path under .venv
which python

# 3. Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in this folder containing:

```
OPENROUTER_API_KEY=sk-or-...
```

Optionally confirm search works from your machine before you blame your agent:

```bash
python search_tools.py --selftest
```

## Run

```bash
# The real claim
python research_langgraph_starter.py "Anthropic was founded by ex-OpenAI staff"
python research_crewai_starter.py    "Anthropic was founded by ex-OpenAI staff"

# The nonsense claim
python research_langgraph_starter.py "the flurbotron 9000 was released in 2019"
python research_crewai_starter.py    "the flurbotron 9000 was released in 2019"
```

## The two architectures

**LangGraph** — the branch is a line of code you can point at:

```
plan_query → run_search → assess → ┬─ ENOUGH ────→ write_answer → END
                                   └─ NOT_ENOUGH → report_gap   → END
```

`choose_next()` reads `state["verdict"]` and returns a node name. Refusing is a **code
path**: `report_gap` has no access to a "write the answer" prompt, so it physically
cannot produce one.

**CrewAI** — there is no such line. A `manager_llm` receives two agents and two tasks
under `Process.hierarchical` and decides who works, in what order, and when the job is
done. Refusing is an **instruction in a backstory**. You are asking the manager nicely.

## Results

Single run of each, `temperature=0`, same model, same tool:

| Claim | LangGraph | CrewAI |
| --- | --- | --- |
| "Anthropic was founded by ex-OpenAI staff" | **GAP REPORTED** | **ANSWERED** (cites Wikipedia) |
| "the flurbotron 9000 was released in 2019" | **GAP REPORTED** | **GAP REPORTED** |

Neither version invented a release date for the flurbotron 9000. Both refused, and both
produced a `NEXT SEARCH:` line. On the question the project was actually built to ask —
*does a backstory instruction hold as well as a code path?* — it held.

### The divergence is not about refusal discipline

The row that disagrees is the **true** claim, and LangGraph is the one that got it
wrong. That is worth being precise about, because it is the opposite of what the setup
leads you to expect.

LangGraph planned the query `Anthropic founded by ex-OpenAI staff` and the search came
back with 1560 characters about **Leopold Aschenbrenner** — a real ex-OpenAI researcher,
entirely the wrong person. `assess` read that evidence and reasoned correctly:

> There is no mention of Anthropic or its founders in the provided evidence, which is
> necessary to substantiate the claim.

So it refused. The refusal was *correct given the evidence it had*. The failure happened
one node earlier, and the graph gave it no way back — `run_search` runs exactly once,
and every downstream node is stuck with whatever it returned.

CrewAI's researcher has `max_iter=4` and holds the tool itself, so when a search comes
back unhelpful it can reformulate and search again. It reached the Anthropic Wikipedia
page and answered correctly.

For the flurbotron, LangGraph's evidence was literally `NO_RESULTS` (10 chars) and
`assess` short-circuited before ever calling the model. That is the code path doing its
job: no LLM was given the opportunity to be creative.

## Observations

- **What did LangGraph make explicit?** The state (`DeskState` as a `TypedDict`), the
  control flow (`add_edge`, `add_conditional_edges`), and the refusal itself. I can
  point at line 111 and say "this is where it decides." I also had to write
  `read_verdict()` by hand to parse `ENOUGH` / `NOT_ENOUGH` out of prose, with a
  default of `NOT_ENOUGH` when parsing fails — the safe direction.

- **What did CrewAI automate or hide?** Everything about routing. There is no verdict
  variable, no parser, no branch. The manager decides. I got retry-on-bad-search for
  free, which is exactly what won it the Anthropic row — and I never wrote a retry loop.
  But I also cannot tell you *why* it retried, or guarantee it will next time.

- **Which would I choose?** For this job, LangGraph — but not for the reason I expected.
  Its refusal is auditable and cannot be talked out of, which matters for a
  fact-checker. What this run proved is that the guarantee is narrower than it looks: a
  hard-coded branch protects the decision, not the evidence feeding it. LangGraph's
  single-shot `run_search` is the real weakness, and it is fixable — add an edge from
  `assess` back to `plan_query` with a retry counter in state, and the graph gets
  CrewAI's second attempt while keeping the branch you can point at.

## Caveat

These are single runs against a live web search. `temperature=0` pins the model, not
DuckDuckGo — the same query on a different day can return different pages, and the
LangGraph result on the Anthropic claim in particular depends on which page came back
first. Re-run before quoting the table.
