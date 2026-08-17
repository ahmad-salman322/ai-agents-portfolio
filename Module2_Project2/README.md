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
| `search_tools.py` | Shared web search tool (DuckDuckGo + Wikipedia, no API key needed). |
| `requirements.txt` | Dependencies. |

Both versions use the **same model** (`gpt-4o-mini` via OpenRouter) and the **same
search tool**, so the comparison is honest. The only variable is the framework.

## Setup and Run

1. Activate your virtual environment.
2. Install requirements: `pip install -r requirements.txt`
3. Create a `.env` file in the project folder containing:
   ```
   OPENROUTER_API_KEY=sk-or-...
   ```
4. (Optional) Confirm search works from your machine — this rules out the network
   before you blame your agent:
   ```bash
   python search_tools.py --selftest
   ```

Run each version against both test claims:

```bash
# The real claim - should be ANSWERED
python research_langgraph_starter.py "Anthropic was founded by ex-OpenAI staff"
python research_crewai_starter.py    "Anthropic was founded by ex-OpenAI staff"

# The nonsense claim - should be GAP REPORTED
python research_langgraph_starter.py "the flurbotron 9000 was released in 2019"
python research_crewai_starter.py    "the flurbotron 9000 was released in 2019"
```

## How Each One Decides

**LangGraph — the decision is a line of code.**

```
plan_query → run_search → assess → ┬─ ENOUGH ────→ write_answer → END
                                   └─ NOT_ENOUGH → report_gap   → END
```

The fork is `choose_next()`, a plain Python function wired in with
`add_conditional_edges`. The LLM produces a verdict; the *routing on that verdict* is
mine.

**CrewAI — the decision is delegated.**

```
Crew(process=Process.hierarchical, manager_llm=llm)
    ├── Researcher  (owns the Web Search tool)
    └── Writer      (owns the final verdict line)
```

There is no fork to point at. A manager LLM decides who works, in what order, and when
the job is done. The refusal path exists only as text — in a backstory and an
`expected_output`.

## Results

| Claim | Framework | Route taken | Refused correctly? |
| --- | --- | --- | --- |
| Anthropic founded by ex-OpenAI staff | LangGraph | `ANSWERED` | n/a |
| Anthropic founded by ex-OpenAI staff | CrewAI | `ANSWERED` | n/a |
| flurbotron 9000 released in 2019 | LangGraph | `GAP REPORTED` | Yes |
| flurbotron 9000 released in 2019 | CrewAI | `GAP REPORTED` | Yes |

<!-- Paste the real terminal output of the four runs below. -->

<details>
<summary>LangGraph — real claim</summary>

```
(paste output here)
```
</details>

<details>
<summary>LangGraph — nonsense claim</summary>

```
(paste output here)
```
</details>

<details>
<summary>CrewAI — real claim</summary>

```
(paste output here)
```
</details>

<details>
<summary>CrewAI — nonsense claim</summary>

```
(paste output here)
```
</details>

## Observation on Refusal (The Nonsense Claim)

When tested with the nonsense claim (`the flurbotron 9000 was released in 2019`), both
frameworks successfully refused to invent an answer — but they achieved this in
fundamentally different ways.

**LangGraph — a structural decision.** Refusing is a physical code path. The graph
explicitly routes execution to the `report_gap` node when the assessment verdict is
`NOT_ENOUGH`. It is structurally impossible for the agent to bypass this and invent an
answer, because `write_answer` is never reached. I can literally read the decision in
the code.

**CrewAI — a delegated decision.** Refusing is only an instruction inside the agents'
backstories and task descriptions. We *asked the manager nicely* to output
`VERDICT: COULD NOT VERIFY` when there is no evidence. The manager listened to the
prompt and delegated the refusal correctly — but it was an LLM-driven decision, not a
hardcoded route.

**Conclusion:** LangGraph gives absolute structural control over the flow; CrewAI relies
on the model's ability to follow strict prompt instructions.

## Side-by-Side

| | LangGraph | CrewAI (hierarchical) |
| --- | --- | --- |
| Control flow | Explicit graph, hand-drawn edges | Decided at runtime by a manager LLM |
| The refusal | A node the router forces you into | A sentence in a backstory |
| Who calls the search tool | I do, inside `run_search` | The researcher agent, if it decides to |
| State | A `TypedDict` I define and mutate | Passed between agents by the framework |
| Boilerplate | More — nodes, edges, state keys | Less — roles, goals, tasks |
| Debugging a wrong answer | Read the graph, find the node | Read the trace, guess at the manager |
| Reproducibility | High — same route every run | Lower — the manager may reorganise the work |

## Honest Limitations

- **CrewAI's refusal is not guaranteed.** It worked here, but nothing enforces it. That
  is exactly why `main()` has a third branch printing
  `UNKNOWN (The model did not follow the exact formatting instruction)` — the route is
  detected by string-matching the model's own output, which can silently drift.
- **LangGraph's routing is safe; its *verdict* is not.** `read_verdict()` parses the
  last line of free text for `ENOUGH` / `NOT_ENOUGH`. The routing is deterministic, but
  it is only ever as good as that one word the LLM wrote. Structured output would
  remove the last soft spot.
- **The search tool is deliberately weak** — DuckDuckGo Instant Answer plus Wikipedia,
  no paid API. `NO_RESULTS` on an obscure but real claim is possible, so "could not
  verify" here means "could not verify *with these two sources*".
- **`NO_RESULTS` vs `SEARCH_UNAVAILABLE`.** "I found nothing" and "I could not look" are
  different facts. LangGraph handles them before the LLM is ever called (`assess`);
  CrewAI hands both markers to the researcher agent and trusts it to tell them apart.

## Takeaway

Use LangGraph when a wrong branch is expensive — refusals, approvals, anything where
"the agent must not do X" is a requirement rather than a preference. Use CrewAI when the
work is open-ended enough that you would rather not hand-draw every path, and a bad
route costs you a retry instead of a wrong answer shipped as fact.
