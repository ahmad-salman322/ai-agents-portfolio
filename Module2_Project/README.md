# Module 2 - Phase 1 Mini-Project: Study Guide Agent

## Setup and Run Commands
1. Ensure your virtual environment is active.
2. Install requirements: `pip install -r requirements.txt`
3. Add your API key to the `.env` file: `OPENROUTER_API_KEY=sk-or-...`
4. Run LangGraph implementation: `python study_guide_langgraph.py "Model Context Protocol"`
5. Run CrewAI implementation: `python study_guide_crewai.py "Model Context Protocol"`

## Tested Topic
"Model Context Protocol"

## Observations
- **What did LangGraph make explicit?** 
  LangGraph required me to explicitly define the state structure (using TypedDict) and manually map the data flow and control flow. I had to explicitly wire the path from one node to the next using `add_edge`.
  
- **What did CrewAI automate or hide?** 
  CrewAI automated the state passing between tasks. By using a sequential process and passing the previous task in the `context` parameter, the framework automatically handled handing off the data from the explanation task to the example task without me writing any routing logic.
  
- **What would you choose for this three-task pipeline, and why?** 
  For this linear, three-task pipeline, I would choose CrewAI. It requires significantly less boilerplate code and feels more natural when defining roles and tasks sequentially without complex conditional routing.
