# Backtester AI Agent

Three Python learning examples calculate first-to-last closing-price returns using prompt chaining, OpenAI function calling or LangChain. This is not a production strategy backtester.

## Documentation

- [Project analysis](docs/PROJECT_ANALYSIS.md): scope and risks.
- [Detailed analysis](docs/PROJECT_ANALYSIS_DETAILED.md): code, configuration and evidence.
- [Improvement plan](docs/IMPROVEMENT_PLAN.md): priorities.
- [Detailed plan](docs/IMPROVEMENT_PLAN_DETAILED.md): acceptance criteria and branch status.

## Quick Start

Offline checks on 2026-10-30 used Python 3.12.13; baseline declares no runtime or lockfile. Preparation commands are illustrative, not a verified live setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure `LLM_API_KEY`, `YAHOO_FINANCE_SEARCH_API` and `YAHOO_FINANCE_QUERY_API` through your environment. All scripts also load dotenv. Search configuration ends before the query value; history configuration ends before the ticker. See [configuration contracts](docs/PROJECT_ANALYSIS_DETAILED.md#configuration).

The hard-coded model is `gpt-4o-mini`, with `base_url=None`. These entry-point commands require trusted endpoints and deliberate model access. Real runs remain unverified and can incur charges. Script 2 is blocked by baseline LangChain imports with the freshly resolved dependencies:

```bash
python 0_backtester_agent_prompt_chained.py
python 1_backtester_agent.py
python 2_backtester_agent_langchain.py
```

Scripts 0 and 1 construct clients on import; script 1 also executes its conversation loop. Each contains an editable Apple/date example, not an interactive CLI. No original tests exist; prepared fixes and synthetic branch tests are not on main.

## Historical Guides

[Original walkthrough](blog/blog.md), [refined walkthrough](blog/blog_refined.md), [debugging journey](blog/agent_debugging_journey.md) and [demo](https://github.com/user-attachments/assets/441f4f66-432a-4a01-9820-99de4784d325). Their setup and reliability claims are historical.
