# Building an Equity Backtester AI Agent from Scratch with Python Using Open/Free LLM APIs

## Introduction

The rise of AI agents has been fascinating, especially with tools like Cursor's agent that now writes most of my code. This curiosity led me to discover [this Anthropic blog post](https://www.anthropic.com/engineering/building-effective-agents) about building effective agents. Inspired by this, I decided to build my own agent to better understand the underlying mechanics and learn best practices firsthand.

Instead of using established frameworks like LangChain, LangGraph, or CrewAI, I chose to build from scratch using just the LLM APIs. This approach would give me deeper insights into how these systems work.

The first step was to choose a practical use case. I wanted something simple yet useful.

<img src="../media/returns_meme.png" alt="meme" style="width:200px;height:200px;">

You've probably seen those viral posts that say something like "If you had invested $10K in [Company X] in [Year Y], you would have [Z] today!" While these posts are common, there aren't many tools to verify their accuracy. For financial market professionals, this is child's play, but for the average person, it's not so straightforward. While Google can provide answers for popular stocks, I wanted to build an AI agent that could handle this calculation for any stock across financial markets.


### Our Goal
When someone asks: "What would have been my return if I had invested in Amazon from 2023-01-01 to 2023-12-31?"
The agent should respond with something like: "You would have made 77% returns on your investment during this period." (Yes, that's the approximate return I calculated using TradingView!)

As an AI/ML enthusiast, I naturally chose Python for this project.

## High-Level Solution

The agent works by:
1. Taking a user's query as input
2. Extracting relevant information (company name, date range)
3. Performing the backtesting calculation
4. Returning a human-readable summary of the returns

Let's break down the solution into manageable tasks:
- Extract company name, start date, and end date from the query
- Format the extracted data for API consumption
- Fetch the stock symbol for the given company
- Retrieve historical price data for the specified period
- Calculate the returns based on historical prices
- Generate a user-friendly summary

Now, let's dive into the implementation! 🥤

> **Note**: If you prefer to skip the setup details, feel free to jump directly to the codebase.

> This blog post covers the `function calling` based implementation. I've also created a separate implementation using prompt chaining, structured/model-led data handling, and gate keeping in `backtester_agent_prompt_chained.py` for those interested in building AI agents with more fine-grained control.

## Setup

1. Install [LMStudio app](https://lmstudio.ai/) (I prefer it because it allows installing open models from Hugging Face)

2. Download your preferred models. For this demonstration, I used these DeepSeek models:
   - deepseek-r1-distill-llama-8b
   - deepseek-r1-distill-qwen-7b

3. Ensure LMStudio's localhost API is running from the `Developer` tab and accessible at `http://127.0.0.1:1234`

4. Install Python extensions for:
   - IntelliSense
   - Virtual environment (.venv)
   - Jupyter notebooks
   - Other preferred extensions

5. Create a `.env` file with these values:
   ```
   YAHOO_FINANCE_QUERY_API=
   YAHOO_FINANCE_SEARCH_API=
   ```
   > **Tip**: I'm not revealing these APIs as they're not public, but you can find them on Yahoo's site by searching for a symbol.

6. Create a `requirements.txt` file with these dependencies:
   - requests (for HTTP requests/API calls)
   - ipykernel (for running the agent interactively with Jupyter notebook)
   - python-dotenv (for reading environment variables)
   - [openai](https://github.com/openai/openai-python?tab=readme-ov-file) (for simplified LLM API calls)

7. Configure your virtual environment (easily done with extensions) and add it to .gitignore

8. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Implementation

Let's start with the basic setup in `backtester_agent.py`:

```python
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()
```

Create an OpenAI client pointing to our localhost API:
```python
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="something",
)
```

The core API call that makes everything work:
```python
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)
```

Key components:
- `model`: The name of the model we're using
- `messages`: An array of messages providing context to the model
- `tools`: A collection of functions that the agent can call (using function calling)

Now, let's implement the core functions for our tasks:

```python
def get_symbol(search_query: str):
    """Gets user query as input and returns a valid symbol"""

def get_ticker_historical_prices(symbol: str, start_date_timestamp: int, end_date_timestamp: int):
    """Gets a valid symbol and start/end timestamp values and returns start/end price"""

def convert_date_to_timestamp(start_date: str, end_date: str):
    """Converts the given date strings into unix timestamp values"""

def get_return_percentage(start_date_timestamp_price: float, end_date_timestamp_price: float):
    """Calculates the return percentage based on the given start/end prices"""
```

> **Note**: You can replace the Yahoo Finance API with any other API by modifying the `get_symbol` and `get_ticker_historical_prices` functions.

Define the tools array for the LLM API:
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": "Get symbol for a given search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string"},
                },
                "required": ["search_query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    # ... other functions
]
```

Create a generic function to handle tool calls:
```python
def call_function(name, args):
    try:
        if name == "get_symbol":
            return get_symbol(**args)
        elif name == "convert_date_to_timestamp":
            return convert_date_to_timestamp(**args)
        elif name == "get_ticker_historical_prices":
            return get_ticker_historical_prices(**args)
        elif name == "get_return_percentage":
            return get_return_percentage(**args)
        else:
            print(f"Unknown function: {name}")
            return None
    except Exception as e:
        print(f"Error calling function {name}: {e}")
        return None
```

Define the system prompt:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What would have been the return of Apple from 2020-01-01 to 2020-12-31?"}
]
```

The system prompt is crucial - I spent considerable time refining it to ensure the model follows the exact steps we want. (Check the codebase for the detailed prompt that worked.)

Go through, the [`backtester_agent.py`](../backtester_agent.py) and run it in the interactive window to see its think through response & the final result


## Overcoming Challenges

Building an AI agent isn't as simple as just defining tools and prompts. Here are the key challenges I faced and how I solved them:

### 1. Infinite Loops and Inconsistent Execution

**Problem**: The agent would get stuck in loops or skip necessary tool calls.

**Solution**: Implemented safeguards:
```python
MAX_ITERATIONS = 10
iteration_count = 0
previous_tool_calls = set()

tool_call_id = f"{name}:{json.dumps(args, sort_keys=True)}"
if tool_call_id in previous_tool_calls:
    print(f"Detected repeated tool call: {tool_call_id}")
    messages.append({
        "role": "system", 
        "content": f"You've already called {name} with these arguments..."
    })
    continue
```

### 2. Hallucinated Responses

**Problem**: The model would skip tools or make up data.

**Solution**: Enhanced system prompt and state tracking:
```python
system_prompt = """
CRITICAL RULES:
- NEVER skip any required tool calls
- NEVER provide an answer without completing ALL tool calls
- NEVER make up or hallucinate data
- NEVER calculate returns manually
"""

calculation_state = {
    "symbol": None,
    "start_date_timestamp": None,
    "end_date_timestamp": None,
    "start_date_timestamp_price": None,
    "end_date_timestamp_price": None,
    "return_percentage": None
}
```

### 3. Parameter Confusion

**Problem**: The model would mix up timestamps and price values.

**Solution**: Implemented type validation and parameter overriding:
```python
if not isinstance(start_date_timestamp, int):
    print(f"Error: start_date_timestamp must be an integer, got {type(start_date_timestamp)}")
    return None

args = {
    "symbol": calculation_state["symbol"],
    "start_date_timestamp": calculation_state["start_date_timestamp"],
    "end_date_timestamp": calculation_state["end_date_timestamp"]
}
```

### 4. Failure to Output Results

**Problem**: The agent would continue iterating even after successful calculation.

**Solution**: Implemented immediate termination:
```python
if name == "get_return_percentage" and calculation_state["return_percentage"] is not None:
    print(f"Calculation complete! Return percentage: {calculation_state['return_percentage']}%")
    final_message = f"The return for {calculation_state['symbol']} from {messages[1]['content']} was {calculation_state['return_percentage']}%."
    final_completion = client.chat.completions.create(
        model=model,
        messages=messages + [{"role": "system", "content": final_message}],
        tools=[]
    )
    completion = final_completion
    break
```

## Key Takeaways

Building reliable AI agents requires more than just defining tools and prompts. Here are the essential lessons:

1. **Implement safeguards against loops**: Track previous calls and limit iterations
2. **Maintain explicit state**: Keep track of all values in a central state object
3. **Validate inputs rigorously**: Check types and values before executing functions
4. **Override parameters when needed**: Don't trust the model to pass the correct values
5. **Force termination when complete**: Break out of the loop as soon as you have the result
6. **Provide clear guidance**: Use explicit system messages to guide the model
7. **Add detailed logging**: Log all values and states to help with debugging



## Demo
here's the working demo of this agent with this user input:
> What would have been the return of Apple from 2020-01-01 to 2020-12-31?

https://github.com/user-attachments/assets/441f4f66-432a-4a01-9820-99de4784d325

#### Output:

<img src="../media/output.png" alt="output" style="width:400px;height:400px;">


## Conclusion

The Backtester agent now reliably calculates historical returns for any stock, following the complete sequence of steps without hallucination or loops. Building agentic systems with LLMs requires careful handling of their quirks and limitations. By implementing proper safeguards, validation, and state management, you can create reliable agents that consistently perform complex tasks.

The techniques described here can be applied to any AI agent to improve its reliability and prevent common failure modes.

Happy agent building! 🚀 