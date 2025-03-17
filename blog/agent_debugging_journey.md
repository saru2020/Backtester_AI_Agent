# Debugging an Equity Backtester AI Agent: Common Pitfalls and Solutions

## Introduction

Building an AI agent that can accurately backtest equity returns seems straightforward at first glance. However, as I discovered during development, LLMs can exhibit several quirky behaviors that need to be addressed for reliable performance. In this post, I'll share the key issues encountered while building my equity backtester agent and the solutions that finally made it work consistently.

## The Problem: Inconsistent Agent Behavior

My backtester agent was designed to perform a simple sequence of operations:
1. Extract company name and date range from user queries
2. Get the stock symbol
3. Convert dates to timestamps
4. Fetch historical prices
5. Calculate the return percentage

Despite this clear workflow, the agent would frequently:
- Get stuck in infinite loops
- Skip necessary tool calls
- Hallucinate data instead of using API results
- Confuse timestamps with price values
- Fail to output the final result

Let's dive into each issue and how I solved them.

## Issue 1: Infinite Loops and Inconsistent Execution

### Problem
The agent would sometimes get stuck in a loop, repeatedly calling the same function with the same arguments, or fail to complete the entire sequence of tool calls.

### Solution
I implemented several safeguards:
```python
# Set a maximum number of iterations
MAX_ITERATIONS = 10
iteration_count = 0

# Track previous tool calls to detect loops
previous_tool_calls = set()

# Create a unique identifier for each tool call
tool_call_id = f"{name}:{json.dumps(args, sort_keys=True)}"

# Check if we've seen this exact tool call before
if tool_call_id in previous_tool_calls:
    print(f"Detected repeated tool call: {tool_call_id}")
    messages.append({
        "role": "system", 
        "content": f"You've already called {name} with these arguments..."
    })
    continue
```

This approach effectively prevented the agent from getting stuck in loops by tracking and rejecting repeated tool calls.

## Issue 2: Hallucinated Responses

### Problem
The model would sometimes skip calling tools entirely and just make up an answer, or hallucinate values instead of using the actual data returned by the API calls.

### Solution
I enhanced the system prompt with explicit instructions and implemented state tracking:

```python
system_prompt = """
CRITICAL RULES:
- NEVER skip any of the required tool calls
- NEVER provide an answer without completing ALL tool calls in the correct sequence
- NEVER make up or hallucinate any data - use ONLY the data returned by the tools
- NEVER attempt to calculate returns yourself - ALWAYS use the get_return_percentage tool
"""

# Track the state of the calculation
calculation_state = {
    "symbol": None,
    "start_date_timestamp": None,
    "end_date_timestamp": None,
    "start_date_timestamp_price": None,
    "end_date_timestamp_price": None,
    "return_percentage": None
}
```

I also added validation to ensure the model couldn't finish without completing all required steps:

```python
if calculation_state["return_percentage"] is None:
    # Force it to continue
    messages.append({
        "role": "system",
        "content": "You must complete the full sequence of tool calls..."
    })
    continue
```

## Issue 3: Parameter Confusion

### Problem
The model would confuse timestamps with price values, trying to calculate returns using timestamps or passing price values to functions expecting timestamps.

### Solution
I implemented extensive type validation and parameter overriding:

```python
# Validate input types
if not isinstance(start_date_timestamp, int):
    print(f"Error: start_date_timestamp must be an integer, got {type(start_date_timestamp)}")
    return None

# Always override the arguments with stored values to prevent hallucination
args = {
    "symbol": calculation_state["symbol"],
    "start_date_timestamp": calculation_state["start_date_timestamp"],
    "end_date_timestamp": calculation_state["end_date_timestamp"]
}

# Add explicit reminders to the model
messages.append({
    "role": "system",
    "content": f"Using the following values for historical prices: symbol={calculation_state['symbol']}, start_timestamp={calculation_state['start_date_timestamp']}, end_timestamp={calculation_state['end_date_timestamp']}"
})
```

This ensured that the correct values were always used, regardless of what the model tried to pass.

## Issue 4: Failure to Output Results

### Problem
Even when the calculation was successful, the agent would sometimes continue iterating and fail to output the final result.

### Solution
I implemented an immediate termination and forced output once the calculation was complete:

```python
# If we've completed the calculation, break out of the loop
if name == "get_return_percentage" and calculation_state["return_percentage"] is not None:
    print(f"Calculation complete! Return percentage: {calculation_state['return_percentage']}%")
    # Force a final response with the result
    try:
        final_message = f"The return for {calculation_state['symbol']} from {messages[1]['content']} was {calculation_state['return_percentage']}%."
        final_completion = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "system", "content": final_message}],
            tools=[]
        )
        print("Final calculated response:", final_completion.choices[0].message.content)
        # Set completion to this final response so it's used at the end
        completion = final_completion
        break
    except Exception as e:
        print(f"Error getting final calculated response: {e}")
```

This ensured that as soon as the return percentage was calculated, the agent would immediately provide the final answer.

## Key Takeaways

Building reliable AI agents requires more than just defining tools and prompts. Here are the key lessons learned:

1. **Implement safeguards against loops**: Track previous calls and limit iterations.
2. **Maintain explicit state**: Keep track of all values in a central state object.
3. **Validate inputs rigorously**: Check types and values before executing functions.
4. **Override parameters when needed**: Don't trust the model to pass the correct values.
5. **Force termination when complete**: Break out of the loop as soon as you have the result.
6. **Provide clear guidance**: Use explicit system messages to guide the model.
7. **Add detailed logging**: Log all values and states to help with debugging.

With these improvements, the equity backtester agent now reliably calculates historical returns for any stock, following the complete sequence of steps without hallucination or loops.

## Conclusion

Building agentic systems with LLMs requires careful handling of their quirks and limitations. By implementing proper safeguards, validation, and state management, you can create reliable agents that consistently perform complex tasks. The techniques described here can be applied to any AI agent to improve its reliability and prevent common failure modes.

Happy agent building! 