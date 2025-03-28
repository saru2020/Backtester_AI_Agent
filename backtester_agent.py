from openai import OpenAI
import requests
import json
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

base_url="http://127.0.0.1:1234/v1" #None
LLM_API_KEY=os.getenv("LLM_API_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=LLM_API_KEY,
)
model = "deepseek-r1-distill-llama-8b" #"gpt-4o-mini"

# --------------------------------------------------------------
# Define the tool (function) that we want to call
# --------------------------------------------------------------


def get_symbol(search_query: str):
    print('inside get_symbol - search_query: ', search_query)
    """This is NOT a publically available API that returns the symbols for a given company name."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(
            f"{os.getenv('YAHOO_FINANCE_SEARCH_API')}{search_query}&lang=en-US&region=US&quotesCount=6",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        print('inside get_symbol - data: ', data)
        
        if not data.get("quotes") or len(data["quotes"]) == 0:
            raise ValueError(f"No symbol found for search query: {search_query}")
            
        return data["quotes"][0]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error in get_symbol: {e}")
        return None

def convert_date_to_timestamp(start_date: str, end_date: str):
    try:
        dt1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')  
        dt1 = dt1.replace(tzinfo=datetime.timezone.utc)  # Set UTC timezone
        dt2 = datetime.datetime.strptime(end_date, '%Y-%m-%d')  
        dt2 = dt2.replace(tzinfo=datetime.timezone.utc)  # Set UTC timezone
        dt2 += datetime.timedelta(days=1)  # Move to the next day to get the EOD timestamp
        timestamps = (int(dt1.timestamp()), int(dt2.timestamp()))
        print(f"Converted timestamps: {timestamps}")
        return {
            "start_date_timestamp": timestamps[0],
            "end_date_timestamp": timestamps[1]
        }
    except ValueError as e:
        print(f"Error converting dates to timestamps: {e}")
        return None


def get_ticker_historical_prices(symbol: str, start_date_timestamp: int, end_date_timestamp: int):
    print('inside get_ticker_historical_prices - symbol: ', symbol)
    print(f'Using timestamps: start={start_date_timestamp}, end={end_date_timestamp}')
    
    # Validate input types
    if not isinstance(symbol, str):
        print(f"Error: symbol must be a string, got {type(symbol)}")
        return None
    if not isinstance(start_date_timestamp, int):
        print(f"Error: start_date_timestamp must be an integer, got {type(start_date_timestamp)}")
        return None
    if not isinstance(end_date_timestamp, int):
        print(f"Error: end_date_timestamp must be an integer, got {type(end_date_timestamp)}")
        return None
        
    """This is NOT a publically available API that returns the historical prices for a given ticker."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(
            f"{os.getenv('YAHOO_FINANCE_QUERY_API')}{symbol}?period1={start_date_timestamp}&period2={end_date_timestamp}&interval=1d",
            headers=headers
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if not data.get("chart") or not data["chart"].get("result"):
            raise ValueError("No data available for the specified ticker and date range")
            
        quotes = data["chart"]["result"][0]["indicators"]["quote"][0]
        if not quotes.get("close"):
            raise ValueError("No closing price data available")
            
        start_date_timestamp_price = quotes["close"][0]
        end_date_timestamp_price = quotes["close"][-1]
        
        if start_date_timestamp_price is None or end_date_timestamp_price is None:
            raise ValueError("Missing price data for start or end date")
            
        # Return a dictionary with clear labels for the prices
        result = {
            "start_date_timestamp_price": start_date_timestamp_price,
            "end_date_timestamp_price": end_date_timestamp_price
        }
        print(f"Historical prices retrieved: {result}")
        return result
        
    except requests.RequestException as e:
        print(f"Error making request to Yahoo Finance API: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing data: {e}")
        return None


def get_return_percentage(start_date_timestamp_price: float, end_date_timestamp_price: float):
    print(f"Calculating return percentage with prices: start={start_date_timestamp_price}, end={end_date_timestamp_price}")
    
    # Validate input types
    if not isinstance(start_date_timestamp_price, (int, float)):
        print(f"Error: start_date_timestamp_price must be a number, got {type(start_date_timestamp_price)}")
        return None
    if not isinstance(end_date_timestamp_price, (int, float)):
        print(f"Error: end_date_timestamp_price must be a number, got {type(end_date_timestamp_price)}")
        return None
        
    try:
        result = {
            "return_percentage": ((end_date_timestamp_price - start_date_timestamp_price) / start_date_timestamp_price) * 100
        }
        print(f"Calculated return percentage: {result['return_percentage']}%")
        return result
    except (TypeError, ZeroDivisionError) as e:
        print(f"Error calculating return percentage: {e}")
        return None


def call_function(name, args):
    try:
        if name == "get_symbol":
            print('args: ', args)
            return get_symbol(**args)
        elif name == "convert_date_to_timestamp":
            print('args: ', args)
            return convert_date_to_timestamp(**args)
        elif name == "get_ticker_historical_prices":
            print('args: ', args)
            return get_ticker_historical_prices(**args)
        elif name == "get_return_percentage":
            print('args: ', args)
            return get_return_percentage(**args)
        else:
            print(f"Unknown function: {name}")
            return None
    except Exception as e:
        print(f"Error calling function {name}: {e}")
        return None

# --------------------------------------------------------------
# Step 1: Call model with get_weather tool defined
# --------------------------------------------------------------

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
    {
        "type": "function",
        "function": {
            "name": "get_ticker_historical_prices",
            "description": "Get historical prices for a given ticker.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}, "start_date_timestamp": {"type": "integer"}, "end_date_timestamp": {"type": "integer"}},
                "required": ["symbol", "start_date_timestamp", "end_date_timestamp"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_date_to_timestamp",
            "description": "Convert a date to a timestamp.",
            "parameters": {
                "type": "object",
                "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}},
                "required": ["start_date", "end_date"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_return_percentage",
            "description": "Get the return percentage for a given start date and end date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date_timestamp_price": {"type": "number"},
                    "end_date_timestamp_price": {"type": "number"}
                },
                "required": ["start_date_timestamp_price", "end_date_timestamp_price"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

system_prompt = """You are a helpful equity backtester assistant that calculates historical returns for stocks.

AVAILABLE TOOLS:
1. get_symbol: Retrieves the stock symbol for a company name
   - Input: Company name as search query
   - Output: Stock symbol and company details

2. convert_date_to_timestamp: Converts dates to Unix timestamps
   - Input: start and end dates in YYYY-MM-DD format
   - Output: Unix timestamp for start and end dates

3. get_ticker_historical_prices: Fetches historical stock prices
   - Input: Stock symbol and start/end date Unix timestamps
   - Output: Opening and closing prices for the period

4. get_return_percentage: Calculates percentage return
   - Input: Start and end prices
   - Output: Percentage return

INSTRUCTIONS:
1. You MUST extract just the company name, start date and end date from user queries
2. You MUST use tools in EXACTLY this sequence with no exceptions:
   a. First call get_symbol with the company name
   b. Then call convert_date_to_timestamp with the start and end dates
   c. Then call get_ticker_historical_prices with the symbol and timestamps
   d. Finally call get_return_percentage with the prices returned by get_ticker_historical_prices
3. You MUST validate all data before proceeding to next step
4. You MUST return only the final percentage as output
5. You MUST handle errors gracefully and inform user of any issues

CRITICAL RULES:
- NEVER skip any of the required tool calls
- NEVER provide an answer without completing ALL tool calls in the correct sequence
- NEVER make up or hallucinate any data - use ONLY the data returned by the tools
- NEVER attempt to calculate returns yourself - ALWAYS use the get_return_percentage tool
- DO NOT give instructions or steps to the user, just return the final output
- When calling get_ticker_historical_prices, use EXACTLY the symbol and timestamps returned by previous tools
- When calling get_return_percentage, use EXACTLY the prices returned by get_ticker_historical_prices
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What would have been the return of Apple from 2020-01-01 to 2020-12-31?"}, # "whats the symbol for apple?"}, 
]

# Track previous tool calls to detect loops
previous_tool_calls = set()
# Set a maximum number of iterations to prevent infinite loops
MAX_ITERATIONS = 10
iteration_count = 0
# Track the state of the calculation
calculation_state = {
    "symbol": None,
    "start_date_timestamp": None,
    "end_date_timestamp": None,
    "start_date_timestamp_price": None,
    "end_date_timestamp_price": None,
    "return_percentage": None
}

# Initialize completion variable
completion = None

# Run the conversation loop until the model indicates it's done or max iterations reached
while iteration_count < MAX_ITERATIONS:
    try:
        iteration_count += 1
        print(f"Iteration {iteration_count}/{MAX_ITERATIONS}")
        
        # Get the completion from the model
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )

        # Check if the model is done
        if completion.choices[0].finish_reason == "stop":
            # Check if we have a complete calculation
            if calculation_state["return_percentage"] is None:
                # If the model tries to finish without calculating a return percentage, force it to continue
                print("Model attempted to finish without completing the calculation. Forcing it to continue.")
                messages.append({
                    "role": "system",
                    "content": "You must complete the full sequence of tool calls before providing a final answer. Please continue with the next required tool call."
                })
                # Add the model's message to the conversation
                messages.append(completion.choices[0].message)
                continue
            else:
                print("Model has completed its response with a valid calculation.")
                break
            
        # If no tool calls in the response, but not finished, add the message and continue
        if not hasattr(completion.choices[0].message, 'tool_calls') or not completion.choices[0].message.tool_calls:
            print("No tool calls in response. Instructing model to use tools.")
            messages.append(completion.choices[0].message)
            
            # Determine which tool should be called next based on the current state
            next_tool = None
            if calculation_state["symbol"] is None:
                next_tool = "get_symbol"
            elif calculation_state["start_date_timestamp"] is None:
                next_tool = "convert_date_to_timestamp"
            elif calculation_state["start_date_timestamp_price"] is None:
                next_tool = "get_ticker_historical_prices"
            elif calculation_state["return_percentage"] is None:
                next_tool = "get_return_percentage"
                
            if next_tool:
                messages.append({
                    "role": "system",
                    "content": f"You must use the tools to calculate the return. Please call the {next_tool} tool next."
                })
            continue

        # Process any tool calls
        tool_call_count = 0
        for tool_call in completion.choices[0].message.tool_calls:
            tool_call_count += 1
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Create a unique identifier for this tool call to detect loops
            tool_call_id = f"{name}:{json.dumps(args, sort_keys=True)}"
            
            # Check if we've seen this exact tool call before
            if tool_call_id in previous_tool_calls:
                print(f"Detected repeated tool call: {tool_call_id}")
                # Add a message to help the model break out of the loop
                messages.append({
                    "role": "system", 
                    "content": f"You've already called {name} with these arguments. If you didn't get the expected response make sure to validate the input parameters and try again or please proceed to the next step or provide a final answer."
                })
                continue
                
            # Add this tool call to our set of previous calls
            previous_tool_calls.add(tool_call_id)
            
            # Add the model's message to the conversation
            messages.append(completion.choices[0].message)

            print(f'Calling function: {name}')
            print(f'With args: {args}')
            
            # Validate tool call sequence
            if name == "get_ticker_historical_prices":
                # Check if we have the symbol and timestamps
                if calculation_state["symbol"] is None:
                    error_message = {"error": "Cannot fetch historical prices without a valid symbol. Please call get_symbol first."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                if calculation_state["start_date_timestamp"] is None or calculation_state["end_date_timestamp"] is None:
                    error_message = {"error": "Cannot fetch historical prices without valid timestamps. Please call convert_date_to_timestamp first."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                
                # Validate timestamp types
                if not isinstance(calculation_state["start_date_timestamp"], int):
                    error_message = {"error": f"Invalid start_date_timestamp type: {type(calculation_state['start_date_timestamp'])}. Must be an integer."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                    
                if not isinstance(calculation_state["end_date_timestamp"], int):
                    error_message = {"error": f"Invalid end_date_timestamp type: {type(calculation_state['end_date_timestamp'])}. Must be an integer."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                
                # Always override the arguments with stored values to prevent hallucination
                args = {
                    "symbol": calculation_state["symbol"],
                    "start_date_timestamp": calculation_state["start_date_timestamp"],
                    "end_date_timestamp": calculation_state["end_date_timestamp"]
                }
                print(f"Using stored values for get_ticker_historical_prices: {args}")
            
            # Special handling for get_return_percentage to ensure it uses the correct values
            elif name == "get_return_percentage":
                # Check if we have the prices from get_ticker_historical_prices
                if calculation_state["start_date_timestamp_price"] is None or calculation_state["end_date_timestamp_price"] is None:
                    error_message = {"error": "Cannot calculate return percentage without valid price data. Please call get_ticker_historical_prices first."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                
                # Validate price types
                if not isinstance(calculation_state["start_date_timestamp_price"], (int, float)):
                    error_message = {"error": f"Invalid start_date_timestamp_price type: {type(calculation_state['start_date_timestamp_price'])}. Must be a number."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                    
                if not isinstance(calculation_state["end_date_timestamp_price"], (int, float)):
                    error_message = {"error": f"Invalid end_date_timestamp_price type: {type(calculation_state['end_date_timestamp_price'])}. Must be a number."}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_message)
                    })
                    continue
                
                # Override the args with the stored values to prevent hallucination
                args = {
                    "start_date_timestamp_price": calculation_state["start_date_timestamp_price"],
                    "end_date_timestamp_price": calculation_state["end_date_timestamp_price"]
                }
                print(f"Using stored price values: {args}")
            
            result = call_function(name, args)
            if result is None:
                print(f"Error: {name} returned None")
                error_message = {"error": f"{name} failed to return valid data. Please try a different approach."}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(error_message)
                })
                
                # If a critical tool in the sequence fails, reset the state for that step and subsequent steps
                if name == "get_symbol":
                    calculation_state["symbol"] = None
                    calculation_state["start_date_timestamp_price"] = None
                    calculation_state["end_date_timestamp_price"] = None
                    calculation_state["return_percentage"] = None
                elif name == "convert_date_to_timestamp":
                    calculation_state["start_date_timestamp"] = None
                    calculation_state["end_date_timestamp"] = None
                    calculation_state["start_date_timestamp_price"] = None
                    calculation_state["end_date_timestamp_price"] = None
                    calculation_state["return_percentage"] = None
                elif name == "get_ticker_historical_prices":
                    calculation_state["start_date_timestamp_price"] = None
                    calculation_state["end_date_timestamp_price"] = None
                    calculation_state["return_percentage"] = None
                
                # Add a message to guide the model on what to do next
                next_tool = None
                if calculation_state["symbol"] is None:
                    next_tool = "get_symbol"
                elif calculation_state["start_date_timestamp"] is None:
                    next_tool = "convert_date_to_timestamp"
                elif calculation_state["start_date_timestamp_price"] is None:
                    next_tool = "get_ticker_historical_prices"
                
                if next_tool:
                    messages.append({
                        "role": "system",
                        "content": f"Please try again with the {next_tool} tool."
                    })
                
                continue
            
            # Store results in our calculation state
            if name == "get_symbol":
                calculation_state["symbol"] = result.get("symbol")
            elif name == "convert_date_to_timestamp":
                if isinstance(result, tuple) and len(result) == 2:
                    calculation_state["start_date_timestamp"] = result[0]
                    calculation_state["end_date_timestamp"] = result[1]
                elif isinstance(result, dict):
                    calculation_state["start_date_timestamp"] = result.get("start_date_timestamp")
                    calculation_state["end_date_timestamp"] = result.get("end_date_timestamp")
                print(f"Stored timestamps: start={calculation_state['start_date_timestamp']}, end={calculation_state['end_date_timestamp']}")
            elif name == "get_ticker_historical_prices":
                if isinstance(result, dict):
                    calculation_state["start_date_timestamp_price"] = result.get("start_date_timestamp_price")
                    calculation_state["end_date_timestamp_price"] = result.get("end_date_timestamp_price")
            elif name == "get_return_percentage":
                if isinstance(result, dict):
                    calculation_state["return_percentage"] = result.get("return_percentage")
                    # Once we have the return percentage, add a message to help the model finish
                    # messages.append({
                    #     "role": "system",
                    #     "content": f"You have successfully calculated the return percentage: {calculation_state['return_percentage']}%. Please provide this as your final answer without making any more tool calls."
                    # })
                
            print(f'Result: {result}')
            print(f'Current calculation state: {calculation_state}')
            
            # Add the tool response to messages with the correct tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
            
            print(f"Messages: {messages}")
            # If we've completed the calculation, break out of the loop
            if name == "get_return_percentage" and calculation_state["return_percentage"] is not None:
                print(f"Calculation complete! Return percentage: {calculation_state['return_percentage']}%")
                # Force a final response with the result
                try:
                    # Add the assistant's message with the final tool call first
                    # messages.append(completion.choices[0].message)
                    
                    #these msgs fails with OpenAI apis
                    # Add the tool response
                    # messages.append({
                    #     "role": "tool",
                    #     "tool_call_id": tool_call.id,
                    #     "content": json.dumps(calculation_state)
                    # })
                    
                    # Add the final system message
                    # final_message = f"The return for {calculation_state['symbol']} from {messages[1]['content']} was {calculation_state['return_percentage']}%."
                    # messages.append({
                    #     "role": "system",
                    #     "content": final_message
                    # })
                    
                    # Get the final response without any tools
                    final_completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=[]
                    )
                    print("Final calculated response:", final_completion.choices[0].message.content)
                    # Set completion to this final response so it's used at the end
                    completion = final_completion
                    break
                except Exception as e:
                    print(f"Error getting final calculated response: {e}")
        
        # If we processed all tool calls but there were none, break to avoid infinite loop
        if tool_call_count == 0:
            print("No valid tool calls to process. Breaking loop.")
            break
            
    except Exception as e:
        print(f"Error in conversation loop: {e}")
        # Add error message to conversation to help model recover
        messages.append({
            "role": "system", 
            "content": f"An error occurred: {str(e)}. Please try a different approach or provide a final answer."
        })
        # If we've hit too many errors, break out
        if iteration_count >= MAX_ITERATIONS / 2:
            print("Too many errors, breaking loop")
            break

# If we've reached max iterations without completion, add a final message
if iteration_count >= MAX_ITERATIONS:
    print(f"Reached maximum iterations ({MAX_ITERATIONS}). Forcing completion.")
    try:
        # Force a final response without tool calls
        final_message = "Please provide a final answer based on the information gathered so far. "
        if calculation_state["return_percentage"] is not None:
            final_message += f"The calculated return percentage is {calculation_state['return_percentage']}%."
            # Add the company name and date range if available
            if calculation_state["symbol"] is not None:
                final_message += f" This is for {calculation_state['symbol']}."
        else:
            final_message += "Unable to complete the calculation. Please try again with a different query."
        
        final_completion = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "system", "content": final_message}],
            tools=[]  # No tools available for this call
        )
        print("Final forced response:", final_completion.choices[0].message.content)
    except Exception as e:
        print(f"Error getting final response: {e}")
elif completion is not None:  # Make sure completion exists before trying to access it
    # Check if we have a valid calculation before accepting the model's response
    if calculation_state["return_percentage"] is None:
        print("Warning: Model finished without completing the calculation.")
        try:
            # Force a final response that indicates the calculation was incomplete
            final_message = "The calculation was not completed properly. Please try again."
            final_completion = client.chat.completions.create(
                model=model,
                messages=messages + [{"role": "system", "content": final_message}],
                tools=[]
            )
            print("Final corrected response:", final_completion.choices[0].message.content)
        except Exception as e:
            print(f"Error getting corrected response: {e}")
    else:
        # Print the final response from the model
        print("Final response:", completion.choices[0].message.content)
else:
    # If we somehow got here without a completion but have a return percentage, show it
    if calculation_state["return_percentage"] is not None:
        print(f"Final calculated return percentage: {calculation_state['return_percentage']}% for {calculation_state['symbol']}")
    else:
        print("No completion was generated and no return percentage was calculated. The process may have failed early.")
