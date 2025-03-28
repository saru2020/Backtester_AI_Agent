from typing import Dict, List, Optional, Tuple, Any
import datetime
import os
import json
import requests
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GetSymbolInput(BaseModel):
    company_name: str = Field(description="Company name (e.g., Apple)")

class ConvertDateInput(BaseModel):
    date_str: str = Field(description="Comma-separated dates in YYYY-MM-DD format (e.g., '2020-01-01,2020-12-31')")

class GetTickerHistoricalPricesInput(BaseModel):
    input_str: str = Field(description="Comma-separated string of symbol and timestamps (e.g., 'AAPL,1577836800,1609459200')")

class GetReturnInput(BaseModel):
    prices_str: str = Field(description="Comma-separated string of start and end prices (e.g., '75.0875,132.69')")

def get_symbol(company_name: str) -> str:
    """Get symbol for a given search query."""
    logger.info(f"get_symbol called with company_name: {company_name}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(
            f"{os.getenv('YAHOO_FINANCE_SEARCH_API')}{company_name}&lang=en-US&region=US&quotesCount=6",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("quotes") or len(data["quotes"]) == 0:
            error_msg = f"No symbol found for search query: {company_name}"
            logger.warning(error_msg)
            return error_msg
            
        result = data["quotes"][0]
        logger.info(f"get_symbol returning: {result}")
        return result
    except (requests.RequestException, ValueError, KeyError) as e:
        error_msg = f"Error in get_symbol: {str(e)}"
        logger.error(error_msg)
        return error_msg

def convert_date_to_timestamp(date_str: str) -> str:
    """Convert dates to Unix timestamps. Input format: '2020-01-01,2020-12-31'"""
    logger.info(f"convert_date_to_timestamp called with date_str: {date_str}")
    try:
        start_date, end_date = date_str.split(",")
        start = int(datetime.datetime.strptime(start_date.strip(), "%Y-%m-%d").timestamp())
        end = int(datetime.datetime.strptime(end_date.strip(), "%Y-%m-%d").timestamp())
        result = f"{start},{end}"
        logger.info(f"convert_date_to_timestamp returning: {result}")
        return result
    except Exception as e:
        error_msg = f"Error converting dates: {str(e)}"
        logger.error(error_msg)
        return error_msg

def get_ticker_historical_prices(input_str: str) -> str:
    """Get historical prices for a stock. Input format: 'AAPL,1577836800,1609459200'"""
    logger.info(f"get_ticker_historical_prices called with input_str: {input_str}")
    try:
        symbol, start_timestamp, end_timestamp = input_str.split(",")
        start_timestamp = int(start_timestamp)
        end_timestamp = int(end_timestamp)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        response = requests.get(
            f"{os.getenv('YAHOO_FINANCE_QUERY_API')}{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("chart") or not data["chart"].get("result"):
            error_msg = "No data available for the specified ticker and date range"
            logger.warning(error_msg)
            return error_msg
            
        quotes = data["chart"]["result"][0]["indicators"]["quote"][0]
        if not quotes.get("close"):
            error_msg = "No closing price data available"
            logger.warning(error_msg)
            return error_msg
            
        start_price = quotes["close"][0]
        end_price = quotes["close"][-1]
        result = f"{start_price},{end_price}"
        logger.info(f"get_ticker_historical_prices returning: {result}")
        return result
    except Exception as e:
        error_msg = f"Error getting historical prices: {str(e)}"
        logger.error(error_msg)
        return error_msg

def get_return_percentage(prices_str: str) -> str:
    """Calculate return percentage. Input format: '75.0875,132.69'"""
    logger.info(f"get_return_percentage called with prices_str: {prices_str}")
    try:
        start_price, end_price = map(float, prices_str.split(","))
        return_pct = ((end_price - start_price) / start_price) * 100
        result = f"{return_pct:.2f}%"
        logger.info(f"get_return_percentage returning: {result}")
        return result
    except Exception as e:
        error_msg = f"Error calculating return: {str(e)}"
        logger.error(error_msg)
        return error_msg

# base_url="http://127.0.0.1:1234/v1"
# model="deepseek-r1-distill-qwen-7b" #NOTE: This model is not working as expected.
base_url=None
model="gpt-4o-mini"
LLM_API_KEY=os.getenv("LLM_API_KEY")

SYSTEM_PROMPT = """You are a stock return calculator. Your task is to calculate the return percentage for a stock between two dates.

IMPORTANT: You must make the actual function calls in sequence. Do not return the function calls as strings.
Use the Action and Action Input format exactly as shown in the example.

Follow these steps in order:

1. Get the stock symbol using get_symbol
2. Convert the dates to timestamps using convert_date_to_timestamp
3. Get the historical prices using get_ticker_historical_prices
4. Calculate the return percentage using get_return_percentage

Do not explain your thinking. Just make the function calls in sequence.
Use the output of each function as input for the next function.
Return only the final percentage.

Example:
Input: "What would have been the return of Apple from 2020-01-01 to 2020-12-31?"

Action: get_symbol
Action Input: "Apple"
Observation: "AAPL"

Action: convert_date_to_timestamp
Action Input: "2020-01-01,2020-12-31"
Observation: "1577836800,1609459200"

Action: get_ticker_historical_prices
Action Input: "AAPL,1577836800,1609459200"
Observation: "75.0875,132.69"

Action: get_return_percentage
Action Input: "75.0875,132.69"
Observation: "76.71%"

Final Answer: 76.71%
"""

# Initialize the LLM
llm = ChatOpenAI(
    model=model,
    base_url=base_url,
    api_key=LLM_API_KEY,
    temperature=0,
    max_tokens=1000,
    stop=["<think>", "###", "---", "**", "*", "```", "Explanation:", "Step", "First", "Next", "Then", "Finally", "Let me", "I will", "We need to"]
)

# Create tools using StructuredTool
tools = [
    StructuredTool.from_function(
        func=get_symbol,
        name="get_symbol",
        description="Get stock symbol for a company name. Example: get_symbol('Apple') -> 'AAPL'",
        args_schema=GetSymbolInput
    ),
    StructuredTool.from_function(
        func=convert_date_to_timestamp,
        name="convert_date_to_timestamp",
        description="Convert dates to Unix timestamps. Input format: '2020-01-01,2020-12-31' -> '1577836800,1609459200'",
        args_schema=ConvertDateInput
    ),
    StructuredTool.from_function(
        func=get_ticker_historical_prices,
        name="get_ticker_historical_prices",
        description="Get historical prices for a stock. Input format: 'AAPL,1577836800,1609459200' -> '75.0875,132.69'",
        args_schema=GetTickerHistoricalPricesInput
    ),
    StructuredTool.from_function(
        func=get_return_percentage,
        name="get_return_percentage",
        description="Calculate return percentage. Input format: '75.0875,132.69' -> '76.71%'",
        args_schema=GetReturnInput
    )
]

# Initialize the agent with CHAT_CONVERSATIONAL_REACT_DESCRIPTION type
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    early_stopping_method="force"
)

def process_query(query: str) -> str:
    """Process a user query and return the calculated return percentage."""
    try:
        print(f"\nProcessing query: {query}")
        
        # Run the agent with empty chat history
        result = agent.invoke({
            "input": query,
            "chat_history": []
        })
        print(f"\nAgent result: {result}")
        
        return result.get("output", "No result found")
        
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    # Example usage
    query = "What would have been the return of asian paints from 2020-01-01 to 2020-12-31?"
    print("\nStarting calculation...")
    result = process_query(query)
    print(f"\nFinal result: {result}") 