from openai import OpenAI
from pydantic import BaseModel, Field
import requests
import datetime
from dotenv import load_dotenv
import os
load_dotenv()

base_url="http://127.0.0.1:1234/v1" #None
LLM_API_KEY=os.getenv("LLM_API_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=LLM_API_KEY,
)
model = "deepseek-r1-distill-qwen-7b" #"deepseek-r1-distill-qwen-7b" # "gpt-4o-mini" 

# -------------- XX -------------- 
# Models
# -------------- XX -------------- 

class StrategyExtraction(BaseModel):
    """First LLM call: Extract basic strategy details from user input"""
    
    company_name: str = Field(description="The name of the company to be backtested")
    start_date: str = Field(description="The start date of the strategy in the format YYYY-MM-DD")
    end_date: str = Field(description="The end date of the strategy in the format YYYY-MM-DD")
    isValidStrategy: bool = Field(description="Whether the strategy is valid or not. True if there's a company name, start date, end date and the dates are valid, else False")
    confidence_meter: float = Field(description="The confidence level of the strategy between 0 and 100 based on the validity of the strategy, it should synonymously be the same as the isValidStrategy field")
        
class Strategy(BaseModel):
    """Second LLM call: Extract strategy details from the yahoo query API and convert the dates to unix timestamps with the convert_date_to_timestamp tool"""
    
    symbol: str = Field(description="The symbol of the company to be backtested")
    start_date: int = Field(description="The unix timestamp of the start date of the strategy")
    end_date: int = Field(description="The unix timestamp of the end date of the strategy")
    
class HistoricalPrices(BaseModel):
    """Third call to Yahoo Finance API: Get the historical prices of the company using the get_ticker_historical_prices tool"""
    
    # symbol: str = Field(description="The symbol of the company to be backtested")
    start_date: int = Field(description="The unix timestamp of the start date of the strategy")
    end_date: int = Field(description="The unix timestamp of the end date of the strategy")
    start_date_price: float = Field(description="The price of the company at the start date of the strategy")
    end_date_price: float = Field(description="The price of the company at the end date of the strategy")

class StrategyResult(BaseModel):
    """Fourth call to get the return percentage of the strategy using the get_return_percentage tool"""

    start_date_price: float = Field(description="The price of the company at the start date of the strategy")
    end_date_price: float = Field(description="The price of the company at the end date of the strategy")
    return_percentage: float = Field(description="The return percentage of the strategy")

class StrategyUserResponse(BaseModel):
    """Fifth LLM call: Generate a user response to the strategy based on the strategy result"""

    strategy_result: StrategyResult = Field(description="The result of the strategy")
    user_response: str = Field(description="Generate a user response to the strategy based on the strategy result")

# -------------- XX -------------- 
# Functions / APIs / Tools
# -------------- XX -------------- 

def convert_date_to_timestamp(start_date: str, end_date: str):
    dt1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')  
    dt1 = dt1.replace(tzinfo=datetime.timezone.utc)  # Set UTC timezone
    dt2 = datetime.datetime.strptime(end_date, '%Y-%m-%d')  
    dt2 = dt2.replace(tzinfo=datetime.timezone.utc)  # Set UTC timezone
    dt2 = dt2 + datetime.timedelta(days=1)  # Move to the next day to get the EOD timestamp
    return int(dt1.timestamp()), int(dt2.timestamp())

def get_symbol(search_query: str):
    print('inside get_symbol - search_query: ', search_query)
    """This is NOT a publically available API that returns the symbols for a given company name."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(
        f"{os.getenv('YAHOO_FINANCE_SEARCH_API')}{search_query}&lang=en-US&region=US&quotesCount=6",
        headers=headers
    )
    # print('inside get_symbol - response: ', response)
    data = response.json()
    # print('inside get_symbol - data: ', data)
    if "quotes" in data and len(data["quotes"]) > 0 and "symbol" in data["quotes"][0]:
        return data["quotes"][0]["symbol"]
    else:
        return None

def get_ticker_historical_prices(symbol: str, start_date_timestamp: int, end_date_timestamp: int):
    print('inside get_ticker_historical_prices - symbol: ', symbol)
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
        if "chart" not in data or "result" not in data["chart"] or "quote" not in data["chart"]["result"][0]["indicators"]:
            raise ValueError("No data available for the specified ticker and date range")
            
        quotes = data["chart"]["result"][0]["indicators"]["quote"][0]
        if "close" not in quotes:
            raise ValueError("No closing price data available")
            
        start_date_timestamp_price = quotes["close"][0]
        end_date_timestamp_price = quotes["close"][-1]
        
        if start_date_timestamp_price is None or end_date_timestamp_price is None:
            raise ValueError("Missing price data for start or end date")
            
        return start_date_timestamp_price, end_date_timestamp_price
        
    except requests.RequestException as e:
        print(f"Error making request to Yahoo Finance API: {e}")
        return None, None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing data: {e}")
        return None, None
    
    
def get_return_percentage(start_date_price: float, end_date_price: float):
    return ((end_date_price - start_date_price) / start_date_price) * 100


# -------------- XX -------------- 
# LLM calls
# -------------- XX -------------- 

def extract_strategy_details(user_input: str):
    """First LLM call: Extract basic strategy details from user input"""
    result = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts basic strategy details from user input."},
            {"role": "user", "content": user_input}
        ],
        response_format=StrategyExtraction
    )
    return result.choices[0].message.parsed

def query_yahoo_finance(query: str):
    symbol = get_symbol(query)
    # print('query_yahoo_finance - symbol: ', symbol)
    return symbol

def get_historical_prices(symbol: str, start_date: int, end_date: int):
    start_date_price, end_date_price = get_ticker_historical_prices(symbol, start_date, end_date)
    # print('start_date_price: ', start_date_price)
    # print('end_date_price: ', end_date_price)
    return HistoricalPrices(start_date=start_date, end_date=end_date, start_date_price=start_date_price, end_date_price=end_date_price)

def get_strategy_result(start_date_price: float, end_date_price: float):
    return_percentage = get_return_percentage(start_date_price, end_date_price)
    # print('return_percentage: ', return_percentage)
    return StrategyResult(start_date_price=start_date_price, end_date_price=end_date_price, return_percentage=return_percentage)


def generate_user_response(strategy_result: StrategyResult):
    """Fifth LLM call: Generate a user response to the strategy based on the strategy result"""
    result = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates a user response to the strategy based on the strategy result emphasizing the return percentage."},
            {"role": "user", "content": f"Generate a user response to the strategy based on the strategy result for {strategy_result}"}
        ],
        response_format=StrategyUserResponse
    )
    return result.choices[0].message.parsed

def main():
    """Main function to run the backtester agent"""
    user_input = "I want to know how much I would have made if I had invested in amazon from 2023-01-01 to 2023-12-31"
    print('user_input: ', user_input)
    strategy_details = extract_strategy_details(user_input)
    print('strategy_details: ', strategy_details)
    if not strategy_details.isValidStrategy or strategy_details.confidence_meter < 50:
        print("Gate check failed: Invalid strategy details")
        return
    
    symbol = query_yahoo_finance(strategy_details.company_name)
    print('symbol: ', symbol)

    start_date_timestamp, end_date_timestamp = convert_date_to_timestamp(strategy_details.start_date, strategy_details.end_date)
    print('start_date_timestamp: ', start_date_timestamp)
    print('end_date_timestamp: ', end_date_timestamp)

    historical_prices = get_historical_prices(symbol, start_date_timestamp, end_date_timestamp)
    print('historical_prices: ', historical_prices)

    strategy_result = get_strategy_result(historical_prices.start_date_price, historical_prices.end_date_price)
    print('strategy_result: ', strategy_result)

    user_response = generate_user_response(strategy_result)
    print('user_response: ', user_response)
    print('Final output: ', user_response.user_response)

if __name__ == "__main__":
    main()

