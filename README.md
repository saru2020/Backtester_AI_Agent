# Backtester AI Agent

This project demonstrates three different implementations of a stock backtesting agent that calculates historical returns using AI:

1. **Prompt Chaining Implementation** (`backtester_agent_prompt_chained.py`)
   - Uses pure prompt chaining to guide the model through the calculation steps
   - Simpler implementation but less structured
   - Good for understanding the basic flow of the calculation

2. **Function Calling Implementation** (`backtester_agent.py`)
   - Uses OpenAI's function calling feature
   - Partially agentic approach with structured tool definitions
   - Better error handling and validation
   - More robust than prompt chaining

3. **LangChain Implementation** (`backtester_agent_langchain.py`)
   - Uses the LangChain framework for full agentic behavior
   - Most structured and maintainable implementation
   - Built-in tools and agent management
   - Best for production use and extensibility

All three implementations provide the same core functionality:
- Calculate historical returns for stocks
- Use Yahoo Finance API for data
- Handle date ranges and price calculations
- Provide detailed step-by-step explanations

## Features

- Calculate historical returns for any stock
- Support for custom date ranges
- Detailed step-by-step explanations
- Error handling and validation
- Multiple implementation approaches

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
YAHOO_FINANCE_SEARCH_API=your_search_api_url
YAHOO_FINANCE_QUERY_API=your_query_api_url
```

## Usage

Choose the implementation that best suits your needs:

1. For simple usage with prompt chaining:
```bash
python backtester_agent_prompt_chained.py
```

2. For function calling implementation:
```bash
python backtester_agent.py
```

3. For LangChain implementation:
```bash
python backtester_agent_langchain.py
```

Example query:
```
What would have been the return of Apple from 2020-01-01 to 2020-12-31?
```

## Implementation Details

### 1. Prompt Chaining (`backtester_agent_prompt_chained.py`)
- Uses sequential prompts to guide the model
- Each step is handled by a separate prompt
- Simpler to understand but less robust
- Good for learning and prototyping

### 2. Function Calling (`backtester_agent.py`)
- Uses OpenAI's function calling feature
- Structured tool definitions
- Better error handling
- More robust than prompt chaining
- Good for production use

### 3. LangChain (`backtester_agent_langchain.py`)
_NOTE: I couldn't get this working with the DeepSeek model in my local but tested it only with OpenAI model(gpt-4o-mini)_
- Uses LangChain framework
- Full agentic behavior
- Built-in tools and agent management
- Most maintainable and extensible
- Best for complex applications

## Demo

here's the working demo of this agent with this user input:
> What would have been the return of Apple from 2020-01-01 to 2020-12-31?

https://github.com/user-attachments/assets/441f4f66-432a-4a01-9820-99de4784d325

## Error Handling

The script includes comprehensive error handling for:
- Invalid company names
- Date format errors
- API connection issues
- Data validation
- Calculation errors

## Limitations

- Requires valid API endpoints for Yahoo Finance data
- Date range must be in YYYY-MM-DD format
- Company names must be recognizable by the Yahoo Finance API
- Historical data availability depends on the stock's listing date

## Contributing

Feel free to submit issues and enhancement requests! 
