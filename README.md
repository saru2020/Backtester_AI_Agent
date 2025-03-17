# Equity Backtester AI Agent

This project implements an AI-powered equity Backtester that calculates historical returns for stocks using LLM APIs and couple of Finance data APIs.

## Demo
![demo](./media/Backtester_AI_Agent_demo_.mov)


## Features

- Get stock symbols from company names
- Calculate historical returns for any stock
- Date range-based return calculations
- Error handling and validation
- Interactive conversation with the AI agent

## Prerequisites

- Python 3.8 or higher
- OpenAI API access (or compatible API endpoint)
- Yahoo Finance API access (or any financial APIs, refer the blog for details)

Refer [this blog](./blog/blog.md) for more details

## Setup

1. Clone the repository:
```bash
git clone https://github.com/saru2020/Backtester_AI_Agent.git
cd equity_backtest_ai_agent_tryout
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install openai python-dotenv requests
```

4. Create a `.env` file in the project root with the following variables:
```env
YAHOO_FINANCE_SEARCH_API=<your-yahoo-finance-search-api-url>
YAHOO_FINANCE_QUERY_API=<your-yahoo-finance-query-api-url>
```

## Configuration

The project uses a local API endpoint by default. To modify the configuration:

1. Open `backtester_agent.py`
2. Update the `base_url` in the OpenAI client configuration if needed:
```python
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",  # Change this to your API endpoint
    api_key="something",
)
```

3. You can also modify the model being used:
```python
model = "deepseek-r1-distill-qwen-7b"  # Change this to your preferred model
```

## Usage

1. Run the script:
```bash
python backtester_agent.py
```

2. The script will process the example query:
```python
"What would have been the return of Apple from 2020-01-01 to 2020-12-31?"
```

3. To modify the query, edit the `messages` list in `backtester_agent.py`:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Your query here"},
]
```

## Example Queries

You can ask questions like:
- "What would have been the return of Microsoft from 2021-01-01 to 2021-12-31?"
- "Calculate the return for Google between 2019-01-01 and 2019-12-31"
- "What's the return percentage for Amazon from 2022-01-01 to 2022-12-31?"

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