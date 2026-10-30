# Project Analysis

Based on `main` at [4cc2fef](https://github.com/saru2020/Backtester_AI_Agent/tree/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088), reviewed 2026-10-30, this is a learning project for developers comparing prompt chaining, function calling and LangChain. Its three scripts contain 766 Python code lines and calculate a single stock's closing-price return, not portfolio or strategy performance.

The prompt-chain version uses structured responses, the function-calling version maintains calculation state, and LangChain wraps four tools. Each depends on configured Yahoo-compatible market endpoints and a model; calculation data stays in memory and console output.

The top three risks are unbounded, unencoded HTTP requests; invalid or non-finite prices and inconsistent date boundaries; and dependency drift that breaks baseline LangChain imports. Scripts 0 and 1 construct model clients on import; script 1 also starts its conversation loop.

Baseline has no tests. Separate prepared branches recorded 8 return/date, 5 compatibility and 6 HTTP tests passing on 2026-10-30 with synthetic inputs and network denial. These are branch checks, not an integrated market/model guarantee or changes on main. The [README quick start](../README.md#quick-start) explains configuration and unverified live commands. [Detailed analysis](PROJECT_ANALYSIS_DETAILED.md).
