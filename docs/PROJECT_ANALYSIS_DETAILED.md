# Project Analysis: Detailed

Based on `main` at [4cc2fef](https://github.com/saru2020/Backtester_AI_Agent/tree/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088) (2026-10-30). [Short analysis](PROJECT_ANALYSIS.md).

## Scope And Status

Three learning examples translate a company/date question into a closing-price percentage change. There is no portfolio engine, strategy execution, order placement or hosted service. All three source files were inspected. The recorded size is 766 Python code lines, excluding 183 comment and 152 blank lines.

The baseline history contains eight commits from 2025-03-18 to 2025-04-04, including filename changes that the old README missed. The 2026-10-30 GitHub snapshot returned zero open issues and zero open PRs. Local preparations below are separate from this baseline; no fix or CI workflow is asserted to be on main.

## Stack And Dependency Snapshot

[requirements.txt][requirements] has lower bounds, no lockfile and no declared Python version. An isolated Python 3.12.13 installation resolved 90 distributions. The following installed versions matched the saved [PyPI](https://pypi.org/) responses dated 2026-10-30; these are dated observations, not reproducible pins or promises about future installs.

| Package | Baseline constraint | Installed and saved latest |
| :--- | :--- | :--- |
| requests | >=2.31.0 | 2.34.2 |
| python-dotenv | >=1.0.0 | 1.2.4 |
| pydantic | >=2.6.0 | 2.14.2 |
| openai | >=1.12.0 | 3.41.0 |
| langchain | >=0.1.0 | 1.4.7 |
| langchain-openai | >=0.0.5 | 1.7.3 |
| langchain-core | >=0.1.0 | 1.7.3 |
| langchain-community | >=0.0.10 | 0.4.2 |
| pandas | >=2.2.0 | 3.1.0 |
| numpy | >=1.26.0 | 2.5.4 |
| typing-extensions | >=4.9.0 | 4.16.0 |
| ipykernel | Unconstrained | 7.4.0 |

The saved [Python lifecycle](https://endoflife.date/python) response dated 2026-10-30 lists 3.12.16 as the latest 3.12 patch, regular support ending 2025-04-02 and security EOL on 2028-10-31. It lists 3.15.2 as the latest overall release. Neither version was tested here; 3.12.13 was the checked interpreter. No major runtime migration is established.

Successful dependency resolution does not establish API compatibility. Baseline [script 2][script-2] fails importing `AgentType` from `langchain.agents` with LangChain 1.4.7. The prepared compatibility branch uses `langchain_classic.agents` and `langchain_core.tools`, explicitly requiring `langchain-classic>=1.0.8,<2`; 1.0.8 was already present in the resolved environment. Its checks emit a `langchain-community` deprecation warning.

The installed OpenAI 3.41.0 client exposed `chat.completions.create` and `beta.chat.completions.parse` under synthetic configuration and network denial. No request to either model API path was made. Pandas, NumPy and notebook tooling are not used by these three scripts and enlarge the installation without supporting their current calculation path.

## Architecture

| Entry point | Control and output | Import behaviour |
| :--- | :--- | :--- |
| [0_backtester_agent_prompt_chained.py][script-0] | Two structured model calls surround deterministic lookup, dates and arithmetic. Pydantic extraction includes a model-supplied validity/confidence gate; the last call supplies prose. | Loads dotenv and constructs `OpenAI`; the example is guarded by `main()`. |
| [1_backtester_agent.py][script-1] | Four JSON-schema tools, a dictionary of calculation state and repeated-call tracking. Ten conversation-loop iterations are allowed, with additional final-response calls. | Loads dotenv, constructs `OpenAI` and executes the example conversation at module scope. |
| [2_backtester_agent_langchain.py][script-2] | Four `StructuredTool` wrappers feed a conversational ReAct executor with five iterations. `process_query()` returns its output or an error string. | After imports resolve, loads dotenv and constructs `ChatOpenAI` and the executor; the example invocation is guarded. |

Data flows from a hard-coded Apple/date question to company search, date conversion, daily chart retrieval, first/last closing prices, arithmetic and model output. There is no interactive argument parser. The implementations duplicate four helpers rather than importing a shared core.

Search reads the first element of `quotes`. Script 0 returns its symbol; scripts 1 and 2 return the quote dictionary. History reads `chart.result[0].indicators.quote[0].close`, using only the first and last values. Dates and prices are held in tuples, dictionaries, Pydantic models or comma-separated strings, depending on the implementation. Nothing persists calculation records; `blog/` contains historical guides and `media/` contains demonstrations.

The calculation is `(last_close - first_close) / first_close * 100`. It is not annualised or total return. The code does not reconcile exchange/currency, actual observation dates, adjustment metadata, dividends, fees or slippage. Selecting the first search match does not establish the intended listing. No call to a separate hosted Backtester application appears in the scripts.

## Configuration

The [README](../README.md#quick-start) supplies illustrative environment preparation and numbered entry-point commands. Requirements installation succeeded without model or market calls, but no live entry-point run was verified. Real runs require deliberate model access and independently supplied, trusted market endpoints; model use can incur charges.

| Name | Contract in the source |
| :--- | :--- |
| `LLM_API_KEY` | Passed to the model client. All three scripts set `model` to `gpt-4o-mini` and `base_url` to `None` in source. |
| `YAHOO_FINANCE_SEARCH_API` | URL prefix ending immediately before the search value; helpers append the company name and language/region/quote-count parameters. Must return a compatible `quotes` array. |
| `YAHOO_FINANCE_QUERY_API` | URL prefix ending immediately before the ticker; helpers append `period1`, `period2` and daily interval parameters. Must return the chart structure above. |

All scripts call `load_dotenv()` at module scope. There is no baseline configuration validator or example settings file. Commented local-model alternatives require source changes; their authentication, structured-output support and reasoning behaviour were not retested. Endpoint availability, access terms, credentials, response schemas and real model paths remain unverified.

Do not import scripts 0 or 1 merely to inspect helpers: both construct clients, and importing a configured script 1 can initiate model and market traffic. Script 2 is not import-safe either, even once its dependency imports are corrected. The prepared setup branch adds placeholder-only configuration and Python 3.12 selection, not import isolation.

## Recorded Checks

All results in this section are dated 2026-10-30, not checks rerun by reading this documentation.

- Baseline syntax compilation succeeded without executing modules; unittest discovery found zero tests. No original coverage report exists.
- The resolved environment passed dependency consistency checks. Corrected pip-audit 2.10.1 invocation reported zero known vulnerabilities across 90 installed distributions. The earlier invalid invocation was not a successful audit; advisory results are time-limited.
- Baseline Ruff 0.17.3 returned findings, including with isolated configuration; full lint is not green. Bandit 1.9.4 reported six medium B113 findings, one per HTTP request without a timeout.
- Redacted tree and full-history secret scans reported zero findings. Git history contains eight commits; the history scanner reported seven scanned patch-bearing commits. These scans do not guarantee the absence of secrets.

| Prepared branch | Recorded passing tests | What was exercised |
| :--- | :--- | :--- |
| `fix/validate-return-inputs` | 8 | Real helper definitions extracted from source; finite-price/overflow checks, invalid/reversed dates, inclusive UTC end dates in three host timezones and a network-denial negative control. |
| `fix/langchain-compatibility` | 5 | Real legacy executor with `FakeListChatModel`, synthetic configuration, two mocked market responses, four-tool execution, error handling and network denial. |
| `fix/bounded-market-requests` | 6 | Real HTTP helper definitions with mocked requests; URL encoding, connect/read timeouts, success/empty/error contracts and network denial. |

The recorded branch test command, with interpreter and test paths made repository-relative, is:

```bash
python -I -B -m unittest discover -s tests -v
```

Use that only in a prepared branch with its tested environment, not as evidence of tests on baseline. All three branches also passed the narrow Ruff selector `E9,F63,F7,F82` and whitespace checks. The HTTP branch's application-only Bandit check returned zero findings. The compatibility branch recorded a fresh dependency installation, `python -m pip check`, repeat tests and actionlint success; these were local checks, not an Ubuntu Actions run.

The helper suites deliberately avoid module initialisation. The compatibility suite disables dotenv and tracing and supplies predetermined model actions, so its successful workflow does not prove independent reasoning, correct argument propagation for arbitrary queries or real provider compatibility. Saved negative controls rejected baseline behaviour. Combined branch validation, whole-entry-point tests for scripts 0/1, coverage, live market/model paths and cross-platform execution are not established.

## Risks And Code Quality

Severity describes the current demonstration, not a deployed trading system.

| Risk | Severity | Evidence and plan |
| :--- | :--- | :--- |
| Dependency API mismatch | High | [Script 2 imports][script-2] fail in the resolved environment; minimum-only [requirements][requirements] permit incompatible majors. IMP-02, IMP-07. |
| Invalid return/date results | High | [Script 0][script-0] can divide by zero; all variants accept non-finite returns. [Script 2][script-2] uses local-time timestamps and excludes the final day, unlike scripts 0/1. Reversed ranges lack explicit rejection. IMP-01. |
| Market request reliability | Medium | All six requests interpolate company/ticker text without component encoding or timeouts; [script 0 search][script-0] parses JSON without checking HTTP status. IMP-03. |
| Import side effects and configuration | High | Client construction and dotenv loading happen at module scope; [script 1][script-1] also executes there. Configuration mistakes can cause failures or unintended external calls. IMP-04. |
| Orchestration correctness | Medium | [Script 1][script-1] records calls before success, skips repeats without tool responses, appends a completion inside the per-tool loop and breaks only that inner loop after a result. Successful symbol/date changes can leave downstream state stale. These paths need transcript tests; live failures were not reproduced. IMP-05. |
| Framework and error contracts | Medium | [Script 2][script-2] defines `SYSTEM_PROMPT` without passing it to the executor, annotates a dictionary-returning search as `str`, and mixes error strings with values. [Script 0][script-0] can pass missing prices into Pydantic construction. IMP-05, IMP-06. |
| Financial/data interpretation | Medium | The first listing and boundary closes are accepted without observation-date or currency reconciliation; malformed nested responses and null prices receive inconsistent handling. No total-return methodology is implemented. IMP-06. |
| Logging and local configuration hygiene | Medium | [Script 1][script-1] prints complete messages, arguments and responses; [script 2][script-2] enables verbose execution. Baseline [.gitignore][ignore] covers only the named dotenv file and virtual environment, not variants or bytecode. IMP-04. |
| Verification and reproducibility | Medium | No original tests, lock, runtime declaration or CI; full lint remains failing and typing is partial. Unused imports and [script 0's unused Strategy model][script-0] add maintenance noise. IMP-07, IMP-08. |
| Historical instructions | Low | Old entry-point names, omitted model-key configuration and production/reliability claims are not a current run contract. IMP-09. |

Error handling differs intentionally between examples: exceptions or tuples in script 0, dictionaries/`None` in script 1, and strings in script 2. Broad catches keep demonstrations running but can hide the failing stage. Pydantic shapes and tool schemas do not replace date, finite-value or provenance checks.

HTTP is synchronous, complete daily series are loaded to obtain two values, and repeated model calls grow transcripts and latency. Iteration limits do not bound request duration, total cost or memory. Prepared connect/read timeouts are inactivity limits, not a total wall-clock budget or response-size cap. No performance benchmark or scaling guarantee exists; extra service infrastructure is not justified for this one-query demonstration.

## Security And Delivery

There is no inbound API, user store, authentication/authorisation layer, CORS policy, database, container configuration or infrastructure deployment manifest to assess. Model authentication and trusted outbound endpoint configuration are the relevant boundaries. URL encoding does not validate endpoint hosts, redirects or provider access rights. Zero dependency and secret findings do not establish overall security.

Baseline has no declared build, lint/typecheck scripts, CI or active local hooks. The saved GitHub snapshot returned no Actions runs, repository webhooks, environments, deployments or check runs; Actions was enabled. Pages returned 404, and the installation endpoint returned 401, leaving connected GitHub App/service automation unverified. Empty status contexts are not passing CI. Publishing requires a separate automation check.

Only `fix/langchain-compatibility` prepares CI: Ubuntu, pull requests and main pushes, read-only contents permission, concurrency cancellation, a ten-minute timeout, pinned checkout/setup actions, dependency checks and synthetic tests. It has no deployment steps or service secrets. It is not baseline CI and has no verified remote run.

## Documentation And Strengths

The [original walkthrough](../blog/blog.md), [refined walkthrough](../blog/blog_refined.md), [debugging journey](../blog/agent_debugging_journey.md) and linked demo remain useful historical material. Their local-model setup, old filenames and claims of reliable results must not override the baseline evidence. Some guide links still use pre-rename source paths; follow the numbered entry points above.

The compact examples make orchestration approaches directly comparable. Arithmetic/date helpers are separable enough for offline tests; script 0 has structured output models, and script 1 overrides model-supplied history/return arguments with stored values. These are useful teaching mechanisms, not production validation.

[Detailed improvement plan](IMPROVEMENT_PLAN_DETAILED.md).

[script-0]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/0_backtester_agent_prompt_chained.py
[script-1]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/1_backtester_agent.py
[script-2]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/2_backtester_agent_langchain.py
[requirements]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/requirements.txt
[ignore]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/.gitignore