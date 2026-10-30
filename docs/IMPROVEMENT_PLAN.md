# Improvement Plan

For the three stock-return examples on `main` at [4cc2fef](https://github.com/saru2020/Backtester_AI_Agent/tree/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088), prioritise runnable, predictable demonstrations. IMP-01 through IMP-03 have open pull requests; other items remain proposed. None is merged.

- **IMP-01, P1:** Reject invalid prices and standardise UTC date boundaries. Addressed in [#2](https://github.com/saru2020/Backtester_AI_Agent/pull/2).
- **IMP-02, P1:** Restore LangChain imports and provide runtime/configuration guidance plus offline CI. Addressed in [#3](https://github.com/saru2020/Backtester_AI_Agent/pull/3).
- **IMP-03, P1:** Encode company/ticker URL components and add HTTP connect/read timeouts. Addressed in [#4](https://github.com/saru2020/Backtester_AI_Agent/pull/4).
- **IMP-04, P1:** Make imports inert, validate explicit configuration and stop printing complete conversations by default. Proposed.
- **IMP-05, P1:** Correct retry, tool-response, state-reset and completion handling; test the orchestration separately from helper arithmetic. Proposed.

The three fixes pass 8, 5 and 6 offline tests; an independent combined tree passes 19. PR #3's hosted checks also pass. Baseline has no tests, and neither fixes nor CI are on main. Real endpoint/model paths remain unverified. The 2026-10-30 GitHub snapshot reported zero open issues.

Further items cover market-data meaning, reproducible dependencies, integrated offline checks and documentation. No integrated market/model guarantee or production deployment is proposed. [Detailed improvement plan](IMPROVEMENT_PLAN_DETAILED.md).
