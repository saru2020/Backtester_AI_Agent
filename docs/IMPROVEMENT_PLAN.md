# Improvement Plan

For the three stock-return examples on `main` at [4cc2fef](https://github.com/saru2020/Backtester_AI_Agent/tree/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088), prioritise runnable, predictable demonstrations. All items remain Proposed; local preparation is not an open-PR or merged status.

- **IMP-01, P1:** Reject invalid prices and standardise UTC date boundaries. Prepared on `fix/validate-return-inputs`.
- **IMP-02, P1:** Restore LangChain imports and provide runtime/configuration guidance plus offline CI. Prepared on `fix/langchain-compatibility`.
- **IMP-03, P1:** Encode company/ticker URL components and add HTTP connect/read timeouts. Prepared on `fix/bounded-market-requests`.
- **IMP-04, P1:** Make imports inert, validate explicit configuration and stop printing complete conversations by default. Proposed.
- **IMP-05, P1:** Correct retry, tool-response, state-reset and completion handling; test the orchestration separately from helper arithmetic. Proposed.

The three prepared branches recorded 8, 5 and 6 passing synthetic tests respectively on 2026-10-30. Baseline has no tests; neither those fixes nor the prepared CI are on main. Combined behaviour and real endpoint/model paths remain unverified. The same dated GitHub snapshot reported zero open issues, so there are no issue numbers to triage.

Further items cover market-data meaning, reproducible dependencies, integrated offline checks and documentation. No integrated market/model guarantee or production deployment is proposed. [Detailed improvement plan](IMPROVEMENT_PLAN_DETAILED.md).
