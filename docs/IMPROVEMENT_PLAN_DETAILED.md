# Improvement Plan: Detailed

Plan for `main` at [4cc2fef](https://github.com/saru2020/Backtester_AI_Agent/tree/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088), reviewed 2026-10-30. [Short plan](IMPROVEMENT_PLAN.md). [Detailed analysis](PROJECT_ANALYSIS_DETAILED.md).

P1 means a blocked or misleading demonstration; P2 means maintainability or evidence worth improving. No P0 critical finding is established. Effort is S (small), M (several related changes) or L (substantial redesign). Risk describes implementing the item.

## Prepared Branches

These branches were prepared independently from the baseline. Status remains **Proposed** because publication and review are not verified. No code fix or CI workflow below is on the analysed main commit. PR numbers and links must come from actual publication, not inference.

| Item | Local branch | Recorded check on 2026-10-30 |
| :--- | :--- | :--- |
| IMP-01 | `fix/validate-return-inputs` | 8 synthetic return/date tests passed. |
| IMP-02 | `fix/langchain-compatibility` | 5 synthetic compatibility tests passed; local install, dependency consistency and actionlint checks passed. |
| IMP-03 | `fix/bounded-market-requests` | 6 synthetic HTTP tests passed; application-only Bandit returned zero findings. |

Each branch also passed narrow fatal/undefined-name lint and whitespace checks. These are separate dated branch checks, not a combined suite, whole-program coverage or a real market/model guarantee. The helper tests avoid module initialisation; the compatibility tests use predetermined fake-model actions and mocked market responses. See [verification limits](PROJECT_ANALYSIS_DETAILED.md#recorded-checks).

## Fixes And Compatibility

### IMP-01: Validate Prices And Dates

P1. Effort S. Risk low, limited to invalid-input behaviour. Dependencies: none. Status: Proposed, prepared on `fix/validate-return-inputs`.

**Problem/evidence:** [Price/date risks](PROJECT_ANALYSIS_DETAILED.md#risks-and-code-quality) occur in [script 0][script-0], [script 1][script-1] and [script 2][script-2]: zero division, non-finite returns, reversed ranges and inconsistent date boundaries.

**Change/benefit:** Require a positive finite starting price, non-negative finite ending price and finite return. Use UTC with an inclusive final calendar day; reject invalid/reversed dates while preserving each variant's result/error format. This prevents impossible or host-dependent calculations.

**Acceptance:** Profit, loss, unchanged price, zero ending price, invalid/non-finite values, overflow, same-day ranges and three host timezones pass synthetic tests. Recheck alongside IMP-03 because both change the same scripts.

### IMP-02: Restore LangChain Compatibility And Setup

P1. Effort S. Risk medium, framework compatibility. Dependencies: none for the isolated fix. Status: Proposed, prepared on `fix/langchain-compatibility`.

**Problem/evidence:** [Script 2 imports][script-2] fail with the [dated dependency resolution](PROJECT_ANALYSIS_DETAILED.md#stack-and-dependency-snapshot); baseline declares neither runtime nor offline checks.

**Change/benefit:** Use `langchain_classic.agents` and `langchain_core.tools`; explicitly require `langchain-classic>=1.0.8,<2`. Preserve the legacy executor rather than redesign it. Prepare Python 3.12 selection, placeholder configuration, dotenv-variant/bytecode ignores and checks-only Ubuntu CI. The recorded compatible package set is in the analysis, not a new lock.

**Acceptance:** Fresh installation and dependency checks pass; the real executor constructs and completes a synthetic four-tool workflow without model/market traffic or dotenv loading. Validate CI syntax and its first real run separately. The community-package warning and real provider compatibility remain unresolved.

### IMP-03: Bound And Encode Market Requests

P1. Effort S. Risk low, request formatting and failure timing. Dependencies: none. Status: Proposed, prepared on `fix/bounded-market-requests`.

**Problem/evidence:** The [six HTTP helpers](PROJECT_ANALYSIS_DETAILED.md#risks-and-code-quality) in [script 0][script-0], [script 1][script-1] and [script 2][script-2] omit timeouts and component encoding. Script 0 search omits status checking.

**Change/benefit:** Encode company names as query values and tickers as path components, add `(5, 30)` connect/read timeouts and check search status before JSON parsing. Preserve existing failure contracts and avoid malformed URLs or indefinite inactivity.

**Acceptance:** Mocked success, empty, HTTP-error and timeout cases pass for all six requests; reserved characters remain one component. Do not describe these timeouts as a total execution deadline, redirect restriction or response-size cap.

## Security And Enhancements

### IMP-04: Make Startup Explicit And Logging Restrained

P1. Effort M. Risk medium, entry-point behaviour. Dependencies: none. Status: Proposed.

**Problem/evidence:** [Import/configuration risks](PROJECT_ANALYSIS_DETAILED.md#configuration) and [verbose logging](PROJECT_ANALYSIS_DETAILED.md#risks-and-code-quality) arise in all scripts, especially [script 1's module-level loop][script-1].

**Change/benefit:** Move dotenv loading, client construction and execution behind explicit entry points. Validate required settings and trusted URL schemes/hosts before SDK construction. Make model/base URL deliberate configuration, and make full transcripts opt-in rather than default output. This permits safe imports and reduces unintended calls and disclosure.

**Acceptance:** Import each module with missing settings under network denial and mocked dotenv; nothing loads configuration or constructs a client. Invalid configuration fails locally. Default logs omit credentials and complete prompts/responses. Placeholder configuration remains non-functional; no live key is required for tests.

### IMP-05: Test And Correct Orchestration State

P1. Effort M. Risk medium, conversation protocol. Dependencies: IMP-04 for import-safe integration tests; IMP-02 for script 2. Status: Proposed.

**Problem/evidence:** [Script 1][script-1] tracks attempted calls before success, can leave unmatched tool messages and stale downstream state, and does not exit its outer loop immediately after a result. [Script 2][script-2] never supplies its defined `SYSTEM_PROMPT` to the executor. See [code-quality evidence](PROJECT_ANALYSIS_DETAILED.md#risks-and-code-quality).

**Change/benefit:** Pair tool calls with responses, define retry rules, invalidate dependent values and terminate on completion. Deliberately apply or remove the unused framework prompt. Keep model-supplied prices subordinate to validated stored data.

**Acceptance:** Synthetic transcripts cover multiple tool calls, failed identical retries, changed symbols/dates, malformed arguments, early model completion and iteration exhaustion. Completion makes no extra model request. Do not treat a predetermined fake answer as proof of reasoning reliability.

### IMP-06: Define Market Data And Return Meaning

P2. Effort M. Risk medium, data interpretation. Dependencies: IMP-01 and IMP-03. Status: Proposed.

**Problem/evidence:** History helpers in [script 0][script-0], [script 1][script-1] and [script 2][script-2] select boundary closes without validating observation dates or adjustment metadata; script 2 search returns a dictionary despite its string contract. [Architecture](PROJECT_ANALYSIS_DETAILED.md#architecture) explains why this is not total return.

**Change/benefit:** Validate nested quote/chart responses and distinguish typed failures from data. Report the selected listing, observed dates, currency and calculation method; reject ambiguity or missing boundaries. Bound supported date ranges. This makes a displayed percentage interpretable without building a portfolio engine.

**Acceptance:** Fixtures cover absent quotes, malformed charts, null/empty closes, non-trading dates and ambiguous listings. Output distinguishes observed closes from requested dates and makes no unsupported dividend, adjustment or currency claim. Real endpoint access and terms require separate confirmation.

## Upgrades, Testing And Infrastructure

### IMP-07: Make Dependencies Reproducible

P2. Effort M. Risk medium, resolution changes. Dependencies: IMP-02 and the combined checks in IMP-08. Status: Proposed.

**Problem/evidence:** [requirements.txt][requirements] permits incompatible major versions, includes unused direct packages and does not record an exact environment. See the [2026-10-30 registry/runtime snapshot](PROJECT_ANALYSIS_DETAILED.md#stack-and-dependency-snapshot).

**Change/benefit:** Evaluate Python 3.12.16, the saved latest security-series patch, against the checked 3.12.13 environment. Record a reproducible supported dependency set, reviewing unused pandas/NumPy/notebook requirements and the community-package warning. Use the analysis table as the tested package reference; do not assume a move to the saved latest Python 3.15.2 is compatible.

**Acceptance:** Fresh installation, dependency audit and offline suites pass with recorded versions. Document any changed API or result contract before upgrading. No new runtime version is claimed tested until its checks run; no major framework migration is part of the prepared fix.

### IMP-08: Establish Combined Offline Checks

P2. Effort M. Risk low, test/delivery changes. Dependencies: IMP-01, IMP-02 and IMP-03 initially; IMP-04/IMP-05 extend module and transcript coverage. Status: Proposed; only initial CI is prepared under IMP-02.

**Problem/evidence:** Baseline has zero tests, failing full lint, partial typing and no CI/hooks. [Recorded checks](PROJECT_ANALYSIS_DETAILED.md#recorded-checks) cover separate branches, while [delivery metadata](PROJECT_ANALYSIS_DETAILED.md#security-and-delivery) cannot exclude external application automation.

**Change/benefit:** Combine overlapping fixes in an isolated validation checkout and run all suites under synthetic settings with socket-denial negative controls. Establish explicit lint scope and useful helper types, removing dead imports/models without mass reformatting. Retain checks-only CI; no service infrastructure is needed.

**Acceptance:** Combined tests and dependency checks pass without credentials, dotenv or application traffic; baseline lint debt is documented rather than labelled green. Qualify external automation before publication and verify the Ubuntu workflow after publication. Record coverage gaps and dates; do not substitute test counts for live integration evidence.

## Documentation

### IMP-09: Keep Instructions Aligned With Evidence

P2. Effort S. Risk low, documentation only. Dependencies: no code merge; publication status must be checked when available. Status: Proposed, prepared on `docs/project-analysis`.

**Problem/evidence:** The [baseline README][readme] names missing entry points, omits model-key configuration and overstates reliability. Historical guides retain earlier setup assumptions.

**Change/benefit:** Provide a short corrected [README](../README.md), four linked documents, baseline-pinned code evidence and clearly dated branch results. Preserve historical guides and the demo without presenting them as current guarantees.

**Acceptance:** Both short documents are strictly below 250 words and end with detail links. Check words, links, rendering, spelling, punctuation and hooks; only add real PR links after publication. Distinguish prepared work from main, disclose import behaviour, zero original tests, scan limitations and untested real paths.

## Open Issues And Order

The saved GitHub query on 2026-10-30 returned **zero open issues**. There are no issue numbers to classify, close or reference. The same snapshot contained no open PRs; later publication is not verified by these documents.

1. Quick wins: review IMP-01, IMP-02 and IMP-03 independently, then combine their overlapping script changes for IMP-08 checks. Publish documentation under IMP-09 with verified statuses and actual PR links.
2. Next: IMP-04 and IMP-05 make complete offline execution testable; IMP-06 makes the output's financial meaning explicit. Evaluate IMP-07 with the combined tests available.
3. Later: consider a framework rewrite only if maintaining the small legacy example becomes impractical. No production deployment or full strategy engine is an acceptance criterion.

[script-0]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/0_backtester_agent_prompt_chained.py
[script-1]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/1_backtester_agent.py
[script-2]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/2_backtester_agent_langchain.py
[requirements]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/requirements.txt
[readme]: https://github.com/saru2020/Backtester_AI_Agent/blob/4cc2fef8f39035203d0fd1d246ec7aca8c4ef088/README.md