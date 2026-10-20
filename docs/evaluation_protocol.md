# Trading Model Evaluation Protocol

**Protocol:** v1  
**Status:** FROZEN  
**Frozen:** 2026-10-10 (Asia/Kolkata; UTC+05:30)  
**Contract amendment:** v1.1 (2026-10-10); these clarifications do not alter periods, metric definitions, or holdout authorization.
**Scope:** Research-period boundaries, holdout governance, result definitions, uncertainty reporting, and experiment records. This protocol does not authorize strategy implementation or evaluation runs.

This protocol is frozen before the next strategy-detector implementation or strategy experiment. A code or rule change informed by revealed holdout results starts a new research cycle; the same 2025 observations then cease to be an untouched validation set for that revised strategy.

## 1. Evaluation periods and source

All timestamps, interval boundaries, session buckets, and execution-event ordering are UTC. Evaluation periods are half-open intervals:

| Period | Included event timestamps | Use |
|---|---|---|
| Development | `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)` | Rule development, implementation, tests, and development-only comparisons |
| Sealed holdout | `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)` | One final, separately authorized out-of-sample evaluation |

Canonical evaluation input is the ordered Dukascopy XAUUSD bid/ask tick dataset. Tick timestamps are Unix epoch milliseconds interpreted as UTC. Preserve source order and equal timestamps; do not sort or deduplicate ticks. The canonical full-data manifest may hash and validate both periods, including 2025, without computing strategy outcomes.

Historical bars and decisions must follow the repository's actual replay semantics: evidence availability is the replay subject's availability timestamp; a decision cannot use evidence unavailable by that timestamp; execution entry requires a tick strictly later than decision availability; and exits are evaluated only after entry. Session labels and any bar-derived evidence must be computed from UTC timestamps and causally available observations, not from future bars or local machine time.

Data before a period's inclusive start may be used only as causal warm-up/context if required by the frozen strategy and replay. It cannot create a counted decision, entry, trade, or performance observation before the period. A trade is attributed to the evaluation period, UTC year, and session of its actual entry-fill timestamp. The holdout is not a development or debugging interval.

**Historical holdout exposure:** The owner confirms that only raw-data presence and integrity were checked on 2025. No strategy outcomes, metrics or charts were inspected. Do not describe the holdout status as unknown. The holdout prohibition still applies. This is an owner attestation, not technical sealing or independent proof of all prior activity. Retain the access prohibition below. If contrary records emerge, document them and reassess whether 2025 can serve as an untouched holdout before authorizing evaluation.

## 2. Holdout access and freeze gate

### Procedural rules

1. During development, strategy code, rule selection, parameter choice, debugging, and strategy tests may use development-period data only. Tests exercising data parsing or integrity may validate the complete raw dataset without producing or inspecting strategy outcomes.
2. Copilot and other AI coding agents, and all developers, must not inspect 2025 strategy results before explicit user authorization of the final evaluation.
3. No developer may use holdout outcomes to choose among variants, alter rules, tune parameters, select metrics, or debug performance.
4. Before any authorized holdout run, record and freeze: the clean code commit, strategy configuration SHA-256, development dataset/version, canonical dataset-manifest SHA-256, execution assumptions/configuration, metric definitions, and run command/environment. Record these in the variant log or its linked immutable result record before opening holdout results.
5. The final holdout evaluation requires explicit user authorization after the strategy and evaluation inputs are frozen. A verification run, test suite, data-integrity scan, or this protocol's documentation validation is not authorization.
6. If holdout results are revealed, any later change informed by them is a new research cycle. Do not present the same 2025 period as untouched validation for that revision.
7. If fewer than 100 completed holdout trades occur, report **INSUFFICIENT EVIDENCE** for the planned initial assessment. Do not change the strategy or mine 2025 to increase this count.

### Technical controls versus promises

These are procedural governance rules, not an access-control boundary. The source code, raw data, reports, and local filesystem remain technically accessible to repository users and tools. No encrypted dataset, permission gate, or automated prevention of reading 2025 outcomes is implemented or claimed. Before an authorized final run, enforce the procedure through review, restricted result handling, and audit records. The data-integrity verifier can inspect/hash raw 2025 files but must not calculate strategy outcomes.

The production `CoreV1TickExecution` path calls the canonical `evaluate_core_v1_pipeline` directly. Production callers may supply governed resolvers and components through a `pipeline_inputs_provider` that returns pipeline keyword arguments; it cannot replace the visible inputs or the canonical evaluator. `pipeline_builder` is a test-only escape hatch, and supplying one requires explicit `allow_custom_pipeline_builder=True`; opted-in custom builders must record their identity. Direct calls to `evaluate_core_v1_pipeline` do not require a test-only opt-in. Direct calls to public component evaluators remain possible, with evidence validation governed by each evaluator's own contract.

### Pre-run freeze record

The holdout is blocked until every item below is recorded and reviewed:

- Clean repository commit SHA containing the frozen strategy and evaluation implementation.
- Frozen strategy-configuration hash (SHA-256 over UTF-8 canonical JSON: sorted keys, compact separators, no NaN/Infinity).
- Development dataset period/version and canonical manifest SHA-256.
- Execution assumptions: bid/ask fill side, strict decision-time eligibility, position overlap, starting equity, contract size, quantity rules, account-currency conversion, spread treatment, commission, slippage, end-of-period position treatment, and session mapping.
- Frozen metric definitions and bootstrap settings in this protocol.
- Exact command, Python/dependency/environment versions, and a unique run/variant ID.
- Explicit user authorization and a record that no holdout-informed changes occurred after the freeze.

At this protocol freeze, current Step 1 files are uncommitted and no clean frozen evaluation commit/configuration has been recorded. Therefore **no holdout run is currently authorized or reproducible from a frozen strategy build**.

## 3. Result definitions

### Trade population, money, and R

- A **completed trade** has both an actual entry fill and an actual exit fill in the evaluation record. Count each execution position once, not each subject, signal, order, or fill.
- The primary period population consists of trades whose entry-fill timestamp is in the period. A trade that has not exited by the period boundary is reported as `OPEN_AT_PERIOD_END`, with its count, last executable-side mark-to-market value and unrealized P&L shown separately. It is not silently discarded, synthetically closed, or included as a completed realized trade.
- Monetary values are in the account currency explicitly configured for the execution run; report currency code. Also report price points and R-multiples where defined. Account currency, starting equity, and conversion source/rate policy are **UNVERIFIED until frozen in the execution configuration**.
- Gross realized P&L is the actual executable fill-price difference times filled quantity, contract size, and the frozen account-currency conversion, before explicit commission and slippage. Because actual long entries/exits use ask/bid and actual short entries/exits use bid/ask, this fill-based gross already includes the bid/ask spread. Net realized P&L is gross realized P&L minus entry and exit commission and explicitly modeled entry and exit slippage. Do not subtract spread a second time.
- Report spread cost separately as a diagnostic: at each entry/exit quote, calculate the executable-side distance from midpoint (ASK minus midpoint for buys; midpoint minus BID for sells), multiply by quantity, contract size and conversion, then sum per trade. This explains spread embedded in gross P&L; it is not an additional deduction from fill-based gross.
- Commission cost is the sum of configured per-unit-per-side commission times filled quantity across entry and exit. Slippage cost is the corresponding explicitly configured slippage amount across both sides. Report the two independently and reconcile `gross - commission - slippage = net`.
- Initial price-risk for a trade is the frozen execution contract's stop-loss price risk at actual entry fill, excluding commission/slippage, using the engine's explicit contract size and currency conversion. Trade net R is net realized P&L divided by this initial monetary price-risk. If that denominator is absent, non-finite, or non-positive, R is undefined; retain monetary results and mark the R result invalid rather than inventing a denominator.

### Required metrics

Report the following for the complete period and, where meaningful, separately by UTC year and entry session. Show development (2022, 2023, 2024) and holdout (2025) in clearly separate sections; never pool the holdout into development summaries.

1. **Completed trades:** count of positions with both fills. Also report open-at-period-end, unfilled, and rejected/no-trade counts separately; none is a completed trade.
2. **Trades per year and session:** assign each trade to UTC calendar year and frozen UTC session of its entry-fill timestamp. Include zero-trade strata.
3. **Gross P&L and net P&L:** sum trade values in account currency, plus price points and valid R totals. Gross uses actual executable fill prices; net deducts commission and slippage as defined above.
4. **Net expectancy per trade:** arithmetic mean net realized account-currency P&L per completed trade, and arithmetic mean net R over trades with valid R. Report valid-R denominator separately.
5. **Win rate:** count of completed trades with net P&L strictly greater than zero divided by completed trades. Report breakeven count (`net == 0`) separately. If there are no completed trades, report N/A.
6. **Average winner and loser:** arithmetic mean net account-currency P&L among winners and among losers separately; average loser is signed (negative). Also report each mean in valid R. Empty groups are N/A.
7. **Profit factor:** sum of positive net P&L divided by the absolute sum of negative net P&L. If gross losses are zero and gross profits are positive, report `+infinity (zero gross losses)` and the numeric field as null; if both are zero or there are no completed trades, report N/A. Here “gross profits/losses” means positive/negative components of net trade outcomes, not pre-cost gross P&L.
8. **Maximum drawdown:** on the chronological realized-equity curve below, report maximum peak-to-subsequent-trough loss in account currency and as a percentage of that prior peak. Report zero only if the curve never falls below its running peak.
9. **Costs:** report total and per-completed-trade spread (embedded diagnostic), commission, and slippage independently, in account currency and points where conversion permits. Reconcile costs to gross/net definitions; do not double-count spread.
10. **Return and loss-tail distribution:** report empirical trade net-P&L and valid net-R median, 5th, 25th, 75th, 95th percentiles, minimum/maximum, and 5% expected shortfall (mean of observations at or below the empirical 5th percentile). Use linear empirical quantile interpolation (NumPy method `linear`, Hyndman–Fan type 7); report sample counts and undefined R values.
11. **Uncertainty:** report 95% confidence intervals for net expectancy in currency and valid R, and maximum drawdown in currency and percent, using the dependence-aware block bootstrap below. Intervals are descriptive uncertainty estimates, not a guarantee or proof of edge.
12. **Stratification:** report the above core count, P&L, expectancy, win-rate, profit-factor, drawdown, and tail summaries by each UTC calendar year and entry session. Identify development-year results separately from holdout results; do not conceal empty/low-count strata.

### Equity curve, overlap, and bootstrap

- Use one active XAUUSD position maximum, consistent with the frozen execution contract. No overlapping-position accounting or concurrent capital reuse is allowed.
- Set starting equity explicitly from the frozen execution configuration. Starting equity and account currency are **UNVERIFIED until that config is committed and hashed**. Compound realized net P&L in executable order; do not include deposits/withdrawals. If an external cash flow is ever required, stop and revise/freeze the protocol before evaluation.
- Begin the realized equity curve with starting equity. Apply each completed trade's net P&L at its actual exit-fill event in UTC timestamp and canonical source-order sequence. Calculate running peaks from that series, including the initial point. Open positions remain separately reported as censored/unrealized at period end and do not alter primary realized equity or drawdown.
- For confidence intervals, form a UTC calendar-day sequence spanning the evaluated interval; retain all trades for each day as a cluster, including zero-trade days. The primary interval uses a moving-block bootstrap of five consecutive calendar-day clusters: draw valid consecutive blocks with replacement until the original number of days is reached, truncate the last block, and preserve trade order within and between sampled blocks. Use 10,000 replicates and NumPy `Generator(PCG64(20261010))`. Recalculate expectancy and drawdown for each replicate; use the 2.5th and 97.5th percentile (`linear`) as the 95% interval. Do not report an interval when there are fewer than five calendar days or no valid trade denominator. Report sensitivity intervals using 1-day and 10-day blocks when the sample permits.
- The five-day block length is a fixed protocol choice, not a universal dependence horizon or statistical-power guarantee. Report the 1/5/10-day sensitivity and note concentrated dates/sessions/regimes. Any formal power calculation requires an explicitly stated target effect size, outcome variance, dependence/block assumptions, type-I error, target power, and comparison/multiplicity plan, all established from development data before holdout access.

### Frozen UTC session labels

For reporting, use the existing mutually exclusive hour buckets represented by `src/session.py`, after converting the actual entry-fill timestamp to UTC. Boundaries are half-open:

- `Asia`: `[00:00, 07:00)` and `[21:00, 24:00)` UTC.
- `London`: `[07:00, 13:00)` UTC.
- `New York`: `[13:00, 17:00)` UTC.
- `Overlap`: `[17:00, 21:00)` UTC.

These are fixed UTC reporting labels, not claims about daylight-saving-adjusted local market sessions. Do not substitute the overlapping legacy entry filter in `src/session_entry.py` for this mutually exclusive reporting classification. Any strategy-specific session gate must be separately frozen in strategy configuration and evaluated only on development data before the holdout freeze.

### Runtime gates before evaluation

The `[start, end)` runtime boundary is verified at the batch, incremental, and tick-replay entry points. `eligible_decision_times()` includes timestamps at `start`, excludes timestamps at `end`, and adjacent windows process a shared timestamp only in the later window. Evidence: `tests/test_replay_engine.py::test_eligible_decision_times_excludes_end_boundary`, `tests/test_replay_engine.py::test_adjacent_replay_windows_process_shared_boundary_once`, `tests/test_incremental_replay.py::test_incremental_replay_respects_half_open_start_and_end_boundaries`, and `tests/test_tick_execution_replay.py::test_raw_ticks_are_clipped_to_half_open_replay_window` / `test_shared_window_boundary_is_excluded_then_included_once`. This records runtime verification only; it does not authorize a strategy replay or evaluation.

The reporting buckets above are fixed UTC reporting labels, not DST-aware market sessions. **Session-filtered experiments are BLOCKED** until a timezone-aware, daylight-saving-aware session policy is separately specified, implemented, tested, and frozen. Do not interpret the current hour-bucket code as DST-correct market-session logic.

## 4. Minimum evidence standard

At least **100 completed out-of-sample trades** is a preliminary operational floor for the planned initial holdout assessment, not proof of profitability, a universal sample-size guarantee, or sufficient statistical power.

- Fewer than 100 completed holdout trades means **INSUFFICIENT EVIDENCE** for the planned initial assessment. Report all observed results and uncertainty without tuning against 2025.
- Reaching 100 does not automatically establish an edge. Consider confidence intervals, effect size, costs, drawdown/loss tails, dependence, and concentration in years, sessions, dates, or regimes.
- Do not claim statistical power without a stated target effect, variance, dependence assumptions, alpha, desired power, and multiple-comparison correction. Do not derive or tune these from holdout outcomes.

## 5. Multiple comparisons and variant log

`docs/strategy_variant_log.csv` is the append-only experiment index. Before running any strategy experiment, assign a unique variant ID and record at least:

`variant_id`, `date_created_utc`, `code_commit`, `configuration_sha256`, `development_dataset_version`, `rule_parameter_changes`, `hypothesis`, `metrics_examined`, `result_location`, `decision`, `decision_rationale`, `holdout_outcomes_accessed`, and `parent_variant_id`.

Record every attempted strategy experiment, including failed runs, zero-trade runs, and abandoned variants. Use `holdout_outcomes_accessed` values `NO`, `YES`, or `UNKNOWN`; never infer or backfill missing historical records. Keep each experiment's result location immutable and link it from the row. Log changes to protocol/metrics before they are used to inspect results. Freeze the holdout independently of which development variant looks best.

Historical variant history is **INCOMPLETE / NOT RECONSTRUCTED**. No imaginary variants are backfilled. The CSV retains its stable experiment schema and currently contains only explicitly labeled characterization/contract-change records, not strategy-experiment results. No strategy experiment or holdout evaluation was run as part of the Step 2 documentation and validation task. Documentation tests and repository verification are protocol checks, not strategy experiments.

### Contract amendment v1.1 — liquidity evidence

- A rejection reaction must carry a finite price matching the side-specific observed extreme of its visible base bar at the reaction confirmation timestamp (low for BSL, high for SSL). An absent, inferred, or mismatched price is insufficient evidence. A custom reaction resolver's output is checked against that visible observation before acceptance.
- A liquidity breach is valid only when `breach_event_timestamp > level_availability_timestamp`. Equality is not a breach; the boundary is strict.
- These are evidence-validity requirements, not strategy variants or performance findings. They do not authorize a replay evaluation or use of the holdout.
- The variant-log rows `CONTRACT-20261010-REJECTION-PRICE` and `CONTRACT-20261010-BREACH-AFTER-AVAILABILITY` record these contract changes. They do not represent experiments; dataset/configuration fields are not applicable.

### Owner decisions for Step 3D

- Rejection remains wick-based: the side-specific wick return is the selected rejection evidence. A close-back-inside is evidence only and is never a rejection gate.
- The frozen nominal level-life table is M1: 1 day, M3: 1 week, M5: 1 month, M15: 3 months, H1: 6 months, H4: 12 months, and 1D: 12 months.
- Lifetime semantics are confirmed, but implementation is deferred. A level lives from swing confirmation until the first of a price sweep or maximum age reached in trading time; weekends and closed hours do not age it.
- It remains unresolved whether a level breached but never rejected is consumed at breach while a later rejection of that same breach remains eligible. No behavior is selected by this record.
- `ACCEPTANCE` reactions are out of scope for v1 and fail closed at the MSS, displacement, and pipeline boundaries.
- Cross-timeframe clustering is not implemented in Step 3D. There is no timeframe priority. Nearby levels may later be recorded as a stronger zone, but clustering must not gate detection. Any future tolerance must be volatility-based (a fraction of ATR), not fixed dollars.
- The daily/H4 boundary convention remains a working assumption pending read-only verification: TradingView OANDA, a 17:00 `America/New_York` cut using the timezone database. Weekly open is Sunday at 17:00 `America/New_York`. Do not replace timezone rules with fixed UTC offsets.
- Session classification for session-filtered experiments remains blocked pending DST-aware verification.
- The half-open replay boundary `[start,end)` remains blocked pending runtime verification.
- The ICT 2022 rejection model is first. Other variants, including the 2023 model, must be frozen separately. Permit one holdout evaluation per frozen model, and no more than one model evaluated on 2025 without a protocol change. No evaluation is authorized by this record.
- The owner confirms that only raw-data presence and integrity were checked on 2025. No strategy outcomes, metrics or charts were inspected. Do not describe the holdout status as unknown. The holdout prohibition still applies.

## 6. Unverified assumptions and authorization gate

The period bounds, UTC rules, metrics, session labels, bootstrap method, and procedural access policy are frozen here. The following implementation/run facts remain **UNVERIFIED** and must be confirmed from the frozen build/config before evaluation: actual starting equity and account currency; conversion source/rate; contract-size/point-value interpretation; realized-P&L accounting consistency with this protocol; explicit slippage and commission values; how execution output records open-at-period-end valuation; strategy-specific session filters; and replay warm-up behavior at period start.

If an unverified fact prevents computing any required metric or respecting causal/replay semantics, do not guess or silently omit it. Resolve it against implementation/configuration and record the decision in a versioned protocol amendment before the holdout is authorized. Protocol v1 remains frozen until then.

## 7. Step 2 activity record

- No strategy backtest, strategy experiment, swing/displacement/MSS implementation, or holdout outcome inspection was performed for this step.
- Validation is limited to protocol/log schema checks, repository tests, raw-data integrity verification, and whitespace checks.
- Owner-confirmed prior exposure is limited to raw-data presence/integrity checks; no strategy outcomes, metrics, or charts were inspected.
- Historical census/forensics tests were reviewed: the Dukascopy forensics tests calculate timestamp coverage, duplicate/gap and quote/OHLC validity metadata, not price ranges or returns; census fixtures use synthetic data dated in 2022, while the real-dataset census test checks Octa metadata only. No such test computed price ranges or returns on 2025 Dukascopy data.
