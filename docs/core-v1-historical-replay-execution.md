# Core v1.0 Historical Replay Execution Specification

**Status:** Core v1.0 historical execution policy. This document specifies the historical execution boundary only. It does not authorize implementation changes to existing replay, execution, broker, or OHLC research modules.

**Scope:** Canonical ordered Dukascopy XAUUSD bid/ask ticks through decision-set arbitration, market execution, position lifecycle, and realized result. This specification does not change `HistoricalReplay`, legacy batch replay, or existing OHLC research simulation.

Normative words **MUST**, **MUST NOT**, and **NO TRADE** describe the approved policy. Any item labeled **UNSPECIFIED** is not an implementation choice; it requires explicit resolution before code relies on it.

**Decision identity ≠ execution intent ≠ order ≠ position ≠ trade.** A `ReplaySubject` identifies evidence and a decision only. Subjects discarded from execution arbitration remain preserved as valid decisions; arbitration does not invalidate their evidence.

## 1. Source Contracts and Terms

- **Tick:** A canonical Dukascopy quote containing UTC `timestamp`, positive `bid`, positive `ask`, and stable source order/provenance. Input order is preserved; duplicate timestamps are valid. Invalid or crossed quotes are rejected by ingestion.
- **Availability group:** The atomic set of finalized replay bars sharing one UTC `available_at`. Tick aggregation assigns a completed bar the timestamp of the tick that causes its emission. This is simulated event-time availability, not measured feed arrival time.
- **ReplaySubject:** An evidence/decision identity consisting of `availability_timestamp` and `base_bar_timestamp`. It is not an order, trade, or position.
- **Decision time:** Exactly `subject.availability_timestamp`. `base_bar_timestamp` MUST NOT be used as execution time.
- **Decision set:** All decisions for subjects emitted at one availability timestamp, considered together before execution and risk arbitration.
- **Execution tick:** A canonical tick with `tick.timestamp > decision_time` (strict inequality). Ticks at the decision timestamp are ineligible for entry, irrespective of source order.
- **Active position:** An open XAUUSD position after an actual entry fill and before an actual exit fill. Core v1.0 permits at most one.

Repository grounding: `src/dukascopy_ticks.py` preserves source order and quote provenance; `src/tick_aggregation.py` aggregates bid/ask OHLC and defines simulated availability; `src/replay_subject.py` defines subject identity; `src/replay_engine.py` groups availability and emits subjects; strategy qualification is separate from execution in `src/replay_strategy.py`. Existing `src/event_backtester.py` resolves declared-price plans against OHLC and is not this tick execution engine.

## 2. Causal Invariants

1. The engine MUST consume the canonical ordered Dukascopy bid/ask tick stream. It MUST NOT reorder or deduplicate equal-timestamp ticks.
2. Every subject, decision, intent, order, entry fill, position, exit, and result MUST retain the originating `ReplaySubject` (or a lossless reference to it).
3. `decision_time` MUST equal the subject's UTC `availability_timestamp`. `base_bar_timestamp` identifies the base bar for evidence visibility only; it MUST NOT determine eligibility or execution time.
4. Evidence for a subject MUST be built only from data visible as of `decision_time`, respecting that subject's base-bar prefix and existing availability/completeness rules.
5. An entry fill MUST use a valid canonical quote whose timestamp is strictly greater than `decision_time`. The tick that made the decision, and every tick with the same timestamp, MUST NOT fill the new position.
6. A subject count MUST NOT be interpreted as an order count or trade count. Each subject can create at most one execution intent.
7. A fill MUST use an observed eligible tick's executable bid or ask according to the side rules in Section 7. Midpoint, OHLC values, strategy-declared entry price, and synthetic prices MUST NOT be used as historical tick fills.
8. Exit monitoring MUST begin only after the entry fill event. A position MUST NOT exit on the same quote event used to enter it.
9. No synthetic ticks or prices may be introduced across missing intervals, weekends, or other data gaps.
10. At all times, active XAUUSD position count MUST be at most one. The engine MUST NOT automatically reverse a position.
11. The approved per-trade risk baseline is 1%; the implementation MUST NOT increase it. Existing legacy paths and their outputs remain unchanged.
12. Costs MUST be represented explicitly/configurably. A zero-cost run, if supported for diagnostics, MUST be identified as such and MUST NOT be treated as final validation.
13. Unsupported order types and conflicting actionable same-availability directions MUST resolve to NO TRADE, not silent approximation or callback-order priority.
14. For same-direction actionable decisions at one availability, only the decision for the latest `base_bar_timestamp` is an execution candidate. Earlier decisions remain independently recorded as valid and do not create intents.
15. If an actionable decision set arrives while a position is active, every execution candidate from that set MUST be rejected as `NO_TRADE_ACTIVE_POSITION`. Historical intents are never queued or deferred.
16. A market intent waits for the first strictly-later valid canonical tick. There is no arbitrary time-based expiry. If input ends first, record `UNFILLED_AT_END_OF_INPUT` without a synthetic fill.
17. Risk is based on account equity immediately before entry: `risk_budget = equity * 0.01`.
18. Execution contract configuration MUST explicitly provide contract size, minimum quantity, quantity step, and required point-value/account-currency conversion information. Missing or invalid required metadata MUST produce `NO_TRADE_INVALID_EXECUTION_CONTRACT`; broker defaults MUST NOT be inferred.
19. Execution identities MUST use a collision-safe namespace containing `execution_run_id + ReplaySubject.subject_id + order discriminator`. The subject ID MUST remain a decision identity and MUST NOT itself be repurposed as a trade ID.

## 3. State Machine

Decision/intent and position lifecycles are related but distinct. A NO TRADE decision does not create an order or position.

```text
Decision:
  OBSERVED
    -> NO_TRADE
    -> ACTIONABLE

Decision set / arbitration:
  COLLECTING
    -> ARBITRATED_NO_TRADE (conflicting directions)
    -> SELECT_LATEST_SAME_DIRECTION (same-direction set)
    -> NO_TRADE_ACTIVE_POSITION
    -> INTENT_CREATED (selected candidate only)

Intent/order:
  CREATED
    -> WAITING_FOR_LATER_TICK
    -> FILLED
    -> NO_TRADE (unsupported, rejected by arbitration/risk, or otherwise ineligible)
    -> UNFILLED_AT_END_OF_INPUT

Position:
  OPEN (only after entry fill)
    -> EXIT_TRIGGERED (on a later eligible quote)
    -> CLOSED (after exit fill)
```

- `ACTIONABLE` means the strategy's existing decision is eligible to be considered for execution; it does not guarantee an intent or fill.
- `FILLED` requires a qualifying later tick and a permitted market order.
- `UNFILLED_AT_END_OF_INPUT` records that no qualifying tick was observed before the supplied stream ended. It MUST NOT be reported as a fill or position.
- A permitted market intent waits for the first strictly-later valid canonical tick, with no arbitrary time-based expiry. It is not filled synthetically across gaps, weekends, sessions, or end of input.
- Historical signals are not queued or deferred. An actionable candidate arriving while a position is active is rejected as `NO_TRADE_ACTIVE_POSITION`.

## 4. Same-Availability Decision-Set Arbitration

1. The replay/execution boundary MUST collect every decision whose subject has the same `availability_timestamp` into one decision set before creating or filling any intent from that set.
2. Each subject and its decision MUST remain independently recorded. Grouping is an arbitration boundary, not identity coalescing.
3. Determine the directions of **actionable** decisions in the set. If both LONG and SHORT are present, the entire conflicting actionable group MUST be rejected as NO TRADE. The engine MUST NOT select a winner based on callback order, subject iteration order, or source row order.
4. Non-actionable decisions do not create direction conflicts.
5. If actionable decisions all have the same direction, select the subject with the latest `base_bar_timestamp` as the sole execution candidate. Earlier same-direction subjects remain recorded as valid decisions and do not create execution intents. This arbitration does not invalidate their evidence.
6. If an actionable decision set arrives while an XAUUSD position is active, reject all execution candidates from that set as `NO_TRADE_ACTIVE_POSITION`. Do not queue or defer them. No automatic reversal is permitted.

## 5. Position and Risk Rules

- Core v1.0 permits no more than one active XAUUSD position at any instant, regardless of subject count or direction.
- There is no automatic reversal. An opposing signal cannot close and reopen, flip, or otherwise reverse an open position without a separately specified exit and later entry lifecycle.
- Additional entry fills while the position is active are prohibited. All new execution candidates are rejected as `NO_TRADE_ACTIVE_POSITION`; they are not queued or deferred.
- The approved frozen per-trade risk baseline is **1%**. It MUST NOT be relaxed. Risk arbitration MUST occur before a fill can create a position.
- Risk base is account equity immediately before entry. `risk_budget = equity * 0.01`; the 1% value is a maximum and MUST NOT be increased.
- Required execution configuration MUST explicitly provide contract size, minimum quantity, quantity step, and the point-value/account-currency conversion information needed to calculate risk and quantity. No implicit broker defaults are allowed. If required metadata is missing or invalid, reject with `NO_TRADE_INVALID_EXECUTION_CONTRACT`.
- Quantity MUST comply with configured minimum quantity and quantity step, and the risk budget MUST NOT exceed 1% of pre-entry equity. Exact rounding behavior and the response when no valid quantity can satisfy both contract constraints and the risk ceiling are **UNSPECIFIED**.
- The repository's existing account-balance-based `OrderManager.compute_position_size` does not define this new historical execution sizing rule and MUST NOT supply implicit contract metadata.

## 6. Market-Only Support and Unsupported Cases

Core v1.0 supports market execution only. A strategy decision that requires a limit, stop-entry, or other unsupported order type MUST produce NO TRADE; the engine MUST NOT approximate it as a market order.

The engine MUST produce no executable trade when any of the following applies:

- The decision is not actionable under the existing strategy gate.
- The same-availability actionable decision set contains both directions.
- Risk/position arbitration rejects the intent.
- The order type is unsupported.
- Required execution contract metadata is missing or invalid (`NO_TRADE_INVALID_EXECUTION_CONTRACT`).
- An active XAUUSD position exists (`NO_TRADE_ACTIVE_POSITION`).
- Required direction, valid executable quote, or valid structural risk geometry is missing or invalid.
- No strictly later valid tick is available before the input ends.
- The quote is invalid, crossed, non-finite, non-positive, or otherwise rejected by canonical tick validation.

A rejected/unfilled decision MUST remain recorded with its subject and a reason. It MUST NOT create a filled order or active position.

An already-created market intent remains eligible for the first strictly-later valid canonical tick; Core v1.0 has no arbitrary time-based expiry. If the input ends before such a tick, record `UNFILLED_AT_END_OF_INPUT`. Do not synthesize ticks or fills across gaps, weekends, sessions, market closures, or EOF.

## 7. Fill-Side and Exit Rules

### Entry

- LONG market entry: fill from the eligible tick's **ASK**.
- SHORT market entry: fill from the eligible tick's **BID**.
- The entry tick MUST have `timestamp > decision_time`.
- Every tick sharing `decision_time` is ineligible, including later source-order records at that timestamp.
- No midpoint, bid/ask OHLC, OHLC open, declared strategy entry, or synthetic value may substitute for the selected tick-side quote.

### Exit

- LONG position exits are priced from **BID**.
- SHORT position exits are priced from **ASK**.
- Stop/target levels are trigger levels. Monitoring begins on the first quote event after the entry-fill event; the entry quote itself MUST NOT trigger an exit.
- Trigger comparisons are inclusive: LONG stop when `BID <= SL`; LONG target when `BID >= TP`; SHORT stop when `ASK >= SL`; SHORT target when `ASK <= TP`.
- When an exit trigger is reached, the historical market exit fill MUST use the actual exit-side quote on the triggering tick, not a fabricated fill at the declared stop/target level. This is the causal market-only interpretation of the approved quote-side and no-synthetic-fill rules.
- Stop and target levels MUST have valid directional geometry, so the inclusive stop and target predicates cannot both be true for the same single-side quote. Exit submission has no separately modeled latency in Core v1.0: the actual exit-side quote on the triggering tick is the fill quote.

## 8. Event Ordering

The engine MUST apply events in this order:

1. Read each canonical tick in preserved source order; reject invalid tick data according to canonical ingestion rules.
2. Apply the tick to any position that was already open before this tick. Evaluate inclusive stop/target predicates using BID for LONG exits and ASK for SHORT exits; if triggered, record the actual exit-side quote and close the position. Do not evaluate an entry on its own fill event.
3. Apply the tick to any market intent created by an earlier decision set. It is eligible only when `tick.timestamp > decision_time`; a permitted intent fills at the first strictly-later valid canonical tick, using ASK for LONG or BID for SHORT. Create the position only after recording the fill.
4. Feed the tick to the aggregator. Completed bars inherit the availability timestamp of the tick that caused emission.
5. Submit all bars sharing that availability as one atomic availability group.
6. Complete the group's replay update and collect all subjects/decisions at that availability timestamp. Evidence remains as-of that timestamp; per-subject base visibility remains bounded by its base-bar timestamp.
7. Close the decision set and perform direction-conflict arbitration before considering any intent created by that set.
8. For a same-direction set, select the latest base-bar subject as the sole candidate; preserve earlier subjects as valid decisions without intents. If an XAUUSD position is active after processing this tick's prior position event, reject all new candidates as `NO_TRADE_ACTIVE_POSITION`; do not queue them.
9. Apply risk and one-position arbitration; create at most one intent for the selected candidate and only for supported market execution. Required invalid or missing contract metadata produces `NO_TRADE_INVALID_EXECUTION_CONTRACT`.
10. An intent created from this tick's decision set cannot fill on this tick because its timestamp equals `decision_time`. It waits for a later valid canonical tick; there is no arbitrary time-based expiry. If input ends first, record `UNFILLED_AT_END_OF_INPUT`.
11. On subsequent quote events, evaluate the inclusive stop/target predicates using BID for LONG exits and ASK for SHORT exits. Record the actual exit-side quote used for the market exit.
12. Close the position only after the exit fill is recorded; calculate realized P&L from recorded entry/exit fills and explicit costs.

The latest `base_bar_timestamp` determines the sole candidate within a same-direction decision set; incidental callback order MUST NOT determine candidate selection. Earlier same-direction subjects remain recorded as valid decisions.

## 9. Required Data Structures

The following are logical records; exact language types and serialization format are implementation details, but the identity and causal fields are required.

| Record | Required fields |
|---|---|
| `QuoteEvent` | UTC tick timestamp, bid, ask, stable source order, source provenance/reference |
| `DecisionRecord` | `ReplaySubject`, decision time equal to subject availability, strategy state, direction if any, evidence reference, reason |
| `DecisionSet` | UTC availability/decision time, complete collection of independent decision records for that time, selected latest-base-bar candidate where applicable, arbitration status/reason; discarded same-direction decisions remain present and valid |
| `ExecutionConfig` | Explicit contract size, minimum quantity, quantity step, point-value/account-currency conversion information, commission configuration, explicit slippage configuration (default zero for deterministic baseline), and financing configuration only if enabled |
| `ExecutionIntent` | Collision-safe identity components (`execution_run_id`, originating `ReplaySubject.subject_id`, order discriminator), subject reference, decision time, direction, market order type, structural stop/target triggers, pre-entry equity and 1% risk budget, validated contract/sizing inputs, explicit cost configuration reference |
| `OrderRecord` | Order/intent ID, subject, status, eligible-after timestamp, selected order type, fill reference or NO TRADE/unfilled reason |
| `FillRecord` | Unique fill ID, collision-safe execution identity, order ID, subject, event timestamp, source order/provenance, side (`ASK`/`BID`), observed price, quantity, explicit commission/slippage/costs |
| `PositionRecord` | Unique position ID, originating subject/order, direction, entry fill, quantity, stop/target trigger levels, open/closed state, exit fill when closed |
| `ExecutionResult` | Subject, decision/order/position references as applicable, final status, entry/exit fill references, gross/net realized P&L, cost breakdown, terminal reason |

Execution record identities MUST use a collision-safe namespace containing `execution_run_id + ReplaySubject.subject_id + order discriminator`. This identity MUST be traceable through order, fill, position, exit, and result records. `ReplaySubject.subject_id` remains the decision identity and MUST NOT itself be repurposed as a trade ID. Decision identity, execution intent, order, position, and trade are distinct entities. The exact serialization delimiter is an implementation detail; implementations MUST preserve the components without collisions.

## 10. Costs and Result Accounting

- Bid/ask-side fills naturally include the observed spread. Spread MUST NOT be charged a second time as an additional cost.
- Commission MUST be explicit/configurable and recorded. Additional slippage MUST remain an explicit cost input; it may default to zero for the deterministic baseline, but that baseline MUST NOT be treated as final performance validation.
- Financing is not applied to Core v1.0 intraday positions unless explicitly configured. Configured financing, commission, and slippage values/models MUST be represented in the execution configuration and result cost breakdown.
- Required point-value/account-currency conversion information MUST be explicitly configured; there are no implicit broker defaults. Gross P&L is derived from actual recorded entry and exit fills, direction, quantity, and the configured conversion information. Net P&L subtracts the explicit recorded costs.
- Open positions and unfilled intents at end-of-input must not be marked to a synthetic price or reported as realized results. End-of-input reporting/valuation convention is **UNSPECIFIED**.

## 11. Compatibility Boundary

The new engine is additive and must sit downstream of the existing causal replay/strategy decision flow. It MUST NOT modify or change observable behavior of:

- `HistoricalReplay` batch decision timestamps, visibility, callback behavior, or observations;
- legacy batch replay and historical census behavior;
- existing OHLC-based `event_backtester.simulate_trade` and its declared-entry/ambiguous-candle semantics;
- current research exports except through a separately approved, additive integration.

Tick-driven execution results must not be presented as replacements for legacy OHLC research results without explicit labeling and provenance.

## 12. Acceptance Tests Required Before Integration

These are acceptance criteria for future implementation; this specification adds no tests.

1. **Canonical quote input:** malformed/crossed/non-finite/non-positive quotes fail closed; equal timestamps preserve source order; decreasing timestamps are rejected; provenance survives to fill records.
2. **Subject causality:** decision time equals `availability_timestamp`; `base_bar_timestamp` never affects fill eligibility; visible evidence cannot include unavailable/future data.
3. **Strictly later entry:** ticks before or equal to decision time never fill, including a later source-order tick with the same timestamp; a prior intent is matched against an incoming tick before that tick's newly generated decision set is arbitrated; the first eligible later quote fills a market LONG at ASK and SHORT at BID.
4. **No OHLC substitution:** no OHLC open/high/low/close, midpoint, declared strategy entry, or synthetic value can appear as a tick fill.
5. **Atomic decision set:** all same-availability subjects are collected before arbitration; opposing actionable directions reject the actionable group as NO TRADE regardless of callback/subject iteration order; non-actionable opposing decisions do not create a conflict.
6. **Independent subjects and same-direction selection:** every subject remains recorded as a decision; for an all-same-direction actionable set, only the latest base-bar subject becomes the execution candidate; earlier subjects remain valid decisions without intents; candidate selection is independent of callback order.
7. **One active position/no reversal:** an actionable candidate arriving while a position is open is rejected as `NO_TRADE_ACTIVE_POSITION`; no second position, queueing, deferral, or automatic reversal occurs.
8. **Exit causality/sides:** LONG stop is inclusive `BID <= SL`, LONG target is `BID >= TP`; SHORT stop is `ASK >= SL`, SHORT target is `ASK <= TP`; exit fill uses the triggering quote; the entry event cannot trigger an exit.
9. **Gap/weekend/EOF behavior:** a permitted intent waits for the first strictly-later valid canonical tick with no arbitrary time expiry; no synthetic ticks/fills are created; EOF before a qualifying tick records `UNFILLED_AT_END_OF_INPUT`.
10. **Risk base and contract metadata:** risk budget is immediately pre-entry account equity times 0.01; missing/invalid explicit contract size, minimum quantity, quantity step, or point-value/account-currency conversion information produces `NO_TRADE_INVALID_EXECUTION_CONTRACT`; no broker defaults are used.
11. **Costs:** observed spread is represented by quote-side fills and is not double-counted; commission is explicit/configurable; slippage is explicit and may default to zero only for the deterministic baseline, not final validation; financing is omitted unless configured.
12. **Identity integrity:** collision-safe execution identity contains `execution_run_id`, subject ID, and order discriminator and remains traceable through order, fill, position, exit, and result; the subject ID itself is not a trade ID.
13. **Legacy regression:** unchanged existing tests and outputs for `HistoricalReplay`, legacy batch replay, and OHLC research simulation continue to pass after future integration.
14. **End of data:** open positions and pending intents are not assigned fabricated fills or realized P&L; terminal reporting follows an explicitly approved convention.

## 13. Remaining Configuration and Implementation Details

The execution policies previously listed as approval blockers are resolved by this specification. The following are required runtime configuration or implementation details, not permission to substitute defaults or change the approved policy:

1. Each execution run must supply valid contract size, minimum quantity, quantity step, and point-value/account-currency conversion information. Missing or invalid required metadata produces `NO_TRADE_INVALID_EXECUTION_CONTRACT`.
2. Quantity rounding behavior, and the reason/status when no legal quantity can satisfy both the configured quantity constraints and the 1% risk ceiling, remain **UNSPECIFIED**. Implementation must not exceed the ceiling while this detail is unresolved.
3. Commission values/models must be explicitly configured. Slippage must be an explicit input, with zero permitted only for the deterministic baseline. Financing is disabled unless explicitly configured.
4. End-of-input reporting for open positions remains **UNSPECIFIED**; open positions must not be represented as realized or synthetically priced results.

No execution code, existing-module changes, tests, commits, or pushes are included in this specification-only change.
