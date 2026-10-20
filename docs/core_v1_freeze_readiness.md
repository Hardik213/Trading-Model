# Core v1 freeze-readiness record

**Audit date:** 2026-10-20  
**Branch:** `wip/step3d-recovered`  
**Starting HEAD:** `f479f0d22b10e1394b9f865271415bbe439a8537`  
**Latest development updates:** E009 implementation `9e7c5c140b91833017e5a26e43ba28c08831427a`; session trade-management record `44ae64366ca61562b65e6f516772662cb0de67c4`; owner-confirmed session windows implemented in the current development worktree (not yet committed).
**Scope:** Engineering readiness review only. No holdout run is authorized. No raw CSV rows, strategy outcomes, performance metrics, charts, replay, backtest, or strategy evaluation was accessed.

## Stage status

| Stage | Implementation present | Verified against contract | Owner development acceptance | Frozen evaluation build | Performance evidence |
| --- | --- | --- | --- | --- | --- |
| Replay/evidence | Yes | Focused repository tests and integration tests | Not a strategy-rule acceptance | No | None |
| Liquidity | Yes | Focused synthetic tests and canonical pipeline tests | Development implementation only | No | None |
| Displacement | Yes | `docs/core_v1_displacement_audit.md`; focused synthetic tests | Technical verification only; no profitability claim | No | None |
| MSS | Yes | `docs/core_v1_mss_audit.md`; focused synthetic tests | **IMPLEMENTATION ACCEPTED FOR DEVELOPMENT PROGRESSION** | No | None |
| FVG/retracement | Yes | `docs/core_v1_retracement_audit.md`; focused synthetic tests | Development implementation only | No | None |
| E009 FVG DRT-quarter entry hypothesis | Yes | Updated contract and focused synthetic tests | **Owner-specified development hypothesis; remains EXPERIMENTAL** | No | **NOT PERFORMANCE EVALUATED** |
| Hypothesis/invalidation/target | Yes | `docs/core_v1_hypothesis_contract.md`; focused synthetic tests | Development implementation only | No | None |
| Decision bridge/tick execution | Yes | Focused bridge, execution, stateful, replay, and Core v1 integration tests | Execution contract verification only | No | None |

## Freeze blockers and categories

| Blocker | Category | Exact missing action/value | Agent work status |
| --- | --- | --- | --- |
| E009 lifecycle/freeze decision | Requires owner strategy decision | The FVG quarter and close-based rejection conditions are specified for development. The candidate's lifetime across session close remains undecided. | E009 is now the development default; it is not frozen or performance evaluated. |
| Trading-time liquidity lifetime | Implementable by agent now, but requires verified external information | Implement the frozen nominal lifetime table and trading-time aging only after the verified XAUUSD calendar is supplied/confirmed; no guessed calendar or broker hours may be inserted. | Not completed because the calendar is missing. |
| Breached-but-unrejected liquidity consumption | Requires owner strategy decision | Specify whether a breach without rejection consumes the level immediately, or permits a later rejection of the same breach. Current config remains `UNRESOLVED`. | Not selected; no behavior invented. |
| Dealing-range anchor policy | Requires owner strategy decision | Select a deterministic anchor policy and its causal availability rules. Current policy is caller-governed/unresolved for freeze. | Existing implementation remains caller-governed. |
| Daily/H4 timezone boundary | Requires verified external/account or calendar information | Verify the TradingView OANDA `America/New_York` 17:00 boundary with timezone-database/DST handling and record the verification source/version. | Not verified; no fixed UTC substitute used. |
| Strategy session filter | Implemented for development; frozen-runtime verification remains | Owner selected ICT London and New York killzones with one preceding hour: `[01:00,05:00)` and `[06:00,10:00)` in `America/New_York`. Entry eligibility uses the planned-entry event timestamp. | Implemented in `src/core_v1_sessions.py` and enforced before downstream hypothesis gates. New entries are blocked outside the windows; an open position continues under normal TP/SL. Verify IANA timezone data is installed in the frozen runtime. |
| Account currency | Requires verified external/account information | Exact account currency code used by execution/P&L conversion. | Missing in development config; not guessed. |
| Starting equity | Requires verified external/account information | Positive numeric starting equity for the execution run. | Missing in development config; not guessed. |
| Contract size/value | Requires verified external/account information | Explicit XAUUSD contract size and point-value/account-currency interpretation. | Missing in development config; not guessed. |
| Quantity limits and step | Requires verified external/account information | Minimum quantity and quantity step accepted by the execution contract. | Missing in development config; not guessed. |
| Conversion policy | Requires verified external/account information | Account-currency conversion source/rate policy, including runtime provenance. | Missing in development config; not guessed. |
| Commission | Requires verified external/account information | Explicit per-unit-per-side commission or documented zero diagnostic baseline. | Missing in development config; not guessed. |
| Slippage | Requires verified external/account information | Explicit per-unit-per-side slippage or documented zero diagnostic baseline. | Missing in development config; not guessed. |
| Frozen configuration/build record | Implementable by agent now after owner/config decisions | Record clean frozen commit, canonical configuration SHA-256, dataset/version and manifest hash, execution assumptions, metrics/bootstrap, command, environment, and unique run/variant ID. | Blocked until configuration and owner decisions are resolved; no frozen record claimed. |
| Holdout authorization | Requires owner authorization | Separate explicit authorization after all freeze gates pass. | Not given; no holdout run authorized. |

## Lifecycle and execution integration evidence

The canonical chain is guarded as follows:

- `src/core_v1_pipeline.py` stops at the first unresolved prerequisite and does not advance later gates as a valid hypothesis.
- `src/core_v1_hypothesis.py` requires explicit, visible, subject-linked invalidation and opposing-liquidity target evidence; it does not add minimum-R, position-sizing, or execution policy.
- `src/core_v1_decision_bridge.py` preserves planned entry as strategy context, rejects non-actionable pipeline states, requires explicit quantity for an actionable decision, and never instantiates/calls the execution engine.
- `src/core_v1_tick_execution.py` uses the canonical pipeline by default; custom builders require explicit test-only opt-in and a stable identity. Visible evidence is built from replay subjects and canonical tick provenance.
- `src/tick_execution.py` enforces strict-later quote entry, bid/ask fill sides, one active position, same-availability arbitration, risk-derived quantity, explicit costs, actual quote exits, unfilled-at-EOF, and open-at-EOF mark-to-market behavior.

Focused tests cover unresolved-stage blocking, subject/timing/provenance lineage, planned-entry versus fill separation, strict-later execution, arbitration, active-position rejection, actual fill sides, costs, risk sizing, EOF behavior, and replay availability boundaries. These are implementation checks only; they are not strategy-performance results.

## Readiness conclusion

The Core v1 implementation is **not freeze-ready**. Independent engineering verification for displacement and lifecycle/execution integration is complete, with no demonstrated production defect requiring a rule change. The owner has specified the E009 development hypothesis: FVG DRT quartiles, continued observation through inside closes, close-only adverse-quarter invalidation, and a close beyond the far edge for rejection. E009 remains experimental and not performance evaluated. The owner-confirmed DST-aware entry windows are implemented for development; the frozen runtime must include working `America/New_York` IANA timezone data. FVG lifetime/session expiry remains unresolved, along with other owner decisions, verified account/calendar inputs, and the pre-run freeze record. The owner specified that an open trade continues under its normal TP/SL after session close. No holdout run is authorized by this record.
