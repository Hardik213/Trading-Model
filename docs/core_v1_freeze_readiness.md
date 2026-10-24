# Core v1 freeze-readiness record

**Audit date:** 2026-10-21
**Branch:** `recovery/e009-three-window`  
**Starting HEAD:** `ad8c5fa3328c337df33b03a91eaac8f1a802c004`  
**Latest development updates:** recovered E009/session work, DST-aware session gate, owner-approved cross-session handling, and the approved three-window candidate lifetime are implemented in this review branch.
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
| E009 maximum lifetime/expiry decision | Development rule implemented; independent freeze clearance remains required | Approved three eligible entry windows, counting the interaction-containing window or the next eligible window when interaction is outside; exclusive expiry at the third window end; no entry at or after expiry; adverse-quarter invalidation remains immediate; open positions continue normal TP/SL. | Implemented and focused-tested synthetically; E009 remains experimental and not performance evaluated. |
| Trading-time liquidity lifetime | Implementable by agent now, but requires verified external information | Implement the frozen nominal lifetime table and trading-time aging only after the verified XAUUSD calendar is supplied/confirmed; no guessed calendar or broker hours may be inserted. | Not completed because the calendar is missing. |
| Breached-but-unrejected liquidity consumption | Requires owner strategy decision | Specify whether a breach without rejection consumes the level immediately, or permits a later rejection of the same breach. Current config remains `UNRESOLVED`. | Not selected; no behavior invented. |
| Dealing-range anchor policy | Requires owner strategy decision | Select a deterministic anchor policy and its causal availability rules. Current policy is caller-governed/unresolved for freeze. | Existing implementation remains caller-governed. |
| Daily/H4 timezone boundary | Requires verified external/account or calendar information | Verify the TradingView OANDA `America/New_York` 17:00 boundary with timezone-database/DST handling and record the verification source/version. | Not verified; no fixed UTC substitute used. |
| Strategy session filter | Implemented for development; frozen-runtime verification remains | Owner selected ICT London and New York killzones with one preceding hour: `[01:00,05:00)` and `[06:00,10:00)` in `America/New_York`. Entry eligibility uses the planned-entry event timestamp. | Implemented in `src/core_v1_sessions.py` and enforced before downstream hypothesis gates. Focused tests cover standard/DST windows, half-open boundaries, outside-window rejection, repeated/skipped DST local times, invalid/naive timestamps, unavailable timezone data, canonical pipeline blocking, and post-window TP/SL management. Verify IANA timezone data is installed in the frozen runtime. |
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

Focused tests cover unresolved-stage blocking, subject/timing/provenance lineage, planned-entry versus fill separation, strict-later execution, arbitration, active-position rejection, actual fill sides, costs, risk sizing, EOF behavior, replay availability boundaries, and the owner-confirmed DST-aware session gate. Existing synthetic bridge/execution fixtures are explicitly isolated from the session clock so their pre-session lifecycle contracts remain focused; session-specific pipeline tests exercise the real gate separately. These are implementation checks only; they are not strategy-performance results.

## Current focused verification

```text
python -m pytest -q tests/test_core_v1_sessions.py tests/test_core_v1_e009_lifetime.py tests/test_core_v1_cross_session_pipeline.py tests/test_core_v1_retracement.py tests/test_core_v1_pipeline.py tests/test_core_v1_tick_execution.py
```

Result: **131 passed in 3.09s**. No full-suite run was performed. `git diff --check` passed.

Canonical cross-session integration verification:

```text
python -m pytest tests/test_core_v1_cross_session_pipeline.py tests/test_core_v1_retracement.py tests/test_core_v1_pipeline.py tests/test_core_v1_sessions.py
```

Result: Covered by the focused command above. The integration fixture uses synthetic invalidation and target evidence only to isolate the cross-session lifecycle; no full-suite run was performed.

E009 session-close lifecycle audit verification:

```text
python -m pytest tests/test_core_v1_retracement.py tests/test_core_v1_pipeline.py tests/test_core_v1_sessions.py
```

Result: Superseded by the focused three-window verification above. The E009 characterization uses the actual resolver and session policy across successive cutoffs: an out-of-window rejection is skipped, a later in-window rejection qualifies with the skipped source retained, adverse-quarter invalidation still wins, the third-window rejection qualifies before its exclusive end, and the boundary event fails closed.

## Next safe roadmap work

The configuration audit confirms that `configs/core_v1_development.json` records the owner-approved cross-session policy, the half-open DST-aware windows, the three-window candidate lifetime, exclusive expiry, normal TP/SL management after entry, and the development-only status. No trading-time lifetime is inferred.

The next safe items remain freeze-readiness work rather than performance evaluation:

1. **Independent freeze clearance:** review the approved three-window E009 lifetime and its synthetic evidence; no performance claim follows from implementation tests.
2. **Owner decisions:** resolve breached-but-unrejected liquidity consumption and the deterministic dealing-range anchor policy.
3. **Verified runtime:** confirm the frozen runtime's `America/New_York` IANA timezone data before any evaluation; the implemented E009 bound is session-window based, not a trading-time calendar lifetime.
4. **Freeze bookkeeping:** complete verified account/execution inputs and the clean configuration/build/run record after the strategy decisions above are settled.

None of these items authorizes replay, backtest, performance evaluation, or holdout access.

## Readiness conclusion

The Core v1 implementation is **not freeze-ready**. Independent engineering verification for displacement, lifecycle/execution integration, and the owner-confirmed session gate is complete. The owner-approved E009 cross-session behavior and three-window lifetime are implemented and focused-tested: outside-window rejection triggers are skipped without consuming the candidate or a window, later in-window rejection may qualify within the third counted window, expiry is exclusive and fail-closed, and skipped sources are retained in provenance. E009 remains experimental and not performance evaluated. The owner-confirmed DST-aware entry windows are implemented and focused-tested for development; the frozen runtime must include working `America/New_York` IANA timezone data. Other owner decisions, verified account/execution inputs, and the pre-run freeze record remain outstanding. The owner specified that an open trade continues under its normal TP/SL after session close. No holdout run is authorized by this record.
