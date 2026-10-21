# Core v1 freeze-readiness record

**Audit date:** 2026-10-21
**Branch:** `wip/step3d-recovered`  
**Starting HEAD:** `f479f0d22b10e1394b9f865271415bbe439a8537`  
**Latest development updates:** E009 implementation `9e7c5c140b91833017e5a26e43ba28c08831427a`; session trade-management record `44ae64366ca61562b65e6f516772662cb0de67c4`; DST-aware session gate `8d66de9`; focused session verification and the canonical cross-session pipeline regression were added in the current review.
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
| E009 maximum lifetime/expiry decision | Requires owner strategy decision | The owner approved cross-session handling: skip outside-window rejection triggers without consuming the candidate and allow the first later in-window rejection. Still select whether any maximum window count, trading-time lifetime, or expiry exists. | Implemented and focused-tested; no lifetime bound or expiry is selected, so this remains a freeze blocker. |
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
python -m pytest tests/test_core_v1_sessions.py tests/test_core_v1_pipeline.py tests/test_core_v1_tick_execution.py tests/test_core_v1_displacement.py tests/test_core_v1_hypothesis.py tests/test_core_v1_decision_bridge.py tests/test_tick_execution.py tests/test_tick_execution_stateful.py tests/test_tick_execution_replay.py
```

Result: **292 passed in 8.68s**. No full-suite run was performed. `git diff --check` passed.

Canonical cross-session integration verification:

```text
python -m pytest tests/test_core_v1_cross_session_pipeline.py tests/test_core_v1_retracement.py tests/test_core_v1_pipeline.py tests/test_core_v1_sessions.py
```

Result: **133 passed in 2.47s**. The integration fixture uses synthetic invalidation and target evidence only to isolate the cross-session lifecycle; no full-suite run was performed.

E009 session-close lifecycle audit verification:

```text
python -m pytest tests/test_core_v1_retracement.py tests/test_core_v1_pipeline.py tests/test_core_v1_sessions.py
```

Result: **132 passed in 4.30s**. The E009 characterization uses the actual resolver and session policy across successive cutoffs: an out-of-window rejection is skipped, a later in-window rejection qualifies with the skipped source retained, adverse-quarter invalidation still wins, and future bars remain unavailable before their cutoff. The session-boundary test is named for its actual outside-window behavior; the six-minutes-after-endpoint fixture is no longer described as an exact endpoint case.

## Next safe roadmap work

The configuration audit confirms that `configs/core_v1_development.json` records the owner-approved cross-session policy, the half-open DST-aware windows, normal TP/SL management after entry, and an explicitly unresolved candidate lifetime. No maximum window count, trading-time lifetime, or expiry rule is inferred by this review.

The next safe items remain freeze-readiness work rather than performance evaluation:

1. **Owner decision:** select or continue to defer the E009 candidate maximum lifetime/expiry policy. Implementation must wait for that decision; no bound is assumed here.
2. **Owner decisions:** resolve breached-but-unrejected liquidity consumption and the deterministic dealing-range anchor policy.
3. **Verified-runtime implementation:** confirm the frozen runtime's `America/New_York` IANA timezone data and, only after a verified XAUUSD calendar is supplied, implement the already-recorded trading-time liquidity lifetime semantics.
4. **Freeze bookkeeping:** complete verified account/execution inputs and the clean configuration/build/run record after the strategy decisions above are settled.

None of these items authorizes replay, backtest, performance evaluation, or holdout access.

## Readiness conclusion

The Core v1 implementation is **not freeze-ready**. Independent engineering verification for displacement, lifecycle/execution integration, and the owner-confirmed session gate is complete. The owner-approved E009 cross-session behavior is implemented and focused-tested: outside-window rejection triggers are skipped without consuming the candidate, later in-window rejection may qualify, and skipped sources are retained in provenance. No maximum window count, trading-time lifetime, or expiry policy is selected, so candidate lifetime remains a freeze blocker. E009 remains experimental and not performance evaluated. The owner-confirmed DST-aware entry windows are implemented and focused-tested for development; the frozen runtime must include working `America/New_York` IANA timezone data. Other owner decisions, verified account/calendar inputs, and the pre-run freeze record remain outstanding. The owner specified that an open trade continues under its normal TP/SL after session close. No holdout run is authorized by this record.
