# E009 session-close lifecycle decision memo

**Date:** 2026-10-20  
**Status:** Awaiting owner decision; no expiry behavior was selected or implemented.  
**Scope:** Synthetic source/document audit only. No replay, backtest, strategy outcome, metric, chart, raw CSV row, or 2025 holdout material was accessed.

## Actual current behavior

The current development implementation separates **observation** from **entry eligibility**:

1. `derive_visible_fvg_interaction()` records the first post-MSS FVG overlap. It does not expire that interaction at session close.
2. `FVGQuarterRejectionEntryResolver` continues scanning visible completed base bars from the initial interaction forward. It keeps inside-FVG closes valid, invalidates only on the specified adverse-quarter close, and returns the first later close beyond the far edge as the planned-entry event.
3. `evaluate_core_v1_pipeline()` applies the session gate only after a planned entry has been produced. The planned-entry event must fall inside `America/New_York` `[01:00,05:00)` or `[06:00,10:00)`; otherwise the retracement is changed to `DEVELOPING`, the planned entry is removed, and invalidation/target/hypothesis construction is not reached for that subject.
4. A later replay subject can see the same still-valid FVG interaction plus additional bars. However, the resolver returns the **first qualifying rejection in visible chronological order**. If that first rejection occurred outside a session window, the canonical pipeline blocks it, and the same earlier rejection remains visible and shadows later rejection bars. A later in-window rejection is therefore **not** reached in full-history replay.
5. There is currently no session-close expiry, window-count limit, trading-time lifetime, or explicit “rejection outside window permanently disqualifies” rule in the E009 resolver or canonical pipeline. In bounded-history replay, bars may eventually fall out of view and behavior may differ, but that is an incidental visibility effect, not a selected lifecycle rule.

This is the actual source behavior. The documented intent in `docs/core_v1_retracement_contract.md` explicitly says that no FVG lifetime or session-close expiry is inferred and that the lifetime remains unresolved for a frozen evaluation rule.

## Owner choices

### A. Expire at the current entry-window close

- **Semantics:** A touched but not yet rejected E009 candidate becomes ineligible when the current London/New York window closes.
- **Rejection outside the window:** It cannot qualify. The candidate is permanently expired for that interaction.
- **Later window:** No later rejection can qualify from the expired interaction; a new post-MSS interaction would be required if the policy permits one.
- **Tradeoff:** Tight causal/session containment and simpler accounting, but discards setups that remain structurally valid across a normal session boundary.

### B. Continue observing into later windows, with explicit first-rejection handling

- **Semantics:** Continue evaluating across session boundaries until E009 invalidation or another explicitly selected lifetime rule, **but change the resolver** so an outside-window first rejection is skipped or otherwise handled rather than shadowing later bars.
- **Rejection outside the window:** It does not create a planned entry. The owner must specify whether it is recorded as a failed event, consumed, or skipped for later evaluation.
- **Later window:** A later qualifying rejection may qualify only if the resolver is changed to reach it and its planned-entry event is inside a later allowed window.
- **Tradeoff:** Preserves more candidate continuity, but changes current resolver semantics and requires explicit handling of the earlier outside-window rejection plus a separate lifetime/aging rule before freezing.

### C. Bounded windows or trading-time lifetime

- **Semantics:** Continue observation for a specified number of later eligible windows or a specified trading-time duration, then expire deterministically. If later rejection after an outside-window first rejection is desired, the resolver must also be changed to skip or otherwise handle that earlier event.
- **Rejection outside the window:** It does not qualify at that event. Whether it consumes the candidate or merely fails that event must be specified together with the bound.
- **Later window:** A later qualifying rejection may qualify only while the candidate remains within the selected bound, the resolver reaches it, and the rejection event is inside an allowed window.
- **Tradeoff:** Controls staleness while preserving limited cross-session continuity, but adds owner-selected parameters and requires a verified trading calendar if the bound uses trading time.

## Decision required

The owner must select **A, B, or C** and, for B/C, explicitly approve any resolver change needed to skip or handle an earlier outside-window rejection. For C, provide the exact window-count or trading-time lifetime plus the treatment of an outside-window rejection. Until then:

- E009 remains **EXPERIMENTAL** and **NOT PERFORMANCE EVALUATED**.
- The model remains **not freeze-ready**.
- No expiry, candidate consumption, or cross-session rule is changed.
- No holdout run is authorized.
