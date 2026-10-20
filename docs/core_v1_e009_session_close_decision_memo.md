# E009 session-close lifecycle decision memo

**Date:** 2026-10-20  
**Status:** Owner-approved cross-session rejection handling implemented; maximum candidate lifetime remains unresolved for freeze.
**Scope:** Synthetic source/document audit only. No replay, backtest, strategy outcome, metric, chart, raw CSV row, or 2025 holdout material was accessed.

## Actual current behavior

The current development implementation separates **observation** from **entry eligibility**:

1. `derive_visible_fvg_interaction()` records the first post-MSS FVG overlap. It does not expire that interaction at session close.
2. `FVGQuarterRejectionEntryResolver` continues scanning visible completed base bars from the initial interaction forward. It keeps inside-FVG closes valid, invalidates only on the specified adverse-quarter close, and returns the first later close beyond the far edge as the planned-entry event.
3. `evaluate_core_v1_pipeline()` applies the session gate only after a planned entry has been produced. The planned-entry event must fall inside `America/New_York` `[01:00,05:00)` or `[06:00,10:00)`; otherwise the retracement is changed to `DEVELOPING`, the planned entry is removed, and invalidation/target/hypothesis construction is not reached for that subject.
4. The owner-approved resolver behavior now skips a qualifying rejection outside both windows as an entry trigger without consuming the candidate, continues evaluating later visible bars, and returns the first later qualifying rejection whose event is inside an eligible window. All evaluated sources, including the skipped outside-window rejection, remain in planned-entry provenance.
5. There is no session-close expiry, window-count limit, or trading-time lifetime. Candidate lifetime remains unresolved for freeze. The canonical pipeline session gate remains defense in depth before invalidation/target/hypothesis construction.

This is the actual source behavior. The documented intent in `docs/core_v1_retracement_contract.md` explicitly says that no FVG lifetime or session-close expiry is inferred and that the lifetime remains unresolved for a frozen evaluation rule.

## Owner choices

### A. Expire at the current entry-window close

- **Semantics:** A touched but not yet rejected E009 candidate becomes ineligible when the current London/New York window closes.
- **Rejection outside the window:** It cannot qualify. The candidate is permanently expired for that interaction.
- **Later window:** No later rejection can qualify from the expired interaction; a new post-MSS interaction would be required if the policy permits one.
- **Tradeoff:** Tight causal/session containment and simpler accounting, but discards setups that remain structurally valid across a normal session boundary.

### B. Continue observing into later windows with outside-window rejection skipped **(owner approved)**

- **Semantics:** Continue evaluating across session boundaries; skip a qualifying rejection outside both windows as an entry trigger, preserve its source, and allow the first later qualifying in-window rejection while the candidate remains valid.
- **Rejection outside the window:** It does not create an entry and does not consume the candidate.
- **Later window:** The first later qualifying rejection inside an eligible window may qualify.
- **Tradeoff:** Preserves cross-session continuity but leaves candidate staleness/lifetime unresolved; that lifetime remains a freeze blocker.

### C. Bounded windows or trading-time lifetime

- **Semantics:** Continue observation for a specified number of later eligible windows or a specified trading-time duration, then expire deterministically. The owner-approved outside-window skip behavior would remain, subject to the added bound.
- **Rejection outside the window:** It does not qualify at that event. Whether it consumes the candidate or merely fails that event must be specified together with the bound.
- **Later window:** A later qualifying rejection may qualify only while the candidate remains within the selected bound, the resolver reaches it, and the rejection event is inside an allowed window.
- **Tradeoff:** Controls staleness while preserving limited cross-session continuity, but adds owner-selected parameters and requires a verified trading calendar if the bound uses trading time.

## Decision required

Option B is owner-approved for cross-session rejection handling. A separate owner decision is still required for any maximum window count, trading-time lifetime, or expiry policy. Until then:

- E009 remains **EXPERIMENTAL** and **NOT PERFORMANCE EVALUATED**.
- The model remains **not freeze-ready**.
- No expiry or candidate lifetime is selected.
- No holdout run is authorized.
