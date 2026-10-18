# Core v1 visible MSS implementation contract

**Status:** Development implementation rule; not yet a frozen evaluation build.  
**Date:** 2026-10-18  
**Scope:** Automatic MSS derivation from `SubjectVisibleInputs` in the canonical Core v1 pipeline. This contract defines detection behavior only; it reports no strategy performance and does not authorize a replay evaluation.

## Preconditions

- The selected visible liquidity assessment has a confirmed rejection classified as `SWEEP`. Acceptance, an unresolved/not-sweep classification, conflicting direction, or missing evidence fails closed.
- A bullish displacement requires a sell-side liquidity rejection; a bearish displacement requires a buy-side liquidity rejection.
- The displacement event must occur strictly after the rejection confirmation and be confirmed within the subject cutoff by the existing displacement stage.

## Structural point

- Bullish MSS must break a swing high; bearish MSS must break a swing low.
- The pivot uses the existing strict swing detector with two visible bars on each side. Equal highs/lows do not qualify.
- Select the most recent matching pivot whose event and confirmation timestamps are both strictly before the liquidity breach and whose entire pivot-window source set was actually available before the breach. A point that was not knowable before the breach is unavailable as the pre-existing structural reference.

## Break and follow-through

- The break is the first visible base bar after the displacement event whose close is strictly beyond the selected structural point in the displacement direction. A wick alone does not qualify.
- Follow-through must occur within the next two visible base bars. A confirming bar must close beyond the break bar's high for bullish MSS or low for bearish MSS.
- A close back through the structural point before confirmation invalidates that break candidate. A later qualifying break can still be evaluated from visible evidence.
- MSS is unresolved until the confirmation bar is visible and available by the ReplaySubject cutoff. Later bars cannot revise an earlier subject's result.

## Evidence and limits

The structural point retains its pivot window; the break retains the point and break-bar sources; follow-through retains those sources plus each intervening bar. Event, confirmation, availability, subject, and source lineage are validated by the existing evidence model. Missing, malformed, conflicting, or ambiguous OHLC evidence fails closed.

This is a deterministic development policy selected to make the model's mandatory structure stage executable and reviewable. It is not a claim that the rule has predictive value. Do not inspect the sealed 2025 outcomes, optimize this rule on outcomes, or evaluate performance until the full strategy and runtime freeze gates pass and the user explicitly authorizes the single holdout run.
