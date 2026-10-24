# Core v1 visible FVG retracement evidence contract

**Status:** Development implementation rule; not a frozen evaluation build.  
**Date:** 2026-10-20
**Scope:** Detect a post-MSS interaction with a resolved FVG candidate and apply an experimental entry trigger. This policy is not performance-tested or frozen for evaluation.

## Directional context and liquidity draw gate

- A planned entry requires a resolved dealing-range context with visible, point-in-time anchor provenance. The range-anchor policy remains caller-governed; the default unavailable resolver does not infer a range.
- A planned entry also requires an explicit governed draw-on-liquidity level. The default pipeline does not guess which visible level is the directional objective.
- Both context and draw evidence must be available strictly before the liquidity-breach event; later evidence cannot retroactively define the setup's original context. The draw must recompute from visible swings.
- The draw side must agree with displacement direction: bullish displacement requires buy-side liquidity (BSL), and bearish displacement requires sell-side liquidity (SSL). Missing, unresolved, late, or conflicting evidence returns `DEVELOPING` before the entry resolver is called.
- Validate context and draw provenance against the same `ReplaySubject`, enforce their availability cutoffs, and preserve both in planned-entry provenance. A `VALID` retracement result cannot omit them.
- Premium/discount location alone does not establish the directional thesis. No range anchor, liquidity objective, or session policy is inferred from that classification.

## Interaction rule

- Consider base-timeframe bars strictly after both the FVG's availability timestamp and MSS confirmation event. The formation, confirmation, and pre-MSS bars cannot count as a retracement.
- Validate each considered bar's source timing, OHLC geometry, and finite values. Malformed or non-monotonic evidence fails closed.
- The first bar whose low is at or below the FVG upper bound and whose high is at or above the FVG lower bound records the initial interaction. This is not a first-touch-only entry decision: the FVG remains under observation while valid.
- Use the bar timestamp as the interaction event and retain the complete FVG source lineage plus the interaction bar source.
- OHLC does not reveal the intrabar path or exact touch price. Record no exact interaction price and do not treat overlap as rejection, entry, or fill.
- A valid planned entry requires both a zone and an interaction strictly after MSS confirmation. Caller-supplied resolvers cannot bypass this sequence.

## E008 experimental first-touch rejection trigger (retained)

- Use only a displacement-aligned FVG whose middle candle is the confirmed displacement event.
- Evaluate the first visible bar that overlaps the FVG strictly after MSS confirmation. If its close does not finish strictly beyond the FVG's far edge in the displacement direction, this resolver returns no planned entry for that first touch; later touches do not replace it.
- For a bullish FVG, the close must be above the upper bound. For a bearish FVG, it must be below the lower bound.
- The planned entry reference is the trigger bar close. It is a plan price, not an exact fill. The decision is available only with its visible evidence; tick execution remains strictly later than the decision availability.
- This operational trigger is an experimental strategy choice because the source documents require an entry condition but do not select this particular trigger. It is recorded for future review and must not be treated as an active, performance-supported rule.

E008 is retained as a historical, explicitly selectable implementation. It is no longer the canonical pipeline's default development hypothesis.

## E009 owner-specified FVG DRT-quarter rejection hypothesis

- Use a displacement-aligned FVG whose middle candle is the confirmed displacement event.
- Normalize the FVG from lower edge `0` to upper edge `1`, with DRT boundaries at `0.25`, `0.50`, and `0.75`. This DRT partition applies to the FVG itself.
- After the first post-MSS overlap, continue evaluating completed visible base-timeframe candle closes. A close inside the FVG does not end the setup; keep waiting while the adverse quarter boundary has not been crossed.
- For bullish displacement, a close strictly below the FVG's `0.25` level invalidates the candidate. A close exactly at `0.25` remains valid. For bearish displacement, a close strictly above `0.75` invalidates it; equality remains valid. Wicks through these boundaries do not invalidate the candidate.
- A bullish rejection candidate occurs only when a later close is strictly above the FVG upper edge (`1`); a bearish candidate requires a later close strictly below the lower edge (`0`). The first qualifying rejection close supplies the planned reference price. This is not a fill.
- Retain the FVG, initial interaction, and every evaluated candle source through the rejection candle. The first eligible close determines the decision; later candles cannot change that historical result. Malformed or conflicting evidence before the trigger fails closed.
- OTE coordinates are contextual only and are not a required entry condition. If used, OTE belongs to a larger dealing range; the FVG uses its own DRT quartiles. No OTE gate is added to this FVG resolver.
- Entry-window gate: the planned-entry event must occur in one of the owner-confirmed, half-open `America/New_York` windows: London ICT killzone plus the preceding hour `[01:00,05:00)`, or New York ICT killzone plus the preceding hour `[06:00,10:00)`. The IANA timezone conversion makes these windows DST-aware. A missing timezone database or an event outside both windows fails closed before invalidation/target/hypothesis construction.
- The session gate applies only to a new planned-entry event. A position already opened continues under its normal TP/SL after the window ends; the window does not force-close it.
- Owner-approved cross-session behavior: a touched E009 candidate remains active across session closes; a qualifying close outside both windows is skipped as an entry trigger and does not consume the candidate or an eligible window; later visible bars may produce the first qualifying in-window rejection. Out-of-window rejection sources remain in provenance when a later entry qualifies.
- Approved development lifetime: count the eligible window containing the initial visible FVG interaction as window one. If the interaction is outside both windows, begin with the next eligible window. The candidate remains eligible through three counted windows, and expires at the exclusive end of the third counted window if still unfilled. A rejection event exactly at or after that boundary cannot qualify. The resolver derives this bound from visible event timestamps and the DST-aware session policy; it does not use future bars or hidden mutable state.
- The existing adverse-quarter invalidation still ends the candidate immediately. Expiry applies only to an unfilled candidate; an already-open position continues under normal TP/SL management.

E009 is the canonical pipeline's development default. The three-window lifetime is an approved development rule and is covered only by synthetic contract tests. E009 remains **EXPERIMENTAL** and **NOT PERFORMANCE EVALUATED**; no replay, backtest, or holdout is authorized by this contract.

Neither the interaction detector nor the experimental trigger authorizes replay evaluation or access to sealed 2025 outcomes. The new directional-context and draw gate is a development contract change, not a frozen evaluation build.
