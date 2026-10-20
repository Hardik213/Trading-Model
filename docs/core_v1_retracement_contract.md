# Core v1 visible FVG retracement evidence contract

**Status:** Development implementation rule; not a frozen evaluation build.  
**Date:** 2026-10-20
**Scope:** Detect a post-MSS interaction with a resolved FVG candidate and apply an experimental entry trigger. This policy is not performance-tested or frozen for evaluation.

## Interaction rule

- Consider base-timeframe bars strictly after both the FVG's availability timestamp and MSS confirmation event. The formation, confirmation, and pre-MSS bars cannot count as a retracement.
- Validate each considered bar's source timing, OHLC geometry, and finite values. Malformed or non-monotonic evidence fails closed.
- The first bar whose low is at or below the FVG upper bound and whose high is at or above the FVG lower bound records an interaction.
- Use the bar timestamp as the interaction event and retain the complete FVG source lineage plus the interaction bar source.
- OHLC does not reveal the intrabar path or exact touch price. Record no exact interaction price and do not treat overlap as rejection, entry, or fill.
- A valid planned entry requires both a zone and an interaction strictly after MSS confirmation. Caller-supplied resolvers cannot bypass this sequence.

## Experimental first-touch rejection trigger

- Use only a displacement-aligned FVG whose middle candle is the confirmed displacement event.
- Evaluate the first visible bar that overlaps the FVG strictly after MSS confirmation. If its close does not finish strictly beyond the FVG's far edge in the displacement direction, this resolver returns no planned entry for that first touch; later touches do not replace it.
- For a bullish FVG, the close must be above the upper bound. For a bearish FVG, it must be below the lower bound.
- The planned entry reference is the trigger bar close. It is a plan price, not an exact fill. The decision is available only with its visible evidence; tick execution remains strictly later than the decision availability.
- This operational trigger is an experimental strategy choice because the source documents require an entry condition but do not select this particular trigger. It is recorded for future review and must not be treated as an active, performance-supported rule.

Neither the interaction detector nor the experimental trigger authorizes replay evaluation or access to sealed 2025 outcomes.
