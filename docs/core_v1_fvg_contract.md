# Core v1 visible FVG candidate contract

**Status:** Development implementation rule; not a frozen evaluation build.  
**Date:** 2026-10-18  
**Scope:** Automatic observation of a displacement-linked FVG after confirmed MSS. This contract defines a candidate zone only. It does not define a retracement trigger, entry, invalidation, target, or trade.

## Candidate rule

- Use only visible, ordered base-timeframe OHLC observations with one source observation per bar and valid finite OHLC geometry.
- Apply the existing three-candle FVG detector: bullish when the third candle's low is above the first candle's high; bearish when the third candle's high is below the first candle's low.
- The middle candle must be the confirmed displacement event, the FVG direction must agree with displacement, and the third candle must be confirmed by the ReplaySubject base-bar cutoff.
- Return a zone only when exactly one candidate meets those conditions. Missing, malformed, or ambiguous evidence remains unresolved.
- Preserve the three source bars as FVG provenance and validate that they are visible to the current subject.

## Limits

An FVG is a candidate reactionary zone, not a setup or signal. The source playbook requires contextual justification and retracement into a PD array but does not provide a deterministic touch/rejection/entry rule. The pipeline's current first-touch rejection policy is recorded as experimental in the retracement contract and has not been performance evaluated. A planned entry is not an execution fill; execution still requires a later eligible quote under the execution contract.

This is a development rule only. It reports no performance and does not authorize replay evaluation or access to the sealed 2025 outcomes.
