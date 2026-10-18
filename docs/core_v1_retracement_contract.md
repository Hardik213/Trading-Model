# Core v1 visible FVG retracement evidence contract

**Status:** Development implementation rule; not a frozen evaluation build.  
**Date:** 2026-10-18  
**Scope:** Detect a post-availability interaction with a resolved FVG candidate. This records retracement evidence only; it does not define or authorize an entry.

## Interaction rule

- Consider base-timeframe bars strictly after the FVG's availability timestamp. The formation and confirmation bars cannot count as a retracement.
- Validate each considered bar's source timing, OHLC geometry, and finite values. Malformed or non-monotonic evidence fails closed.
- The first bar whose low is at or below the FVG upper bound and whose high is at or above the FVG lower bound records an interaction.
- Use the bar timestamp as the interaction event and retain the complete FVG source lineage plus the interaction bar source.
- OHLC does not reveal the intrabar path or exact touch price. Record no exact interaction price and do not treat overlap as rejection, entry, or fill.
- A retracement-entry resolver cannot return a valid planned entry without both a zone and a post-availability interaction. An explicit governed entry policy remains required.

This deterministic interaction detector is a development rule, not a performance-tested strategy choice. It does not authorize replay evaluation or access to sealed 2025 outcomes.
