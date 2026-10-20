# Core v1 structural risk and target contract

**Status:** Development implementation rule; not a frozen evaluation build.  
**Date:** 2026-10-20  
**Scope:** Complete a valid planned-entry hypothesis with a structural invalidation and an opposing-liquidity target. This contract defines evidence selection only. It does not define a minimum R multiple, position sizing, execution, or performance acceptance.

## Structural invalidation

- The default resolver requires a governed liquidity sweep, a confirmed rejection, and an observed breach price.
- For a bullish thesis, the swept level must be sell-side liquidity (SSL); for a bearish thesis, it must be buy-side liquidity (BSL). Side/direction conflicts remain unresolved.
- The observed breach extreme is the invalidation reference. Preserve the breach event, confirmation, and source lineage; do not infer a stop buffer from OHLC data.
- The invalidation must be available by the planned-entry decision time and have geometry consistent with the thesis. Missing or contradictory evidence blocks a complete hypothesis.

## Opposing-liquidity target

- Consider only governed visible swing-liquidity levels on the opposing side of the thesis and strictly beyond the planned-entry reference.
- A candidate must be available by planned-entry decision time. Check for a breach using only visible observations whose aggregate availability is no later than that decision. Later ReplaySubject observations cannot retroactively change target eligibility for the earlier decision.
- A candidate with a visible breach is not selected. In particular, a breach without a classified reaction leaves target eligibility unresolved; this rule does not declare the level consumed. Equal-price ties fail closed. No eligible candidate means the target is unresolved.
- Keep the selected target's swing provenance and selection timing. A level's availability is the latest availability among its supporting observations; its confirmation-bar source identifies confirmation time and need not itself be the latest available support.

## Boundaries

- A valid hypothesis requires both structural invalidation and target evidence. This stage does not silently choose a minimum R:R, since the source framework does not prescribe one.
- This is a research implementation contract with synthetic-fixture tests only. It does not assert profitability, authorize strategy evaluation, or permit access to the protected 2025 outcomes, metrics, or charts.
