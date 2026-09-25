from __future__ import annotations

"""Automatic causal evidence construction for the ICT-2022 precision gate.

This module is intentionally conservative. It constructs evidence only from the
visible replay slice supplied for a decision timestamp. It does not fetch future
rows, infer institutional intent, or convert an unresolved event into a setup.

The builder is a bridge between chronological replay and the Phase 10.3 gate:
visible candles -> causal liquidity -> displacement -> MSS -> post-MSS FVG ->
retracement -> structural geometry -> PrecisionEvidence.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .causal_liquidity import (
    draw_on_liquidity,
    latest_reversal_liquidity,
)
from .displacement import Direction, detect_displacement
from .fvg import FVG, FVGDirection, detect_fvgs
from .market_structure import detect_confirmed_swings, LiquidityLevel
from .mss import MSSEvent, detect_mss
from .sniper_setup import PrecisionEvidence
from .timeframe_context import TimeframeContext


@dataclass(frozen=True)
class EvidenceBuildConfig:
    left_bars: int = 2
    right_bars: int = 2
    liquidity_resolution_bars: int = 3
    equal_tolerance: float = 0.0
    displacement_baseline_bars: int = 20
    displacement_follow_through_bars: int = 2
    min_body_ratio: float = 0.60
    min_range_expansion: float = 1.25
    mss_follow_through_bars: int = 2
    max_mss_lookback_bars: int = 60
    require_post_mss_fvg: bool = True
    # Deliberately optional: a new R threshold requires research approval.
    min_planned_r: Optional[float] = None


@dataclass(frozen=True)
class AutomaticEvidenceResult:
    evidence: PrecisionEvidence
    reason: str


def _visible(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df.index <= as_of].copy()


def _last_close(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    return float(df.iloc[-1]["Close"])


def _event_confirmation(event) -> Optional[pd.Timestamp]:
    return getattr(event, "confirmation_timestamp", None)


def _mss_is_causally_confirmed(mss: MSSEvent, as_of: pd.Timestamp) -> bool:
    """Compatibility gate for MSS implementations with/without confirmation metadata."""
    confirmation = _event_confirmation(mss)
    if confirmation is not None:
        return pd.Timestamp(confirmation) <= as_of

    # Older MSSEvent implementations encode follow-through but not a timestamp.
    # We refuse to invent a timestamp. Such an event is usable only when the
    # detector was built entirely from the visible slice and its follow-through
    # bars are therefore visible at as_of.
    return bool(mss.follow_through)


def _choose_post_mss_fvg(
    fvgs: list[FVG],
    *,
    direction: Direction,
    mss: MSSEvent,
    as_of: pd.Timestamp,
) -> Optional[FVG]:
    expected = FVGDirection.BULLISH if direction is Direction.BULLISH else FVGDirection.BEARISH
    candidates = []
    for fvg in fvgs:
        if fvg.direction is not expected:
            continue
        if pd.Timestamp(fvg.formation_timestamp) < pd.Timestamp(mss.timestamp):
            continue
        if pd.Timestamp(fvg.confirmation_timestamp) > as_of:
            continue
        candidates.append(fvg)
    if not candidates:
        return None
    # Prefer the most recent causal array; no future state is consulted.
    return max(candidates, key=lambda x: pd.Timestamp(x.formation_timestamp))


def _find_retracement_entry(
    df: pd.DataFrame,
    fvg: FVG,
    *,
    as_of: pd.Timestamp,
    direction: Direction,
) -> Optional[float]:
    visible = _visible(df, as_of)
    if visible.empty:
        return None

    # The entry must be a visible bar after the FVG was formed and price must
    # actually trade inside the FVG. We use the first/latest visible touch
    # deterministically; callers can later promote a more specific trigger.
    post = visible.loc[visible.index >= pd.Timestamp(fvg.formation_timestamp)]
    if post.empty:
        return None
    for _, row in post.iloc[::-1].iterrows():
        if float(row["High"]) >= fvg.lower_bound and float(row["Low"]) <= fvg.upper_bound:
            if direction is Direction.BULLISH:
                return max(fvg.lower_bound, min(fvg.upper_bound, float(row["Close"])))
            return max(fvg.lower_bound, min(fvg.upper_bound, float(row["Close"])))
    return None


def _structural_invalidation(
    evidence_event,
    *,
    direction: Direction,
) -> Optional[float]:
    if evidence_event is None:
        return None
    # Conservative structural invalidation: the rejected liquidity level.
    # This is not an execution recommendation; it is a deterministic candidate
    # boundary for the research gate.
    level = evidence_event.breach.level
    return float(level.price)


def build_automatic_evidence(
    timestamp,
    visible_base: pd.DataFrame,
    visible_context: TimeframeContext,
    *,
    direction: Optional[Direction] = None,
    config: Optional[EvidenceBuildConfig] = None,
) -> AutomaticEvidenceResult:
    """Build the strongest causally available evidence at one replay timestamp.

    Direction may be supplied by a separately governed HTF hypothesis provider.
    If omitted, it is derived conservatively from the latest visible opposing
    liquidity rejection. This is a *candidate direction*, not an independent
    predictive signal.
    """
    cfg = config or EvidenceBuildConfig()
    as_of = pd.Timestamp(timestamp)
    df = _visible(visible_base, as_of)
    if df.empty:
        return AutomaticEvidenceResult(
            PrecisionEvidence(as_of, None, None, None, None, None, None, None, None, None),
            "EMPTY_VISIBLE_DATA",
        )

    if direction is None:
        # Evaluate both reversal directions and choose the latest confirmed event.
        bull = latest_reversal_liquidity(
            df, as_of, Direction.BULLISH,
            left_bars=cfg.left_bars, right_bars=cfg.right_bars,
            resolution_bars=cfg.liquidity_resolution_bars,
            equal_tolerance=cfg.equal_tolerance,
        )
        bear = latest_reversal_liquidity(
            df, as_of, Direction.BEARISH,
            left_bars=cfg.left_bars, right_bars=cfg.right_bars,
            resolution_bars=cfg.liquidity_resolution_bars,
            equal_tolerance=cfg.equal_tolerance,
        )
        choices = [(bull, Direction.BULLISH), (bear, Direction.BEARISH)]
        choices = [(e, d) for e, d in choices if e is not None]
        if not choices:
            return AutomaticEvidenceResult(
                PrecisionEvidence(as_of, None, None, None, None, None, None, None, None, None),
                "NO_CONFIRMED_LIQUIDITY_REJECTION",
            )
        event, direction = max(
            choices,
            key=lambda x: pd.Timestamp(
                x[0].breach.resolution_timestamp or x[0].breach.breach_timestamp
            ),
        )

    direction = Direction(direction)
    liquidity = latest_reversal_liquidity(
        df, as_of, direction,
        left_bars=cfg.left_bars, right_bars=cfg.right_bars,
        resolution_bars=cfg.liquidity_resolution_bars,
        equal_tolerance=cfg.equal_tolerance,
    )
    draw = draw_on_liquidity(
        df, as_of, direction,
        current_price=_last_close(df),
        left_bars=cfg.left_bars, right_bars=cfg.right_bars,
        equal_tolerance=cfg.equal_tolerance,
    )
    draw_level: Optional[LiquidityLevel] = draw.level if draw is not None else None
    liquidity_event = liquidity.breach if liquidity is not None else None

    if liquidity_event is None:
        return AutomaticEvidenceResult(
            PrecisionEvidence(as_of, direction, draw_level, None, None, None, None, None, None, None),
            "LIQUIDITY_REJECTION_NOT_CONFIRMED",
        )

    swings = detect_confirmed_swings(
        df,
        left_bars=cfg.left_bars,
        right_bars=cfg.right_bars,
    )

    # Search only bars after the rejection resolution. The visible slice itself
    # prevents future leakage.
    resolution = liquidity_event.resolution_timestamp or liquidity_event.breach_timestamp
    start_pos = int(df.index.searchsorted(pd.Timestamp(resolution), side="right"))
    candidate_positions = range(
        start_pos,
        len(df),
    )
    # Most recent causally detectable MSS is preferred.
    mss_candidates: list[MSSEvent] = []
    for pos in candidate_positions:
        if pos >= len(df):
            break
        if pos - start_pos > cfg.max_mss_lookback_bars:
            break
        for candidate_direction in (direction,):
            displacement = detect_displacement(
                df,
                pos,
                candidate_direction,
                baseline_bars=cfg.displacement_baseline_bars,
                min_body_ratio=cfg.min_body_ratio,
                min_range_expansion=cfg.min_range_expansion,
                follow_through_bars=cfg.displacement_follow_through_bars,
            )
            if displacement is None:
                continue
            confirmation = _event_confirmation(displacement)
            if confirmation is None or pd.Timestamp(confirmation) > as_of:
                continue
            mss = detect_mss(
                df,
                swings=swings,
                liquidity_event=liquidity_event,
                displacement=displacement,
                break_position=pos,
                follow_through_bars=cfg.mss_follow_through_bars,
            )
            if mss is None:
                continue
            if not _mss_is_causally_confirmed(mss, as_of):
                continue
            mss_candidates.append(mss)

    mss = max(mss_candidates, key=lambda x: pd.Timestamp(x.timestamp)) if mss_candidates else None

    if mss is None:
        return AutomaticEvidenceResult(
            PrecisionEvidence(as_of, direction, draw_level, liquidity_event, None, None, None, None, None, None),
            "NO_CAUSALLY_CONFIRMED_MSS",
        )

    fvgs = detect_fvgs(df)
    fvg = _choose_post_mss_fvg(
        fvgs, direction=direction, mss=mss, as_of=as_of
    ) if cfg.require_post_mss_fvg else None

    if fvg is None:
        return AutomaticEvidenceResult(
            PrecisionEvidence(as_of, direction, draw_level, liquidity_event, mss, None, None, None, None, None),
            "NO_POST_MSS_FVG",
        )

    entry = _find_retracement_entry(df, fvg, as_of=as_of, direction=direction)
    if entry is None:
        return AutomaticEvidenceResult(
            PrecisionEvidence(as_of, direction, draw_level, liquidity_event, mss, fvg, None, None, None, None),
            "NO_RETRACEMENT_IN_PD_ARRAY",
        )

    invalidation = _structural_invalidation(liquidity_event, direction=direction)
    target = float(draw_level.price) if draw_level is not None else None

    return AutomaticEvidenceResult(
        PrecisionEvidence(
            as_of=as_of,
            direction=direction,
            draw_on_liquidity=draw_level,
            liquidity_event=liquidity_event,
            mss=mss,
            pd_array=fvg,
            entry_price=entry,
            invalidation_price=invalidation,
            target_price=target,
            target_liquidity=draw_level,
        ),
        "AUTOMATIC_EVIDENCE_BUILT",
    )


__all__ = ["EvidenceBuildConfig", "AutomaticEvidenceResult", "build_automatic_evidence"]
