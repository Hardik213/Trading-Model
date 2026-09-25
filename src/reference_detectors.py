from __future__ import annotations

"""
Conservative reference detectors used for Phase 9 integration tests.

These are deliberately small deterministic examples, not the final trading
rules. They demonstrate the data-flow contract and provide a safe harness for
the existing detector modules.

They never inspect future rows.
"""

import pandas as pd

from .fvg import FVG, detect_fvgs


def no_direction(*args, **kwargs):
    return None


def no_liquidity(*args, **kwargs):
    return None


def no_mss(*args, **kwargs):
    return None


def no_pd_array(*args, **kwargs):
    return None


def no_range(*args, **kwargs):
    return None


def no_price(*args, **kwargs):
    return None


def visible_fvg(
    timestamp,
    visible_base,
    visible_context,
    direction,
    mss,
):
    if direction is None:
        return None

    fvgs = detect_fvgs(visible_base)
    wanted = "BULLISH" if str(direction.value) == "BULLISH" else "BEARISH"

    candidates = [
        f for f in fvgs
        if f.confirmation_timestamp <= timestamp
        and f.direction.value == wanted
    ]
    return candidates[-1] if candidates else None


def reference_detector_bundle():
    """
    Safe default bundle.

    It intentionally establishes no direction, liquidity, MSS, range or trade.
    Only the FVG detector is wired because its purpose is to prove that an
    existing Phase 3 observation can flow through the new provider without
    introducing look-ahead.

    Replace individual callables with the actual project detector adapters
    after their signatures are confirmed.
    """
    from .historical_evidence import DetectorBundle

    return DetectorBundle(
        direction=no_direction,
        draw_on_liquidity=no_liquidity,
        liquidity_event=no_liquidity,
        mss=no_mss,
        pd_array=visible_fvg,
        dealing_range=no_range,
        entry_price=no_price,
        invalidation_price=no_price,
        target_price=no_price,
        target_liquidity=no_liquidity,
    )
