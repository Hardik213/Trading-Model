from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "sma_fast",
    "sma_slow",
    "ema_fast",
    "ema_slow",
    "macd",
    "signal",
    "rsi",
    "atr",
    "volatility",
    "trend_strength",
    "close_z",
    "session_bias",
    "liquidity_gap",
    "regime_score",
    "macro_score",
    "liquidity_sweep",
    "structure_strength",
    "swing_delta",
    "htf_high_24",
    "htf_low_24",
    "htf_liquidity_obj",
    "liquidity_raid_long",
    "liquidity_raid_short",
    "displacement",
    "mss_bull",
    "mss_bear",
    "pd_array_long",
    "pd_array_short",
    "structure_valid_long",
    "structure_valid_short",
    "rr_ratio",
]


def feature_row_decision_times(df: pd.DataFrame) -> pd.Series:
    """Return decision times, marking rows with missing or contradictory timing as NaT."""
    availability_mode = df.attrs.get("availability_mode")
    mtf_availability_mode = df.attrs.get("mtf_availability_mode")
    event_time_mode = (
        mtf_availability_mode == "event_time"
        or availability_mode == "event_time"
    )
    nominal_fallback = (
        mtf_availability_mode == "nominal_close_fallback"
        or availability_mode == "nominal_close_fallback"
    )
    timing_columns = [
        column
        for column in ("available_at", "availability_ts", "interval_end")
        if column in df.columns
    ]
    valid_timing = pd.Series(True, index=df.index)
    valid_modes = {None, "event_time", "nominal_close_fallback"}
    if availability_mode not in valid_modes or mtf_availability_mode not in valid_modes:
        valid_timing[:] = False
    if (
        availability_mode is not None
        and mtf_availability_mode is not None
        and availability_mode != mtf_availability_mode
    ):
        valid_timing[:] = False

    def parse(column: str) -> pd.Series:
        values = df[column]
        if event_time_mode:
            for position, value in enumerate(values):
                if pd.notna(value) and pd.Timestamp(value).tzinfo is None:
                    valid_timing.iloc[position] = False
        try:
            parsed = pd.to_datetime(values, utc=event_time_mode, errors="coerce")
        except (TypeError, ValueError):
            valid_timing[:] = False
            parsed = pd.Series(pd.NaT, index=df.index)
        return pd.Series(parsed, index=df.index)

    parsed_timing = {column: parse(column) for column in timing_columns}
    if "available_at" in parsed_timing and "availability_ts" in parsed_timing:
        matches = parsed_timing["available_at"].eq(parsed_timing["availability_ts"]) | (
            parsed_timing["available_at"].isna()
            & parsed_timing["availability_ts"].isna()
        )
        valid_timing &= matches

    source_column = next(
        (
            column
            for column in ("available_at", "availability_ts", "interval_end")
            if column in parsed_timing
        ),
        None,
    )
    source_times = (
        parsed_timing[source_column]
        if source_column is not None
        else pd.Series(pd.NaT, index=df.index)
    )
    nominal_close = None
    if nominal_fallback:
        if timing_columns:
            valid_timing &= False
        bar_label = df.attrs.get("bar_label")
        timeframe = df.attrs.get("timeframe")
        if bar_label not in {"left", "right"} or (
            bar_label == "left" and timeframe is None
        ):
            valid_timing &= False
        else:
            from .timeframe_context import _nominal_close_times

            try:
                nominal_close = pd.Series(
                    _nominal_close_times(df, str(timeframe or "1min")),
                    index=df.index,
                )
            except (TypeError, ValueError):
                valid_timing &= False
    if event_time_mode and source_column is None:
        valid_timing &= False
    if "interval_end" in parsed_timing and source_column != "interval_end":
        interval_end = parsed_timing["interval_end"]
        valid_timing &= interval_end.notna() & (
            source_times.isna() | (source_times >= interval_end)
        )
    elif event_time_mode and source_column in {"available_at", "availability_ts"}:
        timeframe = df.attrs.get("timeframe")
        bar_label = str(df.attrs.get("bar_label", "left")).lower()
        if timeframe is None and bar_label != "right":
            valid_timing &= False
        else:
            from .timeframe_context import _nominal_close_times

            nominal_close = _nominal_close_times(
                df, str(timeframe or "1min")
            )
            nominal_close = pd.Series(nominal_close, index=df.index)
            valid_timing &= source_times >= nominal_close

    if "mtf_decision_time" in df.columns:
        decision_times = parse("mtf_decision_time")
        valid_timing &= decision_times.notna()
        if source_column is not None:
            valid_timing &= source_times.notna() & (decision_times >= source_times)
    elif source_column is not None:
        decision_times = source_times
        valid_timing &= decision_times.notna()
    else:
        decision_times = pd.Series(df.index, index=df.index)
        if event_time_mode:
            valid_timing &= False

    if nominal_fallback and nominal_close is not None:
        if "mtf_decision_time" in df.columns:
            valid_timing &= decision_times >= nominal_close
        else:
            decision_times = nominal_close

    return decision_times.where(valid_timing)


def feature_row_eligibility_mask(df: pd.DataFrame) -> pd.Series:
    """Identify model rows whose features are present and available at decision time."""
    values = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    eligible = values.notna().all(axis=1)
    decision_times = feature_row_decision_times(df)
    event_time_mode = (
        df.attrs.get("mtf_availability_mode") == "event_time"
        or df.attrs.get("availability_mode") == "event_time"
    )
    eligible &= decision_times.notna()

    availability_subjects = list(FEATURE_COLUMNS)
    if "regime_bias" in df.columns:
        availability_subjects.append("regime_bias")
    availability_columns = {
        f"{feature}_available_at"
        for feature in availability_subjects
        if f"{feature}_available_at" in df.columns
    }
    has_liquidity_confirmation = "liquidity_sweep" in df.columns
    if has_liquidity_confirmation:
        sweep_is_present = df["liquidity_sweep"].fillna(0.0).ne(0.0)
        confirmation_column = "liquidity_confirmation_available_at"
        if confirmation_column not in df.columns:
            eligible &= ~sweep_is_present
        else:
            timestamps = df[confirmation_column]
            if event_time_mode:
                for value in timestamps.dropna():
                    if pd.Timestamp(value).tzinfo is None:
                        raise ValueError("Liquidity confirmation timestamps must be timezone-aware.")
            confirmation_available = pd.to_datetime(
                timestamps, utc=event_time_mode, errors="raise"
            )
            if not event_time_mode and confirmation_available.dt.tz != decision_times.dt.tz:
                raise ValueError("Liquidity confirmation and decision times must use consistent timezone awareness.")
            source_availability_columns = [
                column
                for column in ("available_at", "availability_ts", "interval_end")
                if column in df.columns
            ]
            if source_availability_columns:
                source_availability = pd.concat(
                    [
                        pd.to_datetime(df[column], utc=event_time_mode, errors="raise")
                        for column in source_availability_columns
                    ],
                    axis=1,
                ).max(axis=1)
                confirmation_after_source = confirmation_available >= source_availability
            else:
                confirmation_after_source = confirmation_available.notna()
            eligible &= ~sweep_is_present | (
                confirmation_available.notna()
                & confirmation_after_source
                & (confirmation_available <= decision_times)
            )
    if "mtf_decision_time" in df.columns and any(
        feature in df.columns and f"{feature}_available_at" not in df.columns
        for feature in ("regime_score", "regime_bias")
    ):
        eligible &= False
    for column in sorted(availability_columns):
        timestamps = df[column]
        if event_time_mode:
            for value in timestamps.dropna():
                if pd.Timestamp(value).tzinfo is None:
                    raise ValueError(f"{column} timestamps must be timezone-aware.")
        available_at = pd.to_datetime(
            timestamps, utc=event_time_mode, errors="raise"
        )
        if not event_time_mode and available_at.dt.tz != decision_times.dt.tz:
            raise ValueError("Feature availability and decision times must use consistent timezone awareness.")
        eligible &= available_at.notna() & (available_at <= decision_times)

    return eligible


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ret_1"] = data["Close"].pct_change().fillna(0.0)
    data["ret_5"] = data["Close"].pct_change(5).fillna(0.0)
    data["sma_fast"] = data["Close"].rolling(window=10).mean()
    data["sma_slow"] = data["Close"].rolling(window=30).mean()
    data["ema_fast"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["ema_slow"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_fast"] - data["ema_slow"]
    data["signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    data["rsi"] = 100 - (100 / (1 + rs)).fillna(50.0)

    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    data["true_range"] = high_low.combine(high_close, max).combine(low_close, max)
    data["atr"] = data["true_range"].rolling(14).mean().fillna(0.0)
    data["volatility"] = data["ret_1"].rolling(20).std().fillna(0.0)
    data["trend_strength"] = ((data["Close"] - data["Close"].shift(20)) / data["Close"].shift(20)).fillna(0.0)
    rolling_mean = data["Close"].rolling(20).mean()
    rolling_std = data["Close"].rolling(20).std().replace(0, np.nan)
    data["close_z"] = ((data["Close"] - rolling_mean) / rolling_std).fillna(0.0)

    session_high = data["High"].rolling(window=24).max()
    session_low = data["Low"].rolling(window=24).min()
    data["session_bias"] = np.where(
        data["Close"] > session_high.shift(1), 1.0,
        np.where(data["Close"] < session_low.shift(1), -1.0, 0.0)
    )
    data["liquidity_gap"] = ((data["Close"] - data["Close"].shift(24)) / data["Close"].shift(24)).fillna(0.0)
    data["structure_strength"] = ((data["Close"] - data["Close"].shift(12)) / data["Close"].shift(12)).fillna(0.0)
    data["swing_delta"] = ((data["High"].rolling(12).max() - data["Low"].rolling(12).min()) / data["Close"]).fillna(0.0)
    return data


def add_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    htf_high = data["High"].rolling(24).max().shift(1)
    htf_low = data["Low"].rolling(24).min().shift(1)
    data["htf_high_24"] = htf_high
    data["htf_low_24"] = htf_low
    data["htf_mid_24"] = (htf_high + htf_low) / 2.0
    data["htf_liquidity_obj"] = np.where(data["Close"] >= data["htf_mid_24"], htf_high, htf_low)

    data["liquidity_raid_long"] = (data["High"] > htf_high) & (data["Close"] < htf_high)
    data["liquidity_raid_short"] = (data["Low"] < htf_low) & (data["Close"] > htf_low)

    atr = data["atr"].replace(0, np.nan).ffill().fillna(1.0)
    data["displacement"] = (data["Close"] - data["Close"].shift(5)).abs() / atr

    swing_high = data["High"].shift(1).rolling(8).max()
    swing_low = data["Low"].shift(1).rolling(8).min()
    data["mss_bull"] = (data["Close"] > swing_high) & (data["Low"] > data["Low"].shift(1).rolling(8).min())
    data["mss_bear"] = (data["Close"] < swing_low) & (data["High"] < data["High"].shift(1).rolling(8).max())

    risk_buffer = atr * 0.8
    data["pd_array_long"] = (
        (data["Close"] > data["Close"].shift(5) - risk_buffer)
        & (data["Close"] < data["Close"].shift(5) + risk_buffer)
        & (data["Close"] > data["Low"].shift(1).rolling(5).min())
    )
    data["pd_array_short"] = (
        (data["Close"] < data["Close"].shift(5) + risk_buffer)
        & (data["Close"] > data["Close"].shift(5) - risk_buffer)
        & (data["Close"] < data["High"].shift(1).rolling(5).max())
    )

    data["structure_valid_long"] = (data["Close"] > swing_low) & (data["Low"] > swing_low * 0.999)
    data["structure_valid_short"] = (data["Close"] < swing_high) & (data["High"] < swing_high * 1.001)

    long_risk = np.maximum(data["Close"] - htf_low, 1e-6)
    short_risk = np.maximum(htf_high - data["Close"], 1e-6)
    long_target = np.maximum(htf_high - data["Close"], 1e-6)
    short_target = np.maximum(data["Close"] - htf_low, 1e-6)
    data["rr_ratio"] = np.where(
        data["Close"] >= data["htf_mid_24"],
        long_target / long_risk,
        short_target / short_risk,
    )
    return data


def build_training_frame(df: pd.DataFrame, horizon: int = 12, threshold: float = 0.003) -> tuple[pd.DataFrame, pd.Series]:
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    close_values = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    target_valid = pd.Series(
        np.isfinite(future_return.to_numpy(dtype=float))
        & np.isfinite(close_values)
        & (close_values != 0.0),
        index=df.index,
    )
    target_timing_valid = feature_row_decision_times(df).notna()
    for column in ("historical_complete", "is_complete"):
        if column in df.columns:
            target_timing_valid &= df[column].fillna(False).astype(bool)
    target_timing_valid = target_timing_valid.shift(-horizon, fill_value=False)
    eligible = feature_row_eligibility_mask(df)
    frame = df.loc[eligible & target_valid & target_timing_valid].copy()
    frame["future_return"] = future_return.loc[frame.index]
    labels = (frame["future_return"] > threshold).astype(int)
    X = frame[FEATURE_COLUMNS].copy()
    return X, labels
