import numpy as np
import pandas as pd
import pytest

from src.multitimeframe import add_multitimeframe_features
from src.regime import add_multitimeframe_regime
from src.features import FEATURE_COLUMNS, add_technical_features, build_training_frame
from src.features import (
    build_training_frame,
    feature_row_decision_times,
    feature_row_eligibility_mask,
)
from src.models import train_signal_model
from src.backtest import run_backtest
from src.models import predict_probabilities


def _hourly_bars(periods=48):
    index=pd.date_range("2026-01-01T00:00:00Z",periods=periods,freq="1h")
    close=np.arange(100.0,100.0+periods)
    return pd.DataFrame(
        {
            "Open":close-0.25,
            "High":close+0.5,
            "Low":close-0.5,
            "Close":close,
        },
        index=index,
    )


def test_higher_timeframe_features_wait_for_nominal_close():
    base=_hourly_bars()
    result=add_multitimeframe_features(base)

    assert result.loc[base.index[:23],"daily_trend"].isna().all()
    assert result.loc[base.index[23],"daily_trend"] == 0.0
    assert result.loc[base.index[24],"daily_trend"] == (
        (base["Close"].iloc[23]-base["Close"].iloc[24])
        / base["Close"].iloc[24]
    )

    assert result.loc[base.index[:7],"mtf_trend_4h"].isna().all()
    expected_4h=(base["Close"].iloc[7]-base["Close"].iloc[3])/base["Close"].iloc[3]
    assert np.isclose(result.loc[base.index[7],"mtf_trend_4h"], expected_4h)


def test_interval_end_without_available_at_controls_feature_availability():
    base=_hourly_bars(periods=20)
    base["interval_end"]=base.index+pd.Timedelta(hours=2)

    result=add_multitimeframe_features(base)
    expected=(base["Close"].iloc[10]-base["Close"].iloc[6])/base["Close"].iloc[6]

    assert pd.isna(result.loc[base.index[9],"mtf_trend_4h"])
    assert np.isclose(result.loc[base.index[10],"mtf_trend_4h"], expected)


def test_regime_enrichment_preserves_causal_mtf_values_and_missingness():
    base=_hourly_bars(periods=16)
    base["available_at"]=base.index+pd.Timedelta(hours=1)
    base.loc[base.index[7],"available_at"]+=pd.Timedelta(minutes=30)
    mtf=add_multitimeframe_features(base)

    assert pd.isna(mtf.loc[base.index[6],"mtf_trend_4h"])
    assert np.isclose(mtf.loc[base.index[7],"mtf_trend_4h"], (base["Close"].iloc[7]-base["Close"].iloc[3])/base["Close"].iloc[3])

    enriched=add_multitimeframe_regime(mtf.assign(trend_strength=0.0))
    assert pd.isna(enriched.loc[base.index[6],"mtf_trend_4h"])
    assert pd.isna(enriched.loc[base.index[6],"regime_score"])
    assert pd.isna(enriched.loc[base.index[6],"regime_bias"])
    assert np.isclose(enriched.loc[base.index[7],"mtf_trend_4h"], mtf.loc[base.index[7],"mtf_trend_4h"])


def test_delayed_event_time_higher_bar_waits_for_last_constituent():
    base=_hourly_bars(periods=16)
    base["interval_end"]=base.index+pd.Timedelta(hours=1)
    base["available_at"]=base["interval_end"]
    base.loc[base.index[7],"available_at"]=pd.Timestamp("2026-01-01T09:30:00Z")
    base.loc[base.index[8],"available_at"]=pd.Timestamp("2026-01-01T09:30:00Z")
    base["historical_complete"]=True
    base["is_complete"]=True

    result=add_multitimeframe_features(base)
    assert pd.isna(result.loc[base.index[6],"mtf_trend_4h"])
    assert np.isclose(
        result.loc[base.index[7],"mtf_trend_4h"],
        (base["Close"].iloc[7]-base["Close"].iloc[3])/base["Close"].iloc[3],
    )
    assert not pd.isna(result.loc[base.index[9],"mtf_trend_4h"])
    assert not pd.isna(result.loc[base.index[10],"mtf_trend_4h"])


def test_missing_or_incomplete_constituents_do_not_reuse_stale_higher_features():
    base=_hourly_bars(periods=20).drop(pd.Timestamp("2026-01-01T02:00:00Z"))
    result=add_multitimeframe_features(base)

    assert pd.isna(result.loc[pd.Timestamp("2026-01-01T04:00:00Z"),"mtf_trend_4h"])
    assert pd.isna(result.loc[pd.Timestamp("2026-01-01T08:00:00Z"),"mtf_trend_4h"])
    assert not pd.isna(result.loc[pd.Timestamp("2026-01-01T12:00:00Z"),"mtf_trend_4h"])

    incomplete=_hourly_bars(periods=20)
    incomplete["historical_complete"]=True
    incomplete.loc[incomplete.index[2],"historical_complete"]=False
    incomplete_result=add_multitimeframe_features(incomplete)
    assert pd.isna(incomplete_result.loc[pd.Timestamp("2026-01-01T04:00:00Z"),"mtf_trend_4h"])


def test_multitimeframe_features_are_deterministic_and_keep_schema():
    base=_hourly_bars()
    first=add_multitimeframe_features(base)
    second=add_multitimeframe_features(base)
    pd.testing.assert_frame_equal(first,second)
    assert {
        "mtf_trend_4h",
        "mtf_bias_4h",
        "mtf_strength_4h",
        "daily_trend",
    }.issubset(first.columns)


def test_unavailable_mtf_evidence_survives_production_feature_preparation():
    base=_hourly_bars(periods=48)
    enriched=add_technical_features(base)
    enriched=add_multitimeframe_features(enriched)
    enriched=add_multitimeframe_regime(enriched)
    for column in FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column]=1.0
    enriched["liquidity_sweep"] = 0.0
    enriched[FEATURE_COLUMNS]=enriched[FEATURE_COLUMNS].replace(
        [float("inf"),float("-inf")],float("nan")
    )

    X,_=build_training_frame(enriched)

    assert pd.isna(enriched.loc[base.index[6],"regime_score"])
    assert base.index[6] not in X.index
    assert not pd.isna(enriched.loc[base.index[29],"regime_score"])
    assert base.index[29] in X.index


def test_backtest_skips_rows_with_unavailable_mtf_model_features():
    class RecordingModel:
        def __init__(self):
            self.feature_rows=[]

        def predict_proba(self, features):
            self.feature_rows.append(features.copy())
            return np.array([[0.1,0.9]])

    index=pd.date_range("2026-01-01T00:00:00Z",periods=5,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["regime_score"]=[1.0,np.nan,1.0,1.0,1.0]
    frame["ret_1"]=0.01
    frame["sma_fast"]=2.0
    frame["sma_slow"]=1.0
    frame["session_bias"]=1.0
    frame["regime_bias"]=1.0
    frame["macro_bias"]=0.0
    frame["liquidity_sweep"]=0.0
    model=RecordingModel()

    run_backtest(frame,model)

    assert [features.index[0] for features in model.feature_rows]==[index[2],index[3]]
    assert not model.feature_rows[0].isna().any().any()


def test_mtf_features_preserve_actual_availability_times():
    base=_hourly_bars(periods=40)
    base["interval_end"]=base.index+pd.Timedelta(hours=1)
    base["available_at"]=base["interval_end"]+pd.Timedelta(hours=1)
    base["historical_complete"]=True
    base["is_complete"]=True

    result=add_multitimeframe_features(base)
    decision_row=base.index[31]
    expected_availability=base.loc[decision_row,"available_at"]

    assert not pd.isna(result.loc[decision_row,"mtf_trend_4h"])
    assert result.loc[decision_row,"mtf_decision_time"]==expected_availability
    assert result.loc[decision_row,"mtf_trend_4h_available_at"]==expected_availability
    assert result.loc[decision_row,"mtf_bias_4h_available_at"]==expected_availability
    assert result.loc[decision_row,"daily_trend_available_at"]==expected_availability
    assert not pd.isna(result.loc[decision_row,"mtf_strength_4h"])
    assert result.loc[decision_row,"mtf_strength_4h_available_at"]<=expected_availability


def test_training_frame_excludes_features_unavailable_at_decision_time():
    base=_hourly_bars(periods=40)
    enriched=add_technical_features(base)
    enriched=add_multitimeframe_features(enriched)
    enriched=add_multitimeframe_regime(enriched)
    for column in FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column]=1.0
    enriched["liquidity_sweep"] = 0.0
    target=base.index[31]
    enriched.loc[target,"regime_score_available_at"]=target+pd.Timedelta(hours=2)

    X,_=build_training_frame(enriched,horizon=2)

    assert target not in X.index
    assert base.index[32] in X.index


def test_training_label_horizon_counts_original_bars_across_skipped_rows():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=8,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    close=np.full(8,100.0)
    close[3]=110.0
    close[4]=90.0
    frame["Close"]=close
    frame["liquidity_sweep"]=0.0
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame.loc[index[2],"regime_score_available_at"]=index[2]+pd.Timedelta(hours=1)

    X,y=build_training_frame(frame,horizon=2,threshold=0.05)

    assert index[2] not in X.index
    assert y.loc[index[1]]==1


def test_legacy_naive_ohlc_keeps_nominal_close_feature_eligibility():
    base=_hourly_bars(periods=40)
    base.index=base.index.tz_localize(None)
    enriched=add_technical_features(base)
    enriched=add_multitimeframe_features(enriched)
    enriched=add_multitimeframe_regime(enriched)
    for column in FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column]=1.0
    enriched["liquidity_sweep"] = 0.0

    X,_=build_training_frame(enriched,horizon=2)

    assert base.index[29] in X.index


def test_backtest_does_not_predict_before_regime_feature_availability():
    class RecordingModel:
        def __init__(self):
            self.predicted_rows=[]

        def predict_proba(self, features):
            self.predicted_rows.append(features.index[0])
            return np.array([[0.01,0.99]])

    index=pd.date_range("2026-01-01T00:00:00Z",periods=5,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["ret_1"]=0.01
    frame["sma_fast"]=2.0
    frame["sma_slow"]=1.0
    frame["session_bias"]=1.0
    frame["regime_bias"]=1.0
    frame["macro_bias"]=0.0
    frame["liquidity_sweep"]=0.0
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame.loc[index[1],"regime_score_available_at"]=index[1]+pd.Timedelta(hours=1)
    frame.loc[index[1],"regime_bias_available_at"]=index[1]+pd.Timedelta(hours=1)
    model=RecordingModel()

    run_backtest(frame,model)

    assert model.predicted_rows==[index[2],index[3]]


def test_delayed_mtf_availability_is_preserved_and_backtest_uses_decision_time():
    base=_hourly_bars(periods=40)
    base["interval_end"]=base.index+pd.Timedelta(hours=1)
    base["available_at"]=base["interval_end"]
    delayed_at=pd.Timestamp("2026-01-01T09:30:00Z")
    base.loc[base.index[7:9],"available_at"]=delayed_at
    base["historical_complete"]=True
    base["is_complete"]=True

    mtf=add_multitimeframe_features(base)
    decision_index=base.index[7]
    assert not pd.isna(mtf.loc[decision_index,"mtf_trend_4h"])
    assert mtf.loc[decision_index,"mtf_trend_4h_available_at"]==delayed_at
    assert mtf.loc[decision_index,"mtf_decision_time"]==delayed_at

    frame=pd.DataFrame(1.0,index=base.index,columns=FEATURE_COLUMNS)
    frame["regime_score"]=mtf["mtf_trend_4h"]
    frame["regime_score_available_at"]=mtf["mtf_trend_4h_available_at"]
    frame["regime_bias_available_at"]=mtf["mtf_trend_4h_available_at"]
    frame["mtf_decision_time"]=mtf["mtf_decision_time"]
    frame["ret_1"]=0.01
    frame["ret_5"]=np.arange(len(frame),dtype=float)
    frame["sma_fast"]=2.0
    frame["sma_slow"]=1.0
    frame["session_bias"]=1.0
    frame["regime_bias"]=1.0
    frame["macro_bias"]=0.0
    frame["liquidity_sweep"]=0.0

    class RecordingModel:
        def __init__(self):
            self.rows=[]

        def predict_proba(self,features):
            self.rows.append((features.index[0],features["ret_5"].iloc[0]))
            return np.array([[0.01,0.99]])

    model=RecordingModel()
    run_backtest(frame,model)
    delayed_row_predictions=[time for time,row_id in model.rows if row_id==7.0]

    assert delayed_row_predictions==[decision_index]

    label_time_frame=frame.copy()
    label_time_frame.loc[decision_index,"mtf_decision_time"]=decision_index
    label_time_model=RecordingModel()
    run_backtest(label_time_frame,label_time_model)
    assert decision_index not in [time for time,row_id in label_time_model.rows]


def test_training_and_inference_exclude_mtf_features_not_yet_available():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=48,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=np.arange(100.0,148.0)
    frame["regime_score_available_at"]=pd.Series(
        [timestamp for timestamp in index],index=index,dtype=object
    )
    frame["mtf_decision_time"]=pd.Series(
        [timestamp for timestamp in index],index=index,dtype=object
    )
    unavailable_row=index[32]
    frame.loc[unavailable_row,"regime_score_available_at"]+=pd.Timedelta(hours=1)

    X,y=build_training_frame(frame)

    class RecordingModel:
        def predict_proba(self,features):
            self.index=features.index.copy()
            return np.tile([0.25,0.75],(len(features),1))

    model=RecordingModel()
    probabilities=predict_probabilities(model,X)

    assert unavailable_row not in X.index
    assert unavailable_row not in model.index
    assert unavailable_row not in probabilities.index


def test_training_frame_excludes_liquidity_sweep_until_confirmation_available():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=8,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=np.arange(100.0,108.0)
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    frame["liquidity_confirmation_available_at"]=pd.Series(
        [pd.NaT,index[1]+pd.Timedelta(hours=1),*([pd.NaT]*6)],
        index=index,
        dtype=object,
    )

    X,_=build_training_frame(frame,horizon=2)

    assert index[1] not in X.index


def test_training_frame_excludes_liquidity_confirmation_not_yet_available():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=8,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=np.arange(100.0,108.0)
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    frame["liquidity_confirmation_available_at"]=pd.Series(
        [pd.NaT,index[1]+pd.Timedelta(hours=1),*([pd.NaT]*6)],
        index=index,
        dtype=object,
    )

    X,_=build_training_frame(frame,horizon=2)

    assert index[1] not in X.index


def test_backtest_skips_delayed_liquidity_sweep_and_its_confirmation_return():
    class RecordingModel:
        def __init__(self):
            self.rows=[]

        def predict_proba(self,features):
            self.rows.append(features.index[0])
            return np.array([[0.01,0.99]])

    index=pd.date_range("2026-01-01T00:00:00Z",periods=5,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["ret_1"]=0.0
    frame.loc[index[1],"ret_1"]=0.10
    frame["sma_fast"]=[1.0,2.0,1.0,1.0,1.0]
    frame["sma_slow"]=1.5
    frame["session_bias"]=1.0
    frame["regime_bias"]=[-1.0,1.0,-1.0,-1.0,-1.0]
    frame["macro_bias"]=1.0
    frame["liquidity_sweep"]=[0.0,1.0,0.0,0.0,0.0]
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_confirmation_available_at"]=pd.Series(
        [pd.NaT,index[1]+pd.Timedelta(hours=1),*([pd.NaT]*3)],
        index=index,
        dtype=object,
    )
    model=RecordingModel()

    result=run_backtest(frame,model)

    assert index[1] not in model.rows
    assert result["trades"]==0
    assert result["final_capital"]==result["initial_capital"]


def test_backtest_does_not_consume_delayed_liquidity_confirmation():
    class RecordingModel:
        def __init__(self):
            self.rows=[]

        def predict_proba(self,features):
            self.rows.append(features.index[0])
            return np.array([[0.01,0.99]])

    index=pd.date_range("2026-01-01T00:00:00Z",periods=5,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["ret_1"]=[0.0,0.10,0.0,0.0,0.0]
    frame["sma_fast"]=[1.0,2.0,1.0,1.0,1.0]
    frame["sma_slow"]=1.5
    frame["session_bias"]=1.0
    frame["regime_bias"]=[-1.0,1.0,-1.0,-1.0,-1.0]
    frame["macro_bias"]=1.0
    frame["liquidity_sweep"]=[0.0,1.0,0.0,0.0,0.0]
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_confirmation_available_at"]=pd.Series(
        [pd.NaT,index[1]+pd.Timedelta(hours=1),*([pd.NaT]*3)],
        index=index,
        dtype=object,
    )
    model=RecordingModel()

    result=run_backtest(frame,model)

    assert index[1] not in model.rows
    assert result["trades"]==0
    assert result["final_capital"]==result["initial_capital"]


@pytest.mark.parametrize("availability_mode", [None, "event_time"])
def test_nonzero_sweep_without_confirmation_time_is_ineligible(availability_mode):
    index=pd.date_range("2026-01-01T00:00:00Z",periods=3,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=1.0
    if availability_mode is not None:
        frame.attrs["availability_mode"]=availability_mode

    assert not feature_row_eligibility_mask(frame).any()


def test_fully_available_liquidity_confirmation_remains_eligible():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=3,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    available_at=index+pd.Timedelta(hours=1)
    frame["mtf_decision_time"]=available_at
    frame["regime_score_available_at"]=pd.Series(available_at,index=index)
    frame["regime_bias_available_at"]=pd.Series(available_at,index=index)
    frame["liquidity_sweep"]=1.0
    frame["available_at"]=pd.Series(available_at,index=index)
    frame["liquidity_confirmation_available_at"]=pd.Series(available_at,index=index)
    frame.attrs["availability_mode"]="event_time"
    frame.attrs["timeframe"]="1h"
    frame.attrs["bar_label"]="left"

    assert feature_row_eligibility_mask(frame).all()


def test_source_bar_availability_cannot_follow_mtf_decision_time():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=4,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index
    frame["available_at"]=index
    frame["available_at"]=frame["available_at"]+pd.Timedelta(hours=1)
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=0.0
    frame.attrs["availability_mode"]="event_time"
    frame.loc[index[1],"mtf_decision_time"]=index[1]+pd.Timedelta(minutes=30)

    assert not feature_row_eligibility_mask(frame).loc[index[1]]


def test_contradictory_availability_aliases_exclude_feature_row():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=4,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index+pd.Timedelta(hours=2)
    frame["available_at"]=index+pd.Timedelta(hours=1)
    frame["availability_ts"]=frame["available_at"]
    frame.loc[index[1],"availability_ts"]=index[1]+pd.Timedelta(hours=2)
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=0.0

    assert not feature_row_eligibility_mask(frame).loc[index[1]]


def test_event_time_row_without_source_availability_is_ineligible():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=4,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=0.0
    frame.attrs["availability_mode"]="event_time"

    assert not feature_row_eligibility_mask(frame).any()


def test_event_availability_without_endpoint_must_reach_nominal_close():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=4,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index
    frame["available_at"]=index
    frame["regime_score_available_at"]=pd.Series(index,index=index)
    frame["regime_bias_available_at"]=pd.Series(index,index=index)
    frame["liquidity_sweep"]=0.0
    frame.attrs["availability_mode"]="event_time"
    frame.attrs["timeframe"]="1h"
    frame.attrs["bar_label"]="left"

    assert not feature_row_eligibility_mask(frame).any()


@pytest.mark.parametrize(
    ("decision_offset", "expected_eligible"),
    [
        (pd.Timedelta(minutes=59), False),
        (pd.Timedelta(hours=1), True),
        (pd.Timedelta(hours=1, minutes=1), True),
    ],
)
def test_nominal_close_fallback_enforces_source_bar_boundary(
    decision_offset, expected_eligible
):
    index=pd.date_range("2026-01-01T00:00:00Z",periods=4,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index+pd.Timedelta(hours=1)
    frame.loc[index[1],"mtf_decision_time"]=index[1]+decision_offset
    frame["regime_score_available_at"]=index+pd.Timedelta(hours=1)
    frame["regime_bias"]=1.0
    frame["regime_bias_available_at"]=index+pd.Timedelta(hours=1)
    frame["liquidity_sweep"]=0.0
    frame.attrs["availability_mode"]="nominal_close_fallback"
    frame.attrs["mtf_availability_mode"]="nominal_close_fallback"
    frame.attrs["timeframe"]="1h"
    frame.attrs["bar_label"]="left"

    assert bool(feature_row_eligibility_mask(frame).loc[index[1]]) is expected_eligible


@pytest.mark.parametrize(
    "metadata",
    [
        {"availability_mode": "nominal_close_fallback", "bar_label": "left"},
        {"availability_mode": "nominal_close_fallback", "timeframe": "1h"},
        {
            "availability_mode": "nominal_close_fallback",
            "mtf_availability_mode": "event_time",
            "timeframe": "1h",
            "bar_label": "left",
        },
    ],
)
def test_nominal_fallback_missing_or_contradictory_metadata_fails_closed(metadata):
    index=pd.date_range("2026-01-01T00:00:00Z",periods=3,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index+pd.Timedelta(hours=1)
    frame["regime_score_available_at"]=index+pd.Timedelta(hours=1)
    frame["regime_bias"]=1.0
    frame["regime_bias_available_at"]=index+pd.Timedelta(hours=1)
    frame["liquidity_sweep"]=0.0
    frame.attrs.update(metadata)

    assert not feature_row_eligibility_mask(frame).any()


def test_nominal_fallback_preserves_explicit_right_labeled_legacy_timing():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=3,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=index
    frame["regime_bias"]=1.0
    frame["regime_bias_available_at"]=index
    frame["liquidity_sweep"]=0.0
    frame.attrs["availability_mode"]="nominal_close_fallback"
    frame.attrs["timeframe"]="1h"
    frame.attrs["bar_label"]="right"

    assert feature_row_eligibility_mask(frame).all()


def test_training_frame_excludes_incomplete_future_targets():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=10,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=100.0*(1.02**np.arange(len(index)))
    frame["liquidity_sweep"]=0.0

    X,y=build_training_frame(frame,horizon=3,threshold=0.01)

    assert index[-3:].difference(X.index).tolist()==index[-3:].tolist()
    assert index[-3:].difference(y.index).tolist()==index[-3:].tolist()
    assert y.notna().all()


@pytest.mark.parametrize("missing_position",[4,6,8])
def test_training_frame_excludes_rows_with_missing_future_close(missing_position):
    index=pd.date_range("2026-01-01T00:00:00Z",periods=10,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=100.0*(1.02**np.arange(len(index)))
    frame.loc[index[missing_position],"Close"]=np.nan
    frame["liquidity_sweep"]=0.0
    horizon=2

    X,y=build_training_frame(frame,horizon=horizon,threshold=0.01)

    affected=index[[
        position
        for position in range(len(index))
        if position + horizon >= len(index)
        or not np.isfinite(frame["Close"].iloc[position])
        or not np.isfinite(frame["Close"].iloc[position+horizon])
    ]]
    assert not affected.isin(X.index).any()
    assert X.index.equals(y.index)
    assert y.notna().all()


def test_training_frame_excludes_future_target_without_complete_availability():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=8,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=100.0*(1.02**np.arange(len(index)))
    frame["available_at"]=index+pd.Timedelta(hours=1)
    frame["interval_end"]=frame["available_at"]
    frame["mtf_decision_time"]=frame["available_at"]
    frame["regime_score_available_at"]=frame["available_at"]
    frame["regime_bias"]=1.0
    frame["regime_bias_available_at"]=frame["available_at"]
    frame["historical_complete"]=True
    frame["is_complete"]=True
    frame.loc[index[5],"available_at"]=pd.NaT
    frame.loc[index[5],"historical_complete"]=False
    frame.loc[index[5],"is_complete"]=False
    frame["liquidity_sweep"]=0.0
    frame.attrs["availability_mode"]="event_time"
    frame.attrs["mtf_availability_mode"]="event_time"
    frame.attrs["timeframe"]="1h"
    frame.attrs["bar_label"]="left"

    X,y=build_training_frame(frame,horizon=2,threshold=0.01)

    assert index[3] not in X.index
    assert index[5] not in X.index
    assert X.index.equals(y.index)


def test_training_targets_stay_on_original_sequence_after_causal_filtering():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=10,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=100.0*(1.02**np.arange(len(index)))
    frame["liquidity_sweep"]=0.0
    frame["mtf_decision_time"]=index
    frame["regime_score_available_at"]=index
    frame["regime_bias"]=1.0
    frame["regime_bias_available_at"]=index
    frame.loc[index[2],"regime_score_available_at"]=index[2]+pd.Timedelta(hours=1)

    X,y=build_training_frame(frame,horizon=2,threshold=0.01)

    assert index[2] not in X.index
    assert y.loc[index[1]]==1
    assert np.isclose(
        y.loc[index[1]],
        int(frame["Close"].iloc[3]/frame["Close"].iloc[1]-1.0>0.01),
    )


def test_every_training_row_has_valid_target_before_model_fitting():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=12,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["Close"]=[100.0,104.0,102.0,100.0,106.0,101.0,99.0,110.0,108.0,111.0,112.0,114.0]
    frame.loc[index[-2],"Close"]=np.nan
    frame["liquidity_sweep"]=0.0

    X,y=build_training_frame(frame,horizon=2,threshold=0.01)
    model=train_signal_model(X,y)

    assert X.index.equals(y.index)
    assert X.index[-1] == index[-3]
    assert model.named_steps["scaler"].n_samples_seen_ == len(X)


def test_live_final_bar_without_completeness_or_timing_is_excluded():
    live=_hourly_bars(periods=16)
    live.attrs["market_source"]="live_mt5"

    result=add_multitimeframe_features(live)

    assert pd.isna(result["mtf_decision_time"].iloc[-1])


def test_live_final_bar_with_timing_but_no_completeness_is_excluded():
    live=_hourly_bars(periods=16)
    live["interval_end"]=live.index+pd.Timedelta(hours=1)
    live["available_at"]=live["interval_end"]
    live.attrs["availability_mode"]="event_time"
    live.attrs["market_source"]="live_mt5"

    result=add_multitimeframe_features(live)

    assert pd.isna(result["mtf_decision_time"].iloc[-1])


def test_live_complete_bar_with_valid_event_availability_remains_eligible():
    live=_hourly_bars(periods=16)
    live["interval_end"]=live.index+pd.Timedelta(hours=1)
    live["available_at"]=live["interval_end"]
    live["historical_complete"]=True
    live["is_complete"]=True
    live.attrs["availability_mode"]="event_time"
    live.attrs["market_source"]="live_mt5"

    result=add_multitimeframe_features(live)

    assert result["mtf_decision_time"].iloc[-1]==live["available_at"].iloc[-1]


def test_live_forming_htf_constituent_cannot_enter_model_eligibility():
    live=_hourly_bars(periods=48)
    live["interval_end"]=live.index+pd.Timedelta(hours=1)
    live["available_at"]=live["interval_end"]
    live["historical_complete"]=True
    live["is_complete"]=True
    live.loc[live.index[-1],"historical_complete"]=False
    live.loc[live.index[-1],"is_complete"]=False
    live.attrs["availability_mode"]="event_time"
    live.attrs["market_source"]="live_mt5"

    enriched=add_technical_features(live)
    enriched=add_multitimeframe_features(enriched,base="1h")
    enriched=add_multitimeframe_regime(enriched)
    for column in FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column]=1.0
    enriched["liquidity_sweep"]=0.0

    assert pd.isna(enriched["mtf_decision_time"].iloc[-1])
    assert not feature_row_eligibility_mask(enriched).iloc[-1]


def test_decision_time_covers_latest_available_evidence_source():
    index=pd.date_range("2026-01-01T00:00:00Z",periods=5,freq="1h")
    frame=pd.DataFrame(1.0,index=index,columns=FEATURE_COLUMNS)
    frame["available_at"]=index+pd.Timedelta(hours=1)
    frame["interval_end"]=index+pd.Timedelta(hours=1)
    frame["mtf_decision_time"]=index+pd.Timedelta(hours=3)
    frame["regime_score_available_at"]=index+pd.Timedelta(hours=2)
    frame["regime_bias_available_at"]=index+pd.Timedelta(hours=1)
    frame["liquidity_sweep"]=[0.0,1.0,0.0,0.0,0.0]
    frame["liquidity_confirmation_available_at"]=pd.Series(
        [pd.NaT,index[1]+pd.Timedelta(hours=3),*([pd.NaT]*3)],
        index=index,
        dtype=object,
    )
    frame.attrs["availability_mode"]="event_time"

    decisions=feature_row_decision_times(frame)

    assert decisions.loc[index[1]]==index[1]+pd.Timedelta(hours=3)
    assert feature_row_eligibility_mask(frame).loc[index[1]]