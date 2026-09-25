from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd
from .displacement import Direction, DisplacementEvent
from .market_structure import SwingType

def normalize_direction(value):
    if value is None:
        return None
    if isinstance(value, Direction):
        return value
    text=str(value).upper().split(".")[-1]
    return {
        "LONG": Direction.BULLISH,
        "BULLISH": Direction.BULLISH,
        "SHORT": Direction.BEARISH,
        "BEARISH": Direction.BEARISH,
    }[text]

@dataclass(frozen=True)
class MSSEvent:
    direction: str
    timestamp: pd.Timestamp
    broken_level: float
    displacement: Optional[DisplacementEvent]
    follow_through: bool
    confirmation_timestamp: Optional[pd.Timestamp]=None
    structural_point: Optional[object]=None

def _candidate_structure(df, break_position):
    if break_position<=0 or break_position>=len(df): return None
    prior=df.iloc[:break_position]
    if len(prior)<2: return None
    h=prior["High"] if "High" in prior.columns else prior["high"]
    l=prior["Low"] if "Low" in prior.columns else prior["low"]
    h,l=h.astype(float),l.astype(float)
    if float(h.iloc[-1])>float(h.iloc[-2]) and float(l.iloc[-1])>float(l.iloc[-2]):
        return "BULLISH",float(l.iloc[-1])
    if float(h.iloc[-1])<float(h.iloc[-2]) and float(l.iloc[-1])<float(l.iloc[-2]):
        return "BEARISH",float(h.iloc[-1])
    return None

def _resolve_structural_point(swings,direction:str,as_of):
    if not swings:return None
    target=SwingType.HIGH if direction=="LONG" else SwingType.LOW
    candidates=[s for s in swings if getattr(s,"kind",None)==target and pd.Timestamp(s.confirmation_timestamp)<=pd.Timestamp(as_of)]
    if not candidates:return None
    return max(candidates,key=lambda s:s.price) if direction=="LONG" else min(candidates,key=lambda s:s.price)

def _combined_confirmation(*timestamps):
    values=[pd.Timestamp(x) for x in timestamps if x is not None]
    return max(values) if values else None

def detect_mss(df,break_position:Optional[int]=None,*,swings:Optional[Sequence[object]]=None,
               liquidity_event=None,displacement:Optional[DisplacementEvent]=None,
               follow_through_bars:int=2):
    if break_position is None: break_position=len(df)-1
    if df is None or len(df)==0 or break_position<0 or break_position>=len(df): return None
    ts=pd.Timestamp(df.index[break_position])
    if displacement is not None and getattr(displacement,"confirmation_timestamp",None) is not None:
        if pd.Timestamp(displacement.confirmation_timestamp) > pd.Timestamp(df.index[-1]):
            return None
    candidate=_candidate_structure(df,break_position)

    if candidate is None:
        if liquidity_event is None and displacement is None:return None
        if displacement is None:return None
        direction=normalize_direction(displacement.direction)
        if direction in {Direction.BULLISH,"LONG","BULLISH"}:
            event_direction="LONG"; expected_side="SSL"; structural_price=displacement.end_price
        else:
            event_direction="SHORT"; expected_side="BSL"; structural_price=displacement.end_price
        if liquidity_event is not None:
            if str(liquidity_event.outcome)!="BreachOutcome.REJECTION":return None
            if getattr(liquidity_event.level,"side",None) is None:return None
            if str(liquidity_event.level.side).endswith(expected_side) is False:return None

        own_confirmation=ts
        if follow_through_bars>0:
            future=df.iloc[break_position+1:min(len(df),break_position+1+follow_through_bars)]
            own_confirmation=None
            for idx,row in future.iterrows():
                high=float(row["High"]) if "High" in row.index else float(row["high"])
                low=float(row["Low"]) if "Low" in row.index else float(row["low"])
                if event_direction=="LONG" and high>structural_price:
                    own_confirmation=pd.Timestamp(idx); break
                if event_direction=="SHORT" and low<structural_price:
                    own_confirmation=pd.Timestamp(idx); break

        confirmation=_combined_confirmation(
            own_confirmation,
            getattr(displacement,"confirmation_timestamp",None),
        )
        structural_point=_resolve_structural_point(swings,event_direction,ts)
        return MSSEvent(
            direction=event_direction,timestamp=ts,broken_level=float(structural_price),
            displacement=displacement,follow_through=confirmation is not None,
            confirmation_timestamp=confirmation,structural_point=structural_point,
        )

    prior_structure,level=candidate
    close=float(df.iloc[break_position]["Close"]) if "Close" in df.columns else float(df.iloc[break_position]["close"])
    direction="SHORT" if prior_structure=="BULLISH" else "LONG"
    if not (close<level if direction=="SHORT" else close>level): return None

    if displacement is not None and not getattr(displacement,"follow_through",False):
        return None

    if liquidity_event is not None:
        if str(liquidity_event.outcome)!="BreachOutcome.REJECTION":return None
        expected_side="SSL" if direction=="LONG" else "BSL"
        if getattr(liquidity_event.level,"side",None) is None:return None
        if str(liquidity_event.level.side).endswith(expected_side) is False:return None

    own_confirmation=ts
    if follow_through_bars>0:
        future=df.iloc[break_position+1:min(len(df),break_position+1+follow_through_bars)]
        own_confirmation=None
        for idx,row in future.iterrows():
            high=float(row["High"]) if "High" in row.index else float(row["high"])
            low=float(row["Low"]) if "Low" in row.index else float(row["low"])
            if direction=="LONG" and high>level:
                own_confirmation=pd.Timestamp(idx); break
            if direction=="SHORT" and low<level:
                own_confirmation=pd.Timestamp(idx); break

    confirmation=_combined_confirmation(
        own_confirmation,
        getattr(displacement,"confirmation_timestamp",None),
    )
    return MSSEvent(
        direction=direction,timestamp=ts,broken_level=float(level),
        displacement=displacement,follow_through=confirmation is not None,
        confirmation_timestamp=confirmation,
        structural_point=_resolve_structural_point(swings,direction,ts),
    )

def is_confirmed_as_of(event,as_of):
    return (event is not None and event.follow_through and
            event.confirmation_timestamp is not None and
            pd.Timestamp(event.confirmation_timestamp)<=pd.Timestamp(as_of))
