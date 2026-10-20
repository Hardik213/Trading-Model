from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from typing import Mapping, Protocol

import pandas as pd

from .core_v1_dealing_range import DealingRangeContext, DealingRangeStatus
from .core_v1_displacement import CoreV1DisplacementAssessment, DisplacementStatus
from .core_v1_evidence import SubjectVisibleInputs
from .core_v1_mss import CoreV1MSSResult, MSSStatus
from .evidence_timing import EvidenceSourceObservation, EvidenceTiming
from .displacement import Direction
from .fvg import FVG, FVGDirection, detect_fvgs
from .pd_arrays import PDArray, PDArrayType
from .replay_subject import ReplaySubject, same_subject_id


class ZoneStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class CoreV1FVGObservation:
    """A timed FVG zone observation; it does not imply an entry decision."""

    subject: ReplaySubject
    fvg: FVG
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not isinstance(self.fvg, FVG):
            raise TypeError("fvg must be an existing FVG value.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("FVG observations require EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        if self.timing.event_timestamp != _utc(self.fvg.formation_timestamp):
            raise ValueError("FVG event time must equal its formation timestamp.")
        if self.timing.confirmation_timestamp is not None and (
            self.timing.confirmation_timestamp
            != _utc(self.fvg.confirmation_timestamp)
        ):
            raise ValueError("FVG confirmation timing must match the FVG value.")
        if (
            self.timing.confirmation_timestamp is None
            and _utc(self.fvg.confirmation_timestamp) > self.timing.event_timestamp
        ):
            raise ValueError("FVG timing must preserve its later confirmation timestamp.")
        _validate_bounds(
            self.fvg.lower_bound,
            self.fvg.upper_bound,
            self.fvg.midpoint,
            self.fvg.size,
        )
        if not isinstance(self.fvg.direction, FVGDirection):
            raise TypeError("FVG direction must be an existing FVGDirection.")
        if _utc(self.fvg.first_candle_timestamp) > self.timing.event_timestamp:
            raise ValueError("FVG first candle cannot follow its formation event.")
        if _utc(self.fvg.third_candle_timestamp) != self.timing.event_timestamp:
            raise ValueError("FVG third candle timestamp must equal formation time.")
        if _utc(self.fvg.confirmation_timestamp) < self.timing.event_timestamp:
            raise ValueError("FVG confirmation cannot precede formation.")
        _validate_base_sources(
            self.timing,
            self.subject,
            "FVG",
            confirmation_timestamp=_utc(self.fvg.confirmation_timestamp),
        )
        if (
            self.timing.confirmation_timestamp is not None
            and self.timing.confirmation_timestamp > self.subject.base_bar_timestamp
        ):
            raise ValueError("FVG confirmation exceeds the ReplaySubject base-bar cutoff.")
        if (
            _utc(self.fvg.confirmation_timestamp) > self.timing.event_timestamp
            and _utc(self.fvg.confirmation_timestamp)
            not in {source.event_timestamp for source in self.timing.source_observations}
        ):
            raise ValueError("FVG confirmation requires a timed source observation.")
        _require_event_sources(
            self.timing.source_observations,
            {
                _utc(self.fvg.first_candle_timestamp),
                _utc(self.fvg.third_candle_timestamp),
            },
            "FVG",
        )
        first_timestamp = _utc(self.fvg.first_candle_timestamp)
        third_timestamp = _utc(self.fvg.third_candle_timestamp)
        formation_sources = tuple(
            source
            for source in self.timing.source_observations
            if first_timestamp <= source.event_timestamp <= third_timestamp
        )
        if len(formation_sources) != 3 or len(
            {source.event_timestamp for source in formation_sources}
        ) != 3:
            raise ValueError(
                "FVG provenance must retain exactly one base source for each of its three candles."
            )
        if any(
            not source.is_base_timeframe
            or source.base_bar_timestamp != source.event_timestamp
            for source in formation_sources
        ):
            raise ValueError("FVG candle provenance must identify visible base bars.")


@dataclass(frozen=True)
class CoreV1PDArrayObservation:
    """A contextual PD-array; currently limited to repository-supported FVG."""

    subject: ReplaySubject
    array: PDArray
    timing: EvidenceTiming
    fvg_observation: CoreV1FVGObservation

    def __post_init__(self) -> None:
        if not same_subject_id(self.subject, self.fvg_observation.subject):
            raise ValueError("PD-array and FVG must preserve the same ReplaySubject.")
        if not isinstance(self.array, PDArray):
            raise TypeError("array must be an existing PDArray.")
        if self.array.array_type is not PDArrayType.FVG:
            raise ValueError("Unsupported PD-array type.")
        if self.array.direction is not self.fvg_observation.fvg.direction:
            raise ValueError("PD-array direction must match its FVG.")
        if (
            self.array.lower_bound != self.fvg_observation.fvg.lower_bound
            or self.array.upper_bound != self.fvg_observation.fvg.upper_bound
        ):
            raise ValueError("PD-array bounds must match its FVG.")
        if self.array.midpoint != self.fvg_observation.fvg.midpoint:
            raise ValueError("PD-array midpoint must match its FVG.")
        if _utc(self.array.source_timestamp) != _utc(
            self.fvg_observation.fvg.confirmation_timestamp
        ):
            raise ValueError("PD-array source timestamp must preserve FVG confirmation.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("PD-array observations require EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        _validate_base_sources(
            self.timing,
            self.subject,
            "PD-array",
            confirmation_timestamp=self.timing.confirmation_timestamp,
        )
        _require_sources(self.timing.source_observations,
                         self.fvg_observation.timing.source_observations,
                         "PD-array")
        if self.timing.event_timestamp != self.fvg_observation.timing.event_timestamp:
            raise ValueError("PD-array must preserve the FVG event timestamp.")
        if self.timing.availability_timestamp < self.fvg_observation.timing.availability_timestamp:
            raise ValueError("PD-array availability cannot precede its FVG.")


@dataclass(frozen=True)
class CoreV1OrderBlockObservation:
    """Opaque caller-governed OB zone; no OB identification semantics are implied."""

    subject: ReplaySubject
    block_id: str
    lower_bound: float
    upper_bound: float
    timing: EvidenceTiming

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ReplaySubject):
            raise TypeError("subject must be a ReplaySubject.")
        if not self.block_id:
            raise ValueError("block_id must not be empty.")
        if not isinstance(self.timing, EvidenceTiming):
            raise TypeError("Order-block observations require EvidenceTiming.")
        self.timing.validate_for_subject(self.subject)
        _validate_bounds(self.lower_bound, self.upper_bound)
        _validate_base_sources(self.timing, self.subject, "Order block")


@dataclass(frozen=True)
class FVGResult:
    subject: ReplaySubject
    status: ZoneStatus
    observation: CoreV1FVGObservation | None = None
    reason: str = ""


@dataclass(frozen=True)
class PDArrayResult:
    subject: ReplaySubject
    status: ZoneStatus
    observation: CoreV1PDArrayObservation | None = None
    reason: str = ""


@dataclass(frozen=True)
class OrderBlockResult:
    subject: ReplaySubject
    status: ZoneStatus
    observation: CoreV1OrderBlockObservation | None = None
    reason: str = ""


class FVGResolver(Protocol):
    """Optional governed FVG classifier; no algorithm is provided here."""

    def resolve(self, inputs: SubjectVisibleInputs) -> CoreV1FVGObservation | None: ...


class OrderBlockResolver(Protocol):
    """Optional governed OB resolver; no canonical OB definition currently exists."""

    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> CoreV1OrderBlockObservation | None: ...


class UnavailableFVGResolver:
    def resolve(self, inputs: SubjectVisibleInputs) -> CoreV1FVGObservation | None:
        if not isinstance(inputs, SubjectVisibleInputs):
            raise TypeError("inputs must be SubjectVisibleInputs.")
        return None


class UnavailableOrderBlockResolver:
    def resolve(
        self,
        inputs: SubjectVisibleInputs,
    ) -> CoreV1OrderBlockObservation | None:
        if not isinstance(inputs, SubjectVisibleInputs):
            raise TypeError("inputs must be SubjectVisibleInputs.")
        return None


def evaluate_core_v1_fvg(
    inputs: SubjectVisibleInputs,
    *,
    observation: CoreV1FVGObservation | None = None,
    resolver: FVGResolver | None = None,
    displacement: CoreV1DisplacementAssessment | None = None,
    mss: CoreV1MSSResult | None = None,
    derive_visible: bool = False,
) -> FVGResult:
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    if observation is not None and resolver is not None:
        raise ValueError("Supply an FVG observation or resolver, not both.")
    if type(derive_visible) is not bool:
        raise TypeError("derive_visible must be a boolean.")
    if derive_visible and (observation is not None or resolver is not None):
        raise ValueError("Visible FVG derivation cannot be mixed with supplied evidence.")
    candidate = observation
    if candidate is None and derive_visible:
        candidate = _derive_visible_displacement_fvg(inputs, displacement, mss)
    elif candidate is None:
        candidate = (resolver or UnavailableFVGResolver()).resolve(inputs)
    if candidate is None:
        return FVGResult(
            inputs.subject,
            ZoneStatus.UNRESOLVED,
            reason="No governed FVG observation is available.",
        )
    if not isinstance(candidate, CoreV1FVGObservation):
        raise TypeError("FVG resolver must return CoreV1FVGObservation or None.")
    if not same_subject_id(candidate.subject, inputs.subject):
        raise ValueError("FVG observation belongs to another ReplaySubject.")
    candidate.timing.validate_for_subject(inputs.subject)
    _validate_visible_lineage(candidate.timing, inputs, "FVG")
    _validate_base_sources(
        candidate.timing,
        inputs.subject,
        "FVG",
        confirmation_timestamp=candidate.timing.confirmation_timestamp,
    )
    return FVGResult(inputs.subject, ZoneStatus.RESOLVED, candidate)



def _derive_visible_displacement_fvg(
    inputs: SubjectVisibleInputs,
    displacement: CoreV1DisplacementAssessment | None,
    mss: CoreV1MSSResult | None,
) -> CoreV1FVGObservation | None:
    if (
        displacement is None
        or displacement.status is not DisplacementStatus.DISPLACEMENT
        or displacement.observation is None
        or not same_subject_id(displacement.subject, inputs.subject)
        or mss is None
        or mss.status is not MSSStatus.MSS
        or not same_subject_id(mss.subject, inputs.subject)
        or mss.displacement is not displacement
        or displacement.observation.direction not in {Direction.BULLISH, Direction.BEARISH}
    ):
        return None

    bars: list[dict[str, float]] = []
    sources: list[EvidenceSourceObservation] = []
    for visible in inputs.base_observations:
        if not isinstance(visible.value, Mapping):
            return None
        if len(visible.timing.source_observations) != 1:
            return None
        source = visible.timing.source_observations[0]
        if (
            not source.is_base_timeframe
            or source.event_timestamp != visible.timing.event_timestamp
            or source.base_bar_timestamp != visible.timing.event_timestamp
        ):
            return None
        values = {str(key).lower(): value for key, value in visible.value.items()}
        if not {"open", "high", "low", "close"}.issubset(values):
            return None
        try:
            bar = {
                name: float(values[name])
                for name in ("open", "high", "low", "close")
            }
        except (TypeError, ValueError, OverflowError):
            return None
        if any(not isfinite(value) for value in bar.values()):
            return None
        if (
            bar["high"] < max(bar["open"], bar["close"], bar["low"])
            or bar["low"] > min(bar["open"], bar["close"], bar["high"])
        ):
            return None
        bars.append(bar)
        sources.append(source)

    timestamps = [source.event_timestamp for source in sources]
    if len(timestamps) < 3 or any(
        right <= left for left, right in zip(timestamps, timestamps[1:])
    ):
        return None
    frame = pd.DataFrame(
        bars,
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    frame.columns = [column.title() for column in frame.columns]
    try:
        found = detect_fvgs(frame)
    except (TypeError, ValueError, KeyError, IndexError):
        return None

    expected_direction = (
        FVGDirection.BULLISH
        if displacement.observation.direction is Direction.BULLISH
        else FVGDirection.BEARISH
    )
    displacement_timestamp = displacement.observation.timing.event_timestamp
    candidates = []
    for fvg in found:
        if fvg.direction is not expected_direction:
            continue
        first_position = frame.index.get_loc(fvg.first_candle_timestamp)
        if frame.index[first_position + 1] != displacement_timestamp:
            continue
        if fvg.confirmation_timestamp > inputs.subject.base_bar_timestamp:
            continue
        candidates.append((fvg, first_position))
    if len(candidates) != 1:
        return None

    fvg, first_position = candidates[0]
    fvg_sources = tuple(sources[first_position : first_position + 3])
    timing = EvidenceTiming.from_inputs(
        event_timestamp=fvg.formation_timestamp,
        confirmation_timestamp=fvg.confirmation_timestamp,
        source_observations=fvg_sources,
        provenance={
            "resolver": "core-v1-visible-fvg",
            "detector": "src.fvg.detect_fvgs",
            "middle_candle_is_displacement": True,
        },
        base_bar_scoped=True,
    )
    return CoreV1FVGObservation(
        subject=inputs.subject,
        fvg=fvg,
        timing=timing,
    )


def evaluate_core_v1_pd_array(
    inputs: SubjectVisibleInputs,
    fvg_result: FVGResult,
    *,
    dealing_range: DealingRangeContext | None = None,
) -> PDArrayResult:
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    if not isinstance(fvg_result, FVGResult) or not same_subject_id(
        fvg_result.subject,
        inputs.subject,
    ):
        raise ValueError("PD-array input must be the FVG result for this ReplaySubject.")
    if fvg_result.status is not ZoneStatus.RESOLVED or fvg_result.observation is None:
        return PDArrayResult(
            inputs.subject,
            ZoneStatus.UNRESOLVED,
            reason="No supported FVG exists for PD-array representation.",
        )
    fvg_observation = fvg_result.observation
    fvg = fvg_observation.fvg
    location = None
    if dealing_range is not None:
        if not same_subject_id(dealing_range.subject, inputs.subject):
            raise ValueError("Dealing range belongs to another ReplaySubject.")
        visible = _visible_sources(inputs)
        if any(source not in visible for source in dealing_range.provenance):
            raise ValueError(
                "Dealing-range provenance references observations outside visible inputs."
            )
        if dealing_range.status is DealingRangeStatus.RESOLVED:
            location = dealing_range.dealing_range.location(fvg.midpoint)
    array = PDArray(
        array_type=PDArrayType.FVG,
        direction=fvg.direction,
        lower_bound=fvg.lower_bound,
        upper_bound=fvg.upper_bound,
        midpoint=fvg.midpoint,
        source_timestamp=fvg.confirmation_timestamp,
        range_location=location,
    )
    observation = CoreV1PDArrayObservation(
        subject=inputs.subject,
        array=array,
        timing=fvg_observation.timing,
        fvg_observation=fvg_observation,
    )
    return PDArrayResult(inputs.subject, ZoneStatus.RESOLVED, observation)


def evaluate_core_v1_order_block(
    inputs: SubjectVisibleInputs,
    *,
    observation: CoreV1OrderBlockObservation | None = None,
    resolver: OrderBlockResolver | None = None,
) -> OrderBlockResult:
    if not isinstance(inputs, SubjectVisibleInputs):
        raise TypeError("inputs must be SubjectVisibleInputs.")
    if observation is not None and resolver is not None:
        raise ValueError("Supply an order-block observation or resolver, not both.")
    candidate = observation or (
        resolver or UnavailableOrderBlockResolver()
    ).resolve(inputs)
    if candidate is None:
        return OrderBlockResult(
            inputs.subject,
            ZoneStatus.UNRESOLVED,
            reason="No governed order-block observation is available.",
        )
    if not isinstance(candidate, CoreV1OrderBlockObservation):
        raise TypeError(
            "Order-block resolver must return CoreV1OrderBlockObservation or None."
        )
    if not same_subject_id(candidate.subject, inputs.subject):
        raise ValueError("Order-block observation belongs to another ReplaySubject.")
    candidate.timing.validate_for_subject(inputs.subject)
    _validate_visible_lineage(candidate.timing, inputs, "Order block")
    _validate_base_sources(candidate.timing, inputs.subject, "Order block")
    return OrderBlockResult(inputs.subject, ZoneStatus.RESOLVED, candidate)


def _validate_bounds(
    lower: float,
    upper: float,
    midpoint: float | None = None,
    size: float | None = None,
) -> None:
    lower = float(lower)
    upper = float(upper)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("Zone bounds must be finite and lower_bound < upper_bound.")
    if midpoint is not None:
        midpoint = float(midpoint)
        if not isfinite(midpoint) or not lower <= midpoint <= upper:
            raise ValueError("Zone midpoint must be finite and within its bounds.")
        if not isclose(midpoint, (lower + upper) / 2, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("FVG midpoint must equal the bounds midpoint.")
    if size is not None:
        size = float(size)
        if not isfinite(size) or size <= 0 or not isclose(
            size,
            upper - lower,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("FVG size must equal the positive distance between bounds.")


def _validate_base_sources(
    timing: EvidenceTiming,
    subject: ReplaySubject,
    name: str,
    *,
    confirmation_timestamp: object | None = None,
) -> None:
    if timing.base_bar_scoped:
        if timing.event_timestamp > subject.base_bar_timestamp:
            raise ValueError(f"{name} event exceeds the ReplaySubject base-bar cutoff.")
        if (
            confirmation_timestamp is not None
            and _utc(confirmation_timestamp) > subject.base_bar_timestamp
        ):
            raise ValueError(
                f"{name} confirmation exceeds the ReplaySubject base-bar cutoff."
            )
    base_sources = tuple(
        source for source in timing.source_observations if source.is_base_timeframe
    )
    if any(
        source.event_timestamp > subject.base_bar_timestamp
        or source.base_bar_timestamp > subject.base_bar_timestamp
        for source in base_sources
    ):
        raise ValueError(f"{name} source exceeds the ReplaySubject base-bar cutoff.")


def _validate_visible_lineage(
    timing: EvidenceTiming,
    inputs: SubjectVisibleInputs,
    name: str,
) -> None:
    visible = _visible_sources(inputs)
    if any(source not in visible for source in timing.source_observations):
        raise ValueError(f"{name} provenance references observations outside visible inputs.")


def _visible_sources(inputs: SubjectVisibleInputs) -> tuple[EvidenceSourceObservation, ...]:
    return tuple(
        source
        for item in inputs.base_observations + inputs.context_observations
        for source in item.timing.source_observations
    )


def _require_event_sources(
    sources: tuple[EvidenceSourceObservation, ...],
    event_timestamps: set,
    name: str,
) -> None:
    actual = {source.event_timestamp for source in sources}
    if not event_timestamps.issubset(actual):
        raise ValueError(f"{name} provenance must include its formation source observations.")


def _require_sources(
    actual: tuple[EvidenceSourceObservation, ...],
    required: tuple[EvidenceSourceObservation, ...],
    name: str,
) -> None:
    if any(source not in actual for source in required):
        raise ValueError(f"{name} provenance must retain its underlying FVG sources.")


def _utc(value: object):
    import pandas as pd

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("FVG timestamps must be timezone-aware.")
    return timestamp.tz_convert("UTC")


__all__ = [
    "CoreV1FVGObservation",
    "CoreV1OrderBlockObservation",
    "CoreV1PDArrayObservation",
    "FVGResolver",
    "FVGResult",
    "OrderBlockResolver",
    "OrderBlockResult",
    "PDArrayResult",
    "UnavailableFVGResolver",
    "UnavailableOrderBlockResolver",
    "ZoneStatus",
    "evaluate_core_v1_fvg",
    "evaluate_core_v1_order_block",
    "evaluate_core_v1_pd_array",
]
