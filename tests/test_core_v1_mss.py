from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import pytest

from src.core_v1_displacement import (
    CoreV1DisplacementAssessment,
    CoreV1DisplacementObservation,
    DisplacementStatus,
)
from src.core_v1_evidence import (
    ObservationScope,
    SubjectVisibleInputs,
    SubjectVisibleObservation,
)
from src.core_v1_liquidity import (
    CoreV1LiquidityAssessment,
    CoreV1LiquidityBreach,
    CoreV1LiquidityLevel,
    CoreV1LiquidityReaction,
    ReactionKind,
    SweepKind,
)
from src.core_v1_mss import (
    CoreV1MSSConfirmation,
    CoreV1MSSResult,
    CoreV1StructuralBreak,
    CoreV1StructuralPoint,
    MSSStatus,
    _validate_component_subject,
    evaluate_core_v1_mss,
)
from src.displacement import Direction
from src.evidence_timing import EvidenceSourceObservation, EvidenceTiming
from src.market_structure import LiquiditySide, SwingType
from src.replay_subject import ReplaySubject


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def subject() -> ReplaySubject:
    return ReplaySubject(
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        base_bar_timestamp=ts("2026-01-01T10:05:00Z"),
    )


def source(
    source_id: str,
    event: str,
    available: str | None = None,
) -> EvidenceSourceObservation:
    available = available or event
    return EvidenceSourceObservation(
        source_id=source_id,
        source="synthetic",
        event_timestamp=ts(event),
        availability_timestamp=ts(available),
        provenance={"source_id": source_id},
        base_bar_timestamp=ts(event),
        is_base_timeframe=True,
    )


def timing(
    event: str,
    observations: tuple[EvidenceSourceObservation, ...],
    *,
    confirmation: str | None = None,
    available: str | None = None,
    base_scoped: bool = False,
    provenance: dict[str, object] | None = None,
) -> EvidenceTiming:
    return EvidenceTiming(
        event_timestamp=ts(event),
        confirmation_timestamp=ts(confirmation) if confirmation else None,
        availability_timestamp=ts(available) if available else max(
            item.availability_timestamp for item in observations
        ),
        source_observations=observations,
        provenance=provenance or {"test": "governed input"},
        base_bar_scoped=base_scoped,
    )


def visible_inputs() -> tuple[
    SubjectVisibleInputs,
    dict[str, EvidenceSourceObservation],
]:
    current = subject()
    sources = {
        "point": source("point", "2026-01-01T10:00:00Z"),
        "liquidity-level": source("liquidity-level", "2026-01-01T10:00:30Z"),
        "liquidity-breach": source("liquidity-breach", "2026-01-01T10:01:00Z"),
        "liquidity-confirmation": source(
            "liquidity-confirmation",
            "2026-01-01T10:01:30Z",
        ),
        "displacement": source("displacement", "2026-01-01T10:02:00Z"),
        "displacement-confirmation": source(
            "displacement-confirmation",
            "2026-01-01T10:02:30Z",
        ),
        "break": source("break", "2026-01-01T10:03:00Z"),
        "follow-through": source(
            "follow-through",
            "2026-01-01T10:05:00Z",
            "2026-01-01T10:06:17Z",
        ),
    }
    observations = tuple(
        SubjectVisibleObservation(
            value={"source_id": item.source_id},
            timing=timing(
                item.event_timestamp.isoformat(),
                (item,),
                available=item.availability_timestamp.isoformat(),
                base_scoped=True,
            ),
            scope=ObservationScope.BASE,
        )
        for item in sources.values()
    )
    inputs = SubjectVisibleInputs(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=observations,
        context_observations=(),
    )
    return inputs, sources


def components(
    inputs: SubjectVisibleInputs,
    sources: dict[str, EvidenceSourceObservation],
    *,
    point_present: bool = True,
    liquidity_present: bool = True,
    liquidity_resolved: bool = True,
    displacement_present: bool = True,
    displacement_resolved: bool = True,
    break_present: bool = True,
    confirmation_present: bool = True,
    confirmation_resolved: bool = True,
    point_time: str = "2026-01-01T10:00:00Z",
    liquidity_time: str = "2026-01-01T10:01:00Z",
    displacement_time: str = "2026-01-01T10:02:00Z",
    break_time: str = "2026-01-01T10:03:00Z",
    confirmation_time: str = "2026-01-01T10:05:00Z",
) -> dict[str, object]:
    point = None
    if point_present:
        point_source = sources["point"]
        point = CoreV1StructuralPoint(
            subject=inputs.subject,
            point_id="governed-point-1",
            kind=SwingType.HIGH,
            price=105.0,
            timing=timing(point_time, (point_source,), base_scoped=True),
        )

    liquidity = None
    liquidity_breach = None
    if liquidity_present:
        level_source = sources["liquidity-level"]
        breach_source = sources["liquidity-breach"]
        reaction_source = sources["liquidity-confirmation"]
        level = CoreV1LiquidityLevel(
            subject=inputs.subject,
            level_id="liquidity-1",
            side=LiquiditySide.BSL,
            price=106.0,
            timing=timing(
                level_source.event_timestamp.isoformat(),
                (level_source,),
            ),
        )
        liquidity_breach = CoreV1LiquidityBreach(
            subject=inputs.subject,
            level=level,
            price=107.0,
            timing=timing(
                liquidity_time,
                (level_source, breach_source),
                base_scoped=True,
            ),
        )
        reaction = None
        if liquidity_resolved:
            reaction = CoreV1LiquidityReaction(
                subject=inputs.subject,
                breach=liquidity_breach,
                kind=ReactionKind.REJECTION,
                timing=timing(
                    liquidity_time,
                    (level_source, breach_source, reaction_source),
                    confirmation=reaction_source.event_timestamp.isoformat(),
                    provenance={"resolver": "governed fixture"},
                ),
            )
        liquidity = CoreV1LiquidityAssessment(
            subject=inputs.subject,
            level=level,
            breach=liquidity_breach,
            reaction=reaction,
            sweep=SweepKind.UNRESOLVED,
        )

    displacement = None
    if displacement_present:
        event_source = sources["displacement"]
        follow_source = sources["displacement-confirmation"]
        displacement_observation = CoreV1DisplacementObservation(
            subject=inputs.subject,
            direction=Direction.BULLISH,
            timing=timing(
                displacement_time,
                (event_source, follow_source),
                confirmation=follow_source.event_timestamp.isoformat(),
                base_scoped=True,
            ),
        )
        displacement = CoreV1DisplacementAssessment(
            subject=inputs.subject,
            status=(
                DisplacementStatus.DISPLACEMENT
                if displacement_resolved
                else DisplacementStatus.UNRESOLVED
            ),
            availability_timestamp=inputs.subject.availability_timestamp,
            observation=(
                displacement_observation if displacement_resolved else None
            ),
            reason="synthetic",
        )

    structural_break = None
    if break_present and point is not None:
        break_source = sources["break"]
        structural_break = CoreV1StructuralBreak(
            subject=inputs.subject,
            structural_point=point,
            break_price=108.0,
            direction=Direction.BULLISH,
            timing=timing(
                break_time,
                (sources["point"], break_source),
                base_scoped=True,
            ),
        )

    confirmation = None
    if confirmation_present and structural_break is not None:
        follow_source = sources["follow-through"]
        confirmation_observations = (
            *structural_break.timing.source_observations,
            follow_source,
        )
        if confirmation_resolved:
            confirmation = CoreV1MSSConfirmation(
                subject=inputs.subject,
                structural_break=structural_break,
                timing=timing(
                    break_time,
                    confirmation_observations,
                    confirmation=confirmation_time,
                    available=inputs.subject.availability_timestamp.isoformat(),
                    base_scoped=True,
                ),
            )

    return {
        "structural_point": point,
        "liquidity_event": liquidity,
        "liquidity_breach": liquidity_breach,
        "displacement": displacement,
        "structural_break": structural_break,
        "confirmation": confirmation,
    }


def evaluate(
    inputs: SubjectVisibleInputs,
    values: dict[str, object],
) -> CoreV1MSSResult:
    return evaluate_core_v1_mss(
        inputs,
        structural_point=values["structural_point"],
        liquidity_event=values["liquidity_event"],
        displacement=values["displacement"],
        structural_break=values["structural_break"],
        confirmation=values["confirmation"],
    )


def test_valid_mss_requires_all_governed_components():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    result = evaluate(inputs, values)
    assert result.status is MSSStatus.MSS
    assert result.subject is inputs.subject
    assert result.structural_point is values["structural_point"]
    assert result.liquidity_event is values["liquidity_event"]
    assert result.displacement is values["displacement"]
    assert result.structural_break is values["structural_break"]
    assert result.confirmation is values["confirmation"]


def test_mss_component_subject_validation_uses_subject_id():
    canonical = subject()
    reconstructed = ReplaySubject.from_dict(canonical.to_dict())
    assert reconstructed is not canonical
    assert reconstructed.subject_id == canonical.subject_id

    _validate_component_subject(reconstructed, canonical, "MSS test")

    different = ReplaySubject(
        canonical.availability_timestamp,
        canonical.base_bar_timestamp - pd.Timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="ReplaySubject"):
        _validate_component_subject(different, canonical, "MSS test")


def test_reconstructed_equal_subject_passes_full_mss_validation_path():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    reconstructed = ReplaySubject.from_dict(inputs.subject.to_dict())
    point = values["structural_point"]
    structural_break = values["structural_break"]
    confirmation = values["confirmation"]

    rebound_point = replace(point, subject=reconstructed)
    rebound_break = replace(
        structural_break,
        subject=reconstructed,
        structural_point=rebound_point,
    )
    rebound_confirmation = replace(
        confirmation,
        subject=reconstructed,
        structural_break=rebound_break,
    )
    rebound_liquidity = replace(values["liquidity_event"], subject=reconstructed)
    rebound_displacement = replace(values["displacement"], subject=reconstructed)
    values.update(
        structural_point=rebound_point,
        liquidity_event=rebound_liquidity,
        displacement=rebound_displacement,
        structural_break=rebound_break,
        confirmation=rebound_confirmation,
    )

    result = evaluate(inputs, values)
    assert result.status is MSSStatus.MSS
    assert result.subject.subject_id == inputs.subject.subject_id
    assert result.structural_point.subject is reconstructed


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"point_present": False}, "structural point"),
        ({"liquidity_present": False}, "liquidity event"),
        ({"liquidity_resolved": False}, "liquidity event"),
        ({"displacement_present": False}, "displacement"),
        ({"displacement_resolved": False}, "displacement"),
        ({"confirmation_present": False}, "follow-through"),
        ({"confirmation_resolved": False}, "follow-through"),
    ],
)
def test_missing_or_unresolved_mandatory_component_fails_closed(override, reason):
    inputs, sources = visible_inputs()
    values = components(inputs, sources, **override)
    result = evaluate(inputs, values)
    assert result.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert reason in result.reason.lower()
    assert result.structural_point is None
    assert result.liquidity_event is None
    assert result.displacement is None
    assert result.structural_break is None
    assert result.confirmation is None


def test_structural_point_event_and_confirmation_timing_are_preserved():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    result = evaluate(inputs, values)
    assert result.structural_point_timestamp == ts("2026-01-01T10:00:00Z")
    assert (
        result.structural_point.timing.availability_timestamp
        == ts("2026-01-01T10:00:00Z")
    )


def test_liquidity_event_timestamp_is_preserved():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    assert result.liquidity_event_timestamp == ts("2026-01-01T10:01:00Z")
    assert (
        result.liquidity_event.reaction.timing.confirmation_timestamp
        == ts("2026-01-01T10:01:30Z")
    )


def test_displacement_timestamp_is_preserved():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    assert result.displacement_event_timestamp == ts("2026-01-01T10:02:00Z")
    assert (
        result.displacement.observation.timing.confirmation_timestamp
        == ts("2026-01-01T10:02:30Z")
    )


def test_mss_confirmation_timestamp_remains_separate_from_event_and_availability():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    assert result.confirmation_timestamp == ts("2026-01-01T10:05:00Z")
    assert result.confirmation_timestamp != result.structural_point_timestamp
    assert result.confirmation_timestamp != result.liquidity_event_timestamp
    assert result.confirmation_timestamp != result.displacement_event_timestamp
    assert result.availability_timestamp == inputs.subject.availability_timestamp


def test_component_availability_after_subject_is_rejected():
    inputs, sources = visible_inputs()
    late_source = EvidenceSourceObservation(
        source_id="late-point",
        source="synthetic",
        event_timestamp=ts("2026-01-01T10:00:00Z"),
        availability_timestamp=ts("2026-01-01T10:06:20Z"),
        provenance={"source_id": "late-point"},
        base_bar_timestamp=ts("2026-01-01T10:00:00Z"),
        is_base_timeframe=True,
    )
    with pytest.raises(ValueError, match="unavailable"):
        late_point = CoreV1StructuralPoint(
            subject=inputs.subject,
            point_id="late-point",
            kind=SwingType.HIGH,
            price=105.0,
            timing=timing(
                "2026-01-01T10:00:00Z",
                (late_source,),
                available="2026-01-01T10:06:20Z",
                base_scoped=True,
            ),
        )
        values = components(inputs, sources)
        evaluate_core_v1_mss(
            inputs,
            structural_point=late_point,
            liquidity_event=values["liquidity_event"],
            displacement=values["displacement"],
            structural_break=values["structural_break"],
            confirmation=values["confirmation"],
        )


def test_future_structural_point_is_rejected():
    inputs, _ = visible_inputs()
    future = source("future-point", "2026-01-01T10:06:00Z")
    with pytest.raises(ValueError, match="base-bar"):
        CoreV1StructuralPoint(
            inputs.subject,
            "future",
            SwingType.HIGH,
            110.0,
            timing(future.event_timestamp.isoformat(), (future,), base_scoped=True),
        )


def test_future_liquidity_is_rejected():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    future_liquidity = values["liquidity_event"]
    future_source = source("future-liquidity", "2026-01-01T10:07:00Z")
    level = future_liquidity.level
    with pytest.raises(ValueError, match="unavailable"):
        future_breach = CoreV1LiquidityBreach(
            subject=inputs.subject,
            level=level,
            price=108.0,
            timing=timing(
                "2026-01-01T10:07:00Z",
                (*level.timing.source_observations, future_source),
                base_scoped=True,
            ),
        )
        evaluate_core_v1_mss(
            inputs,
            structural_point=values["structural_point"],
            liquidity_event=CoreV1LiquidityAssessment(
                subject=inputs.subject,
                level=level,
                breach=future_breach,
            ),
            displacement=values["displacement"],
            structural_break=values["structural_break"],
            confirmation=values["confirmation"],
        )


def test_future_displacement_is_rejected():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    future = source("future-displacement", "2026-01-01T10:06:00Z")
    with pytest.raises(ValueError, match="base-bar"):
        future_observation = CoreV1DisplacementObservation(
            subject=inputs.subject,
            direction=Direction.BULLISH,
            timing=timing(
                future.event_timestamp.isoformat(),
                (future,),
                base_scoped=True,
            ),
        )
        future_displacement = CoreV1DisplacementAssessment(
            subject=inputs.subject,
            status=DisplacementStatus.DISPLACEMENT,
            availability_timestamp=inputs.subject.availability_timestamp,
            observation=future_observation,
        )
        evaluate_core_v1_mss(
            inputs,
            structural_point=values["structural_point"],
            liquidity_event=values["liquidity_event"],
            displacement=future_displacement,
            structural_break=values["structural_break"],
            confirmation=values["confirmation"],
        )


def test_future_confirmation_is_rejected():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    structural_break = values["structural_break"]
    future_source = EvidenceSourceObservation(
        source_id="future-follow-through",
        source="synthetic",
        event_timestamp=ts("2026-01-01T10:06:00Z"),
        availability_timestamp=ts("2026-01-01T10:06:17Z"),
        provenance={"source_id": "future-follow-through"},
        base_bar_timestamp=ts("2026-01-01T10:06:00Z"),
        is_base_timeframe=True,
    )
    with pytest.raises(ValueError, match="base-bar"):
        future_timing = timing(
            structural_break.timing.event_timestamp.isoformat(),
            (*structural_break.timing.source_observations, future_source),
            confirmation="2026-01-01T10:06:00Z",
            available=inputs.subject.availability_timestamp.isoformat(),
            base_scoped=True,
        )
        CoreV1MSSConfirmation(
            inputs.subject,
            structural_break,
            future_timing,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"point_time": "2026-01-01T10:02:00Z"}, "structural point"),
        ({"liquidity_time": "2026-01-01T10:00:00Z"}, "liquidity event"),
        ({"displacement_time": "2026-01-01T10:00:30Z"}, "displacement"),
        ({"break_time": "2026-01-01T10:01:30Z"}, "structural break"),
        ({"confirmation_time": "2026-01-01T10:02:00Z"}, "confirmation"),
    ],
)
def test_causal_ordering_violation_is_rejected(override, message):
    inputs, sources = visible_inputs()
    with pytest.raises(
        ValueError,
        match=(
            "Causal MSS ordering violation|cannot precede|availability cannot precede"
            "|provenance must identify"
        ),
    ):
        values = components(inputs, sources, **override)
        evaluate(inputs, values)


def test_replay_subject_identity_is_preserved_exactly():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    assert result.subject is inputs.subject
    assert result.structural_point.subject is inputs.subject
    assert result.liquidity_event.subject is inputs.subject
    assert result.displacement.subject is inputs.subject


def test_provenance_for_each_required_component_is_preserved():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    assert set(result.provenance) == {
        "structural_point",
        "liquidity_event",
        "displacement",
        "structural_break",
        "confirmation",
    }
    assert sources["point"] in result.provenance["structural_point"]
    assert sources["liquidity-breach"] in result.provenance["liquidity_event"]
    assert sources["displacement"] in result.provenance["displacement"]
    assert sources["break"] in result.provenance["structural_break"]
    assert sources["follow-through"] in result.provenance["confirmation"]


def test_future_sentinel_cannot_change_earlier_mss_result():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    sentinel = EvidenceSourceObservation(
        source_id="future-sentinel",
        source="synthetic",
        event_timestamp=ts("2026-01-01T10:07:00Z"),
        availability_timestamp=ts("2026-01-01T10:07:17Z"),
        provenance={"sentinel": "must not enter"},
        base_bar_timestamp=ts("2026-01-01T10:07:00Z"),
        is_base_timeframe=True,
    )
    future_observation = SubjectVisibleObservation(
        value={"sentinel": True},
        timing=timing(
            sentinel.event_timestamp.isoformat(),
            (sentinel,),
            base_scoped=False,
        ),
        scope=ObservationScope.CONTEXT,
    )
    with pytest.raises(ValueError, match="unavailable"):
        SubjectVisibleInputs(
            subject=inputs.subject,
            availability_timestamp=inputs.subject.availability_timestamp,
            base_observations=inputs.base_observations,
            context_observations=(future_observation,),
        )
    assert evaluate(inputs, components(inputs, sources)) == result


def test_unrestricted_historical_frame_is_rejected():
    inputs, sources = visible_inputs()
    with pytest.raises(TypeError, match="not frames"):
        SubjectVisibleObservation(
            value=pd.DataFrame({"High": [105.0, 110.0]}),
            timing=timing(
                "2026-01-01T10:00:00Z",
                (sources["point"],),
                base_scoped=True,
            ),
            scope=ObservationScope.BASE,
        )


def test_mss_does_not_create_trade_or_execution_output():
    inputs, sources = visible_inputs()
    result = evaluate(inputs, components(inputs, sources))
    assert not hasattr(result, "entry")
    assert not hasattr(result, "trade")
    assert not hasattr(result, "fill")
    assert not hasattr(result, "risk")


def test_price_break_alone_is_not_mss():
    inputs, sources = visible_inputs()
    values = components(
        inputs,
        sources,
        liquidity_present=False,
        displacement_present=False,
        confirmation_present=False,
    )
    values["structural_break"] = components(inputs, sources)["structural_break"]
    result = evaluate(inputs, values)
    assert result.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert result.confirmation is None
    assert result.liquidity_event is None
    assert result.displacement is None


def test_unresolved_result_cannot_fabricate_partial_mss():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    with pytest.raises(ValueError, match="partial MSS"):
        CoreV1MSSResult(
            subject=inputs.subject,
            status=MSSStatus.NOT_MSS_UNRESOLVED,
            availability_timestamp=inputs.subject.availability_timestamp,
            structural_point=values["structural_point"],
        )
