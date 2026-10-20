from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import pytest

import src.core_v1_mss as mss_module
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
    detect_core_v1_liquidity_levels,
    evaluate_core_v1_liquidity,
)
from src.core_v1_mss import (
    CoreV1MSSConfirmation,
    CoreV1MSSResult,
    CoreV1StructuralBreak,
    CoreV1StructuralPoint,
    MSSStatus,
    _validate_component_subject,
    _validate_liquidity,
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
        "level-left-2": source("level-left-2", "2026-01-01T09:59:00Z"),
        "point": source("point", "2026-01-01T10:00:00Z"),
        "level-left-1": source("level-left-1", "2026-01-01T10:00:15Z"),
        "liquidity-level": source("liquidity-level", "2026-01-01T10:00:30Z"),
        "level-right-1": source("level-right-1", "2026-01-01T10:00:40Z"),
        "level-right-2": source("level-right-2", "2026-01-01T10:00:50Z"),
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
    bars = (
        (102.0, 104.0, 101.0, 103.0),
        (103.0, 105.0, 100.0, 104.0),
        (104.0, 105.5, 101.0, 105.0),
        (104.0, 104.0, 95.0, 100.0),
        (103.0, 105.0, 99.0, 104.0),
        (102.0, 104.0, 98.0, 103.0),
        (102.0, 107.0, 94.0, 106.0),
        (103.0, 108.0, 96.0, 107.0),
        (104.0, 109.0, 95.0, 108.0),
        (105.0, 110.0, 94.0, 109.0),
        (106.0, 111.0, 93.0, 110.0),
        (107.0, 112.0, 92.0, 111.0),
    )
    observations = tuple(
        SubjectVisibleObservation(
            value={
                "source_id": item.source_id,
                "open": bar[0],
                "high": bar[1],
                "low": bar[2],
                "close": bar[3],
            },
            timing=timing(
                item.event_timestamp.isoformat(),
                (item,),
                available=item.availability_timestamp.isoformat(),
                base_scoped=True,
            ),
            scope=ObservationScope.BASE,
        )
        for item, bar in zip(sources.values(), bars)
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
        point_confirmation_source = sources["level-right-2"]
        point = CoreV1StructuralPoint(
            subject=inputs.subject,
            point_id="governed-point-1",
            kind=SwingType.HIGH,
            price=105.0,
            timing=timing(
                point_time,
                (point_source, point_confirmation_source),
                confirmation=point_confirmation_source.event_timestamp.isoformat(),
                base_scoped=True,
            ),
        )

    liquidity = None
    liquidity_breach = None
    if liquidity_present:
        level_source = sources["liquidity-level"]
        breach_source = sources["liquidity-breach"]
        reaction_source = sources["liquidity-confirmation"]
        (level,) = tuple(
            candidate
            for candidate in detect_core_v1_liquidity_levels(inputs)
            if candidate.timing.event_timestamp == level_source.event_timestamp
            and candidate.side is LiquiditySide.SSL
        )
        if liquidity_resolved and liquidity_time == "2026-01-01T10:01:00Z":
            (liquidity,) = evaluate_core_v1_liquidity(
                inputs,
                levels=(level,),
            )
            liquidity_breach = liquidity.breach
        else:
            liquidity_breach = CoreV1LiquidityBreach(
                subject=inputs.subject,
                level=level,
                price=(94.0 if level.side is LiquiditySide.SSL else 107.0),
                timing=timing(
                    liquidity_time,
                    (*level.timing.source_observations, breach_source),
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
                        (
                            *liquidity_breach.timing.source_observations,
                            reaction_source,
                        ),
                        confirmation=reaction_source.event_timestamp.isoformat(),
                        provenance={"resolver": "governed fixture"},
                    ),
                    price=106.0,
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
            break_price=_visible_price_at(inputs, ts(break_time), "close"),
            direction=Direction.BULLISH,
            timing=timing(
                break_time,
                (*point.timing.source_observations, break_source),
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


def _visible_price_at(
    inputs: SubjectVisibleInputs,
    timestamp: pd.Timestamp,
    field: str,
) -> float:
    (observation,) = tuple(
        observation
        for observation in inputs.base_observations
        if observation.timing.event_timestamp == timestamp
    )
    return float(observation.value[field])


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


def test_mss_rejects_liquidity_level_not_recomputed_from_visible_bars():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    accepted = values["liquidity_event"]
    fabricated_level = CoreV1LiquidityLevel(
        subject=inputs.subject,
        level_id="fabricated-level",
        side=accepted.level.side,
        price=105.0,
        timing=accepted.level.timing,
    )
    breach = replace(accepted.breach, level=fabricated_level)
    reaction = replace(accepted.reaction, breach=breach, price=105.0)
    values["liquidity_event"] = replace(
        accepted,
        level=fabricated_level,
        breach=breach,
        reaction=reaction,
        sweep=SweepKind.UNRESOLVED,
        sweep_timing=None,
        sweep_provenance={},
    )

    with pytest.raises(ValueError, match="recomputed from visible base observations"):
        evaluate(inputs, values)


def test_mss_rejects_forged_breach_on_genuine_visible_swing():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    canonical = values["liquidity_event"]
    assert canonical.level in detect_core_v1_liquidity_levels(inputs)
    assert canonical.breach.price == _visible_price_at(
        inputs,
        canonical.breach.timing.event_timestamp,
        "low" if canonical.level.side is LiquiditySide.SSL else "high",
    )
    other_visible_breach_price = next(
        float(
            observation.value["low"]
            if canonical.level.side is LiquiditySide.SSL
            else observation.value["high"]
        )
        for observation in inputs.base_observations
        if observation.timing.event_timestamp
        != canonical.breach.timing.event_timestamp
        and float(
            observation.value["low"]
            if canonical.level.side is LiquiditySide.SSL
            else observation.value["high"]
        ) != canonical.breach.price
        and (
            float(observation.value["low"]) < canonical.level.price
            if canonical.level.side is LiquiditySide.SSL
            else float(observation.value["high"]) > canonical.level.price
        )
    )
    forged_breach = replace(canonical.breach, price=other_visible_breach_price)
    forged = replace(
        canonical,
        breach=forged_breach,
        reaction=replace(canonical.reaction, breach=forged_breach),
    )
    values["liquidity_event"] = forged

    with pytest.raises(ValueError, match="full assessment recomputed"):
        evaluate(inputs, values)


def test_mss_rejects_forged_reaction_on_genuine_visible_swing():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    canonical = values["liquidity_event"]
    confirmation_price = _visible_price_at(
        inputs,
        canonical.reaction.timing.confirmation_timestamp,
        "high" if canonical.level.side is LiquiditySide.SSL else "low",
    )
    assert canonical.reaction.price == confirmation_price
    other_visible_low = next(
        float(observation.value["low"])
        for observation in inputs.base_observations
        if observation.timing.event_timestamp
        != canonical.reaction.timing.confirmation_timestamp
        and (
            float(observation.value["high"]) >= canonical.level.price
            if canonical.level.side is LiquiditySide.SSL
            else float(observation.value["low"]) <= canonical.level.price
        )
    )
    forged = replace(
        canonical,
        reaction=replace(canonical.reaction, price=other_visible_low),
    )
    values["liquidity_event"] = forged

    with pytest.raises(ValueError, match="full assessment recomputed"):
        evaluate(inputs, values)


def test_mss_rejects_acceptance_reaction(monkeypatch):
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    canonical = values["liquidity_event"]
    acceptance = replace(
        canonical,
        reaction=replace(canonical.reaction, kind=ReactionKind.ACCEPTANCE),
        sweep=SweepKind.UNRESOLVED,
        sweep_timing=None,
        sweep_provenance={},
    )
    monkeypatch.setattr(
        mss_module,
        "evaluate_core_v1_liquidity",
        lambda inputs, *, levels: (acceptance,),
    )

    with pytest.raises(ValueError, match="ACCEPTANCE reactions"):
        _validate_liquidity(acceptance, inputs)


def test_mss_rejects_forged_not_sweep_classification():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    values["liquidity_event"] = replace(
        values["liquidity_event"],
        sweep=SweepKind.NOT_SWEEP,
    )

    with pytest.raises(ValueError, match="full assessment recomputed"):
        evaluate(inputs, values)


def test_mss_is_unresolved_when_displacement_direction_conflicts_with_swept_side():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    displacement = values["displacement"]
    values["displacement"] = replace(
        displacement,
        observation=replace(
            displacement.observation,
            direction=Direction.BEARISH,
        ),
    )

    result = evaluate(inputs, values)

    assert result.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert "conflicts with displacement direction" in result.reason


def test_mss_is_unresolved_when_structural_point_type_conflicts_with_direction():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    values["structural_point"] = replace(
        values["structural_point"],
        kind=SwingType.LOW,
    )

    result = evaluate(inputs, values)

    assert result.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert "Structural point type conflicts" in result.reason


def test_mss_is_unresolved_when_break_direction_conflicts_with_displacement():
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    values["structural_break"] = replace(
        values["structural_break"],
        direction=Direction.BEARISH,
    )

    result = evaluate(inputs, values)

    assert result.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert "Structural break direction conflicts" in result.reason


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
        == ts("2026-01-01T10:00:50Z")
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
            "|provenance must identify|Breach event must occur strictly after level availability"
        ),
    ):
        values = components(inputs, sources, **override)
        evaluate(inputs, values)


@pytest.mark.parametrize(
    ("point_source_name", "point_available", "point_confirmation"),
    [
        ("liquidity-breach", "2026-01-01T10:01:00Z", "2026-01-01T10:01:00Z"),
        (
            "liquidity-confirmation",
            "2026-01-01T10:01:30Z",
            "2026-01-01T10:01:30Z",
        ),
        ("point", "2026-01-01T10:01:00Z", "2026-01-01T10:00:00Z"),
        ("point", "2026-01-01T10:00:00Z", None),
    ],
)
def test_structural_point_must_be_confirmed_and_available_before_liquidity_breach(
    point_source_name,
    point_available,
    point_confirmation,
):
    inputs, sources = visible_inputs()
    values = components(inputs, sources)
    point_source = sources[point_source_name]
    point_time = point_source.event_timestamp.isoformat()
    point_sources = (point_source,)
    if point_source.availability_timestamp != ts(point_available):
        delayed_source = replace(
            source(
                "delayed-point-availability",
                point_time,
                point_available,
            ),
            base_bar_timestamp=None,
            is_base_timeframe=False,
        )
        context_observation = SubjectVisibleObservation(
            value={"source_id": delayed_source.source_id},
            timing=timing(
                point_time,
                (delayed_source,),
                available=point_available,
                base_scoped=False,
            ),
            scope=ObservationScope.CONTEXT,
        )
        inputs = replace(
            inputs,
            context_observations=(*inputs.context_observations, context_observation),
        )
        point_sources = (point_source, delayed_source)
    point = CoreV1StructuralPoint(
        subject=inputs.subject,
        point_id="post-breach-or-unconfirmed-point",
        kind=SwingType.HIGH,
        price=105.0,
        timing=timing(
            point_time,
            point_sources,
            confirmation=point_confirmation,
            available=point_available,
            base_scoped=True,
        ),
    )

    with pytest.raises(ValueError, match="strictly before the liquidity breach"):
        evaluate_core_v1_mss(
            inputs,
            structural_point=point,
            liquidity_event=values["liquidity_event"],
            displacement=values["displacement"],
        )


def test_structural_point_confirmation_requires_its_own_visible_source():
    inputs, sources = visible_inputs()

    with pytest.raises(ValueError, match="identify its confirmation observation"):
        CoreV1StructuralPoint(
            subject=inputs.subject,
            point_id="unsupported-pivot-confirmation",
            kind=SwingType.HIGH,
            price=105.0,
            timing=timing(
                "2026-01-01T10:00:00Z",
                (sources["point"],),
                confirmation="2026-01-01T10:00:50Z",
                available="2026-01-01T10:00:50Z",
                base_scoped=True,
            ),
        )


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


def visible_auto_mss_inputs(bar_count: int = 32) -> SubjectVisibleInputs:
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(bar_count)]
    bars[3] = (100.0, 110.0, 99.0, 100.0)
    bars[5] = (100.0, 101.0, 97.0, 100.0)
    bars[8] = (100.0, 111.0, 99.0, 100.0)
    bars[9] = (100.0, 101.0, 99.0, 100.0)
    bars[28] = (100.0, 100.5, 94.0, 95.5)
    bars[29] = (95.5, 96.0, 93.0, 94.0)
    if bar_count > 30:
        bars[30] = (94.0, 94.4, 92.8, 93.2)
    if bar_count > 31:
        bars[31] = (93.2, 93.8, 92.5, 92.8)

    observations = []
    start = ts("2026-01-01T10:00:00Z")
    for index, bar in enumerate(bars):
        timestamp = start + pd.Timedelta(minutes=index)
        item = source(
            f"auto-mss-bar-{index}",
            timestamp.isoformat(),
            (timestamp + pd.Timedelta(seconds=1)).isoformat(),
        )
        observations.append(
            SubjectVisibleObservation(
                value={
                    "open": bar[0],
                    "high": bar[1],
                    "low": bar[2],
                    "close": bar[3],
                    "source_id": item.source_id,
                },
                timing=timing(
                    timestamp.isoformat(),
                    (item,),
                    available=item.availability_timestamp.isoformat(),
                    base_scoped=True,
                ),
                scope=ObservationScope.BASE,
            )
        )
    last = observations[-1].timing.source_observations[0]
    current = ReplaySubject(
        availability_timestamp=last.availability_timestamp,
        base_bar_timestamp=last.event_timestamp,
    )
    return SubjectVisibleInputs(
        subject=current,
        availability_timestamp=current.availability_timestamp,
        base_observations=tuple(observations),
        context_observations=(),
    )


def test_default_pipeline_derives_point_in_time_mss_from_visible_price_action():
    from src.core_v1_pipeline import evaluate_core_v1_pipeline

    inputs = visible_auto_mss_inputs()
    result = evaluate_core_v1_pipeline(inputs)

    assert result.liquidity is not None
    assert result.displacement is not None
    assert result.displacement.status is DisplacementStatus.DISPLACEMENT
    assert result.mss is not None
    assert result.mss.status is MSSStatus.MSS
    assert result.mss.structural_point.kind is SwingType.LOW
    assert result.mss.structural_point.price == 97.0
    assert result.mss.structural_point.timing.event_timestamp == ts(
        "2026-01-01T10:05:00Z"
    )
    assert result.mss.structural_break.timing.event_timestamp == ts(
        "2026-01-01T10:29:00Z"
    )
    assert result.mss.confirmation_timestamp == ts("2026-01-01T10:31:00Z")
    assert result.zone is not None
    assert result.retracement is not None
    assert result.retracement.planned_entry is None
    assert result.hypothesis is None


def test_visible_mss_follow_through_after_subject_cutoff_stays_unresolved():
    from src.core_v1_pipeline import evaluate_core_v1_pipeline

    earlier = evaluate_core_v1_pipeline(visible_auto_mss_inputs(bar_count=31))
    later = evaluate_core_v1_pipeline(visible_auto_mss_inputs(bar_count=32))

    assert earlier.mss is not None
    assert earlier.mss.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert "follow-through is unresolved" in earlier.mss.reason
    assert later.mss is not None
    assert later.mss.status is MSSStatus.MSS
    assert earlier.subject.base_bar_timestamp < later.mss.confirmation_timestamp


def test_visible_mss_requires_sweep_classification_not_just_rejection():
    from src.core_v1_pipeline import evaluate_core_v1_pipeline

    inputs = visible_auto_mss_inputs()
    pipeline = evaluate_core_v1_pipeline(inputs)
    assert pipeline.liquidity is not None
    assert pipeline.displacement is not None
    unresolved_liquidity = replace(
        pipeline.liquidity,
        sweep=SweepKind.UNRESOLVED,
        sweep_timing=None,
        sweep_provenance={},
    )

    result = evaluate_core_v1_mss(
        inputs,
        liquidity_event=unresolved_liquidity,
        displacement=pipeline.displacement,
        derive_visible=True,
    )

    assert result.status is MSSStatus.NOT_MSS_UNRESOLVED
    assert result.structural_point is None
    assert result.structural_break is None



def test_mss_pivot_window_must_be_available_before_liquidity_breach():
    from src.core_v1_pipeline import evaluate_core_v1_pipeline

    inputs = visible_auto_mss_inputs()
    observations = list(inputs.base_observations)
    confirmation_bar = observations[7]
    old_source = confirmation_bar.timing.source_observations[0]
    late_source = replace(
        old_source,
        availability_timestamp=ts("2026-01-01T10:08:01Z"),
    )
    late_timing = replace(
        confirmation_bar.timing,
        availability_timestamp=late_source.availability_timestamp,
        source_observations=(late_source,),
    )
    observations[7] = replace(confirmation_bar, timing=late_timing)
    delayed_inputs = SubjectVisibleInputs(
        subject=inputs.subject,
        availability_timestamp=inputs.subject.availability_timestamp,
        base_observations=tuple(observations),
        context_observations=inputs.context_observations,
    )

    result = evaluate_core_v1_pipeline(delayed_inputs)

    assert result.displacement is not None
    assert result.displacement.status is DisplacementStatus.DISPLACEMENT
    assert result.mss is not None
    assert result.mss.status is MSSStatus.NOT_MSS_UNRESOLVED
