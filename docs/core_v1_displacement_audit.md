# Core v1 displacement contract audit

**Audit date:** 2026-10-20  
**Branch:** `wip/step3d-recovered`  
**Scope:** Synthetic contract verification only. No raw CSV rows, strategy outcomes, replay, backtest, performance report, chart, or holdout material was inspected.

## Requirements-to-source-and-test matrix

| Requirement | Source evidence | Synthetic test evidence | Status |
| --- | --- | --- | --- |
| Use a unique visible sweep with confirmed rejection as prerequisite. | `_derive_visible_displacement()` filters to exactly one `SweepKind.SWEEP`, rejection reaction, and sweep timing; `evaluate_core_v1_displacement()` recomputes supplied liquidity context. | `test_visible_price_action_derives_displacement_after_confirmed_sweep`, `test_ambiguous_sweep_context_does_not_select_displacement`, `test_liquidity_context_is_optional_and_not_a_displacement_trigger`, `test_displacement_rejects_acceptance_reaction`. | Verified |
| Direction follows the rejected liquidity side. | `_derive_visible_displacement()` maps `LiquidityEventDirection` to governed `Direction`; recomputed liquidity assessment supplies the authoritative event direction. | `test_visible_price_action_derives_displacement_after_confirmed_sweep`, `test_displacement_rejects_acceptance_reaction`, plus pipeline/MSS direction-conflict coverage. | Verified |
| Event occurs strictly after reaction confirmation. | Candidate loop in `_derive_visible_displacement()` skips timestamps `<= reaction_confirmation`. | `test_visible_price_action_derives_displacement_after_confirmed_sweep` asserts event is after sweep confirmation; pipeline causal-order tests cover downstream enforcement. | Verified |
| OHLC evidence is visible, ordered, finite, geometrically valid, and base-timeframe sourced one bar at a time. | `_derive_visible_displacement()` validates mapping payloads, one source, source/event/base-bar timestamp equality, finite OHLC, geometry, and strictly increasing timestamps. | `test_unrestricted_historical_frame_is_rejected`, `test_displacement_provenance_must_reference_visible_sources`, `test_future_observation_is_rejected_at_subject_input_boundary`, `test_base_bar_cutoff_is_distinct_from_availability_cutoff`. | Verified |
| Subject identity, availability, and base-bar cutoffs are preserved. | `CoreV1DisplacementObservation`, `CoreV1DisplacementAssessment`, and evaluator validate `ReplaySubject`, timing, source lineage, and result availability. | `test_replay_subject_identity_is_preserved`, `test_availability_cutoff_rejects_late_displacement`, `test_future_confirmation_is_rejected`, `test_future_sentinel_cannot_alter_earlier_subject_result`. | Verified |
| Confirmation must be available by the subject cutoff. | `is_confirmed_as_of(event, inputs.subject.base_bar_timestamp)` gates automatic candidates; `EvidenceTiming`/subject validation gates availability. | `test_future_follow_through_does_not_confirm_earlier_subject`, `test_displacement_event_confirmation_and_availability_remain_distinct`, `test_availability_cutoff_rejects_late_displacement`. | Verified |
| Select the latest eligible confirmed candidate deterministically, with confirmation time then event time tie-breaking. | Candidate tuples are selected by `max((confirmation_timestamp, event_timestamp))` in `_derive_visible_displacement()`. | `test_latest_confirmed_candidate_is_selected_as_of_subject_cutoff`, `test_future_follow_through_does_not_confirm_earlier_subject`. | Verified |
| Provenance retains liquidity sweep and detector evidence. | Automatic derivation merges sweep timing sources with detector event-through-confirmation sources; supplied observations must reference visible sources. | `test_provenance_survives_the_displacement_result`, `test_displacement_provenance_must_reference_visible_sources`, `test_visible_price_action_derives_displacement_after_confirmed_sweep`. | Verified |
| Unresolved displacement blocks later pipeline stages. | `evaluate_core_v1_pipeline()` returns an incomplete/developing result before MSS when displacement is unresolved; no later FVG, retracement, invalidation, target, or hypothesis gate is advanced as valid. | `test_pipeline_derives_displacement_from_visible_price_action_by_default`, `test_missing_or_unresolved_prerequisite_blocks_every_later_stage`, `test_public_default_path_derives_displacement_without_custom_builder`. | Verified |

## Findings

No reproducible displacement contract violation was found. The committed implementation and focused synthetic coverage satisfy the audited requirements. No production code or strategy rule was changed in this audit.

## Status distinction

- **Implementation present:** yes, in `src/core_v1_displacement.py` and its canonical pipeline integration.
- **Implementation verified against contract:** yes, by the source review and focused tests listed above.
- **Owner acceptance of a development rule:** separate from this audit; MSS retains its prior owner status, while displacement is recorded here as technically verified.
- **Frozen evaluation build:** no. The development configuration remains unfrozen and uncommitted.
- **Strategy performance evidence:** none. No replay, backtest, outcome, metric, chart, or holdout result was produced.
