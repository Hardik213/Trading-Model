# XAUUSD Core v1 Research System

This repository is developing a point-in-time-correct implementation of the ICT 2022 Core v1 setup pipeline for XAUUSD. It records visible evidence and provenance through liquidity, displacement, MSS, retracement, invalidation, and opposing-liquidity target stages. A setup with missing, conflicting, late, or unsupported evidence must remain unresolved.

## Current status

- The Core v1 research pipeline and tick-execution integration are under development.
- The first-touch FVG close-back-out entry rule is experimental and has not been evaluated or frozen.
- The default application launcher and dashboard are disabled because the older probabilistic bot trains on outcome labels, can summarize historical results, and does not implement the governed Core v1 pipeline.
- No backtest or strategy evaluation is authorized until the frozen-build gates in `docs/evaluation_protocol.md` are satisfied. Keep the 2025 holdout sealed.
- Broker sizing and live execution remain unavailable until the actual Octa account and XAUUSD contract specifications are confirmed and configured.

## Repository map

- `src/core_v1_*.py` implements the evidence-gated setup pipeline.
- `src/core_v1_tick_execution.py` connects visible replay decisions to quote-based execution simulation.
- `src/tick_execution.py` contains execution records and explicit execution configuration.
- `docs/core_v1_*_contract.md` records current development contracts and unresolved rules.
- `docs/evaluation_protocol.md` governs model freeze and any future evaluation.
- `configs/core_v1_development.json` records current development assumptions and unresolved freeze blockers; it is not a frozen run configuration.
- `scripts/check_core_v1_freeze.py` reports configuration hash and blockers without loading market data or running a replay.
- `docs/strategy_variant_log.csv` is the append-only development and experiment record.
- `data/raw/` is local input data and is not committed to Git.

## Verification

Use focused synthetic Core v1 tests while implementation work continues. Do not run market replays, outcome calculations, performance reports, or charts before the evaluation protocol gates pass. The broad verification script runs the full repository suite, which includes historical census and outcome-oriented checks; it is not a safe substitute for focused synthetic tests during this development stage.

The Core v1 tick runner is a library integration and does not yet have a frozen, user-facing run configuration. `python main.py` exits with a disabled-runner message; `python portal.py` serves a disabled status snapshot without reading market data or writing a result file.
