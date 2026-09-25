"""Fail-closed execution guard for research and paper workflows."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ExecutionPolicy:
    mode: str = "research"
    allow_live: bool = False
    allow_demo: bool = False

    def authorize(self, *, destination: str) -> None:
        destination = destination.lower()
        if self.mode in {"research", "backtest"}:
            raise PermissionError("Execution disabled in research/backtest mode")
        if destination == "live" and not self.allow_live:
            raise PermissionError("Live execution is not explicitly enabled")
        if destination == "demo" and not self.allow_demo:
            raise PermissionError("Demo execution is not explicitly enabled")
        if destination not in {"live", "demo"}:
            raise ValueError(f"Unknown execution destination: {destination}")
