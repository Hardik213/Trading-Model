from __future__ import annotations

from typing import Any


class PaperTrader:
    """Disabled compatibility shim for the superseded probabilistic runner."""

    def __init__(self, config_path: str | None = None) -> None:
        # Keep the old constructor shape for callers while avoiding config,
        # market-data, broker, and execution side effects.
        self.config_path = config_path
        self.broker = None
        self.order_manager = None

    def build_dashboard_snapshot(self) -> dict[str, Any]:
        return {
            "symbol": "XAUUSD",
            "status": "DISABLED",
            "market_source": "unavailable",
            "reasons": [
                "The legacy probabilistic dashboard is disabled. The Core v1 research runner is not configured as the application entry point."
            ],
            "passed": False,
        }

    def run(self) -> dict[str, Any]:
        raise RuntimeError(
            "The legacy probabilistic runner is disabled. Use the governed Core v1 research pipeline after its run configuration is frozen."
        )
