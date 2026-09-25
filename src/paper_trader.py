from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.backtest import run_backtest
from src.broker import BrokerFactory
from src.data_loader import load_market_data
from src.features import FEATURE_COLUMNS, add_ict_features, add_technical_features, build_training_frame
from src.liquidity import add_liquidity_features
from src.macro import add_macro_features
from src.models import predict_probabilities, train_signal_model
from src.multitimeframe import add_multitimeframe_features
from src.order_manager import OrderManager
from src.regime import add_multitimeframe_regime
from src.session import add_session_features
from src.session_entry import add_session_entry_filter


class PaperTrader:
    @staticmethod
    def _resolve_env_value(value):
        if isinstance(value, dict):
            return {k: PaperTrader._resolve_env_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [PaperTrader._resolve_env_value(v) for v in value]
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            if expanded.startswith("${") and expanded.endswith("}"):
                return None
            return expanded
        return value

    def __init__(self, config_path: str | None = None):
        root = Path(__file__).resolve().parents[1]
        config_file = Path(config_path) if config_path else root / "config.yaml"
        self.config = self._resolve_env_value(yaml.safe_load(config_file.read_text(encoding="utf-8")))
        market_cfg = self.config["market"]
        self.broker = BrokerFactory.create(self.config.get("broker", {}), market_cfg["symbol"])
        self.order_manager = OrderManager(max_positions=self.config["risk"].get("max_positions", 2), max_risk_per_trade=self.config["risk"].get("risk_per_trade", 0.01))
        self.broker.connect() if hasattr(self.broker, "connect") else None

    def _evaluate_ict_gate(self, row: pd.Series, probability: float) -> Dict[str, Any]:
        model_cfg = self.config["model"]
        regime_bias = float(row.get("regime_bias", 0.0))
        macro_bias = float(row.get("macro_bias", 0.0))
        session_bias = float(row.get("session_bias", 0.0))
        liquidity_bias = float(row.get("liquidity_sweep", 0.0))
        trend_bias = 1.0 if float(row["sma_fast"]) >= float(row["sma_slow"]) else -1.0
        is_entry_session = bool(row.get("is_entry_session", True))

        htf_high = float(row.get("htf_high_24", 0.0))
        htf_low = float(row.get("htf_low_24", 0.0))
        price = float(row.get("Close", 0.0))
        atr = float(row.get("atr", 1.0))
        displacement = float(row.get("displacement", 0.0))
        rr_ratio = float(row.get("rr_ratio", 0.0))

        has_meaningful_htf_objective = abs(price - htf_high) <= max(atr * 1.5, 1e-6) or abs(price - htf_low) <= max(atr * 1.5, 1e-6)
        long_ok = (
            is_entry_session
            and probability >= model_cfg["probability_threshold"]
            and (regime_bias >= 0 or trend_bias >= 0)
            and macro_bias >= 0
            and session_bias >= 0
            and has_meaningful_htf_objective
            and bool(row.get("liquidity_raid_long", False))
            and bool(row.get("mss_bull", False))
            and bool(row.get("pd_array_long", False))
            and bool(row.get("structure_valid_long", False))
            and displacement >= 1.2
            and rr_ratio >= 1.5
        )
        short_ok = (
            is_entry_session
            and probability <= (1.0 - model_cfg["probability_threshold"])
            and (regime_bias <= 0 or trend_bias <= 0)
            and macro_bias <= 0
            and session_bias <= 0
            and has_meaningful_htf_objective
            and bool(row.get("liquidity_raid_short", False))
            and bool(row.get("mss_bear", False))
            and bool(row.get("pd_array_short", False))
            and bool(row.get("structure_valid_short", False))
            and displacement >= 1.2
            and rr_ratio >= 1.5
        )

        if long_ok and liquidity_bias in (-1.0, 0.0, 1.0):
            signal = 1
        elif short_ok and liquidity_bias in (-1.0, 0.0, 1.0):
            signal = -1
        else:
            signal = 0

        return {
            "signal": signal,
            "probability": probability,
            "threshold": model_cfg["probability_threshold"],
            "entry_session": is_entry_session,
            "trend_bias": trend_bias,
            "regime_bias": regime_bias,
            "macro_bias": macro_bias,
            "session_bias": session_bias,
            "liquidity_bias": liquidity_bias,
            "htf_high": htf_high,
            "htf_low": htf_low,
            "price": price,
            "atr": atr,
            "displacement": displacement,
            "rr_ratio": rr_ratio,
            "has_meaningful_htf_objective": has_meaningful_htf_objective,
            "liquidity_raid_long": bool(row.get("liquidity_raid_long", False)),
            "liquidity_raid_short": bool(row.get("liquidity_raid_short", False)),
            "mss_bull": bool(row.get("mss_bull", False)),
            "mss_bear": bool(row.get("mss_bear", False)),
            "pd_array_long": bool(row.get("pd_array_long", False)),
            "pd_array_short": bool(row.get("pd_array_short", False)),
            "structure_valid_long": bool(row.get("structure_valid_long", False)),
            "structure_valid_short": bool(row.get("structure_valid_short", False)),
            "session": row.get("session", ""),
        }

    def _compose_signal(self, row: pd.Series, probability: float) -> int:
        return self._evaluate_ict_gate(row, probability)["signal"]

    def build_dashboard_snapshot(self) -> Dict[str, Any]:
        market_cfg = self.config["market"]
        model_cfg = self.config["model"]

        try:
            df = load_market_data(
                symbol=market_cfg["symbol"],
                start_date=market_cfg["start_date"],
                end_date=market_cfg["end_date"],
                use_live_data=self.config["trading"].get("use_live_data", False),
                allow_synthetic_fallback=self.config["trading"].get("allow_synthetic_fallback", False),
            )
        except RuntimeError as exc:
            return {
                "symbol": market_cfg["symbol"],
                "status": "ERROR",
                "signal": 0,
                "probability": 0.0,
                "threshold": float(model_cfg["probability_threshold"]),
                "session": "UNAVAILABLE",
                "entry_session": False,
                "regime_bias": 0.0,
                "macro_bias": 0.0,
                "session_bias": 0.0,
                "liquidity_bias": 0.0,
                "htf_high": 0.0,
                "htf_low": 0.0,
                "price": 0.0,
                "displacement": 0.0,
                "rr_ratio": 0.0,
                "has_meaningful_htf_objective": False,
                "liquidity_raid_long": False,
                "liquidity_raid_short": False,
                "mss_bull": False,
                "mss_bear": False,
                "pd_array_long": False,
                "pd_array_short": False,
                "structure_valid_long": False,
                "structure_valid_short": False,
                "market_source": "unavailable",
                "reasons": [str(exc)],
                "passed": False,
            }

        market_source = df.attrs.get("market_source", "unavailable")
        enriched = add_technical_features(df)
        enriched = add_multitimeframe_features(enriched)
        enriched = add_session_features(enriched)
        enriched = add_session_entry_filter(enriched, self.config.get("execution", {}).get("entry_windows") or None)
        enriched = add_liquidity_features(enriched)
        enriched = add_multitimeframe_regime(enriched)
        enriched = add_macro_features(enriched)
        enriched = add_ict_features(enriched)
        enriched[FEATURE_COLUMNS] = enriched[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        X, y = build_training_frame(enriched, horizon=model_cfg["horizon"], threshold=model_cfg["threshold"])
        model = train_signal_model(X, y)
        probabilities = predict_probabilities(model, X)
        enriched["signal_probability"] = probabilities
        enriched["signal"] = enriched.apply(lambda row: self._compose_signal(row, float(row["signal_probability"])), axis=1)

        latest = enriched.iloc[-1]
        decision = self._evaluate_ict_gate(latest, float(latest["signal_probability"]))
        reasons = []
        if not decision["has_meaningful_htf_objective"]:
            reasons.append("no meaningful HTF objective")
        if not (decision["liquidity_raid_long"] or decision["liquidity_raid_short"]):
            reasons.append("liquidity raid failed")
        if not (decision["mss_bull"] or decision["mss_bear"]):
            reasons.append("MSS failed")
        if not (decision["pd_array_long"] or decision["pd_array_short"]):
            reasons.append("PD-array invalid")
        if not (decision["structure_valid_long"] or decision["structure_valid_short"]):
            reasons.append("structure invalid")
        if decision["displacement"] < 1.2:
            reasons.append("weak displacement")
        if decision["rr_ratio"] < 1.5:
            reasons.append("RR below threshold")

        status = "LONG" if decision["signal"] > 0 else "SHORT" if decision["signal"] < 0 else "FLAT"
        return {
            "symbol": market_cfg["symbol"],
            "status": status,
            "signal": int(decision["signal"]),
            "probability": float(decision["probability"]),
            "threshold": float(decision["threshold"]),
            "session": decision["session"],
            "entry_session": decision["entry_session"],
            "regime_bias": float(decision["regime_bias"]),
            "macro_bias": float(decision["macro_bias"]),
            "session_bias": float(decision["session_bias"]),
            "liquidity_bias": float(decision["liquidity_bias"]),
            "htf_high": float(decision["htf_high"]),
            "htf_low": float(decision["htf_low"]),
            "price": float(decision["price"]),
            "displacement": float(decision["displacement"]),
            "rr_ratio": float(decision["rr_ratio"]),
            "has_meaningful_htf_objective": bool(decision["has_meaningful_htf_objective"]),
            "liquidity_raid_long": bool(decision["liquidity_raid_long"]),
            "liquidity_raid_short": bool(decision["liquidity_raid_short"]),
            "mss_bull": bool(decision["mss_bull"]),
            "mss_bear": bool(decision["mss_bear"]),
            "pd_array_long": bool(decision["pd_array_long"]),
            "pd_array_short": bool(decision["pd_array_short"]),
            "structure_valid_long": bool(decision["structure_valid_long"]),
            "structure_valid_short": bool(decision["structure_valid_short"]),
            "market_source": market_source,
            "reasons": reasons,
            "passed": not reasons,
        }

    def run(self) -> Dict[str, Any]:
        market_cfg = self.config["market"]
        model_cfg = self.config["model"]
        risk_cfg = self.config["risk"]

        df = load_market_data(
            symbol=market_cfg["symbol"],
            start_date=market_cfg["start_date"],
            end_date=market_cfg["end_date"],
            use_live_data=self.config["trading"].get("use_live_data", False),
        )
        enriched = add_technical_features(df)
        enriched = add_multitimeframe_features(enriched)
        enriched = add_session_features(enriched)
        enriched = add_session_entry_filter(enriched, self.config.get("execution", {}).get("entry_windows") or None)
        enriched = add_liquidity_features(enriched)
        enriched = add_multitimeframe_regime(enriched)
        enriched = add_macro_features(enriched)
        enriched = add_ict_features(enriched)

        enriched[FEATURE_COLUMNS] = enriched[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        X, y = build_training_frame(enriched, horizon=model_cfg["horizon"], threshold=model_cfg["threshold"])

        model = train_signal_model(X, y)
        probabilities = predict_probabilities(model, X)
        enriched["signal_probability"] = probabilities
        enriched["signal"] = enriched.apply(lambda row: self._compose_signal(row, float(row["signal_probability"])), axis=1)

        latest = enriched.iloc[-1]
        decision = self._evaluate_ict_gate(latest, float(latest["signal_probability"]))
        print("=== ICT Decision Log ===")
        print(f"session={decision['session']}")
        print(f"probability={decision['probability']:.4f} threshold={decision['threshold']:.2f}")
        print(f"signal={decision['signal']} price={decision['price']:.2f}")
        print(f"htf_high={decision['htf_high']:.2f} htf_low={decision['htf_low']:.2f}")
        print(f"meaningful_htf_objective={decision['has_meaningful_htf_objective']}")
        print(f"liquidity_raid_long={decision['liquidity_raid_long']} liquidity_raid_short={decision['liquidity_raid_short']}")
        print(f"mss_bull={decision['mss_bull']} mss_bear={decision['mss_bear']}")
        print(f"pd_array_long={decision['pd_array_long']} pd_array_short={decision['pd_array_short']}")
        print(f"structure_valid_long={decision['structure_valid_long']} structure_valid_short={decision['structure_valid_short']}")
        print(f"displacement={decision['displacement']:.4f} atr={decision['atr']:.4f}")
        print(f"rr_ratio={decision['rr_ratio']:.2f}")
        print(f"regime_bias={decision['regime_bias']:.2f} macro_bias={decision['macro_bias']:.2f} session_bias={decision['session_bias']:.2f} liquidity_bias={decision['liquidity_bias']:.2f}")
        passed = (
            decision["has_meaningful_htf_objective"]
            and (decision["liquidity_raid_long"] or decision["liquidity_raid_short"])
            and (decision["mss_bull"] or decision["mss_bear"])
            and (decision["pd_array_long"] or decision["pd_array_short"])
            and (decision["structure_valid_long"] or decision["structure_valid_short"])
            and decision["rr_ratio"] >= 1.5
            and decision["displacement"] >= 1.2
        )
        if passed:
            print("ICT gate: PASSED -> valid trade setup")
        else:
            reasons = []
            if not decision["has_meaningful_htf_objective"]:
                reasons.append("no meaningful HTF objective")
            if not (decision["liquidity_raid_long"] or decision["liquidity_raid_short"]):
                reasons.append("liquidity raid failed")
            if not (decision["mss_bull"] or decision["mss_bear"]):
                reasons.append("MSS failed")
            if not (decision["pd_array_long"] or decision["pd_array_short"]):
                reasons.append("PD-array invalid")
            if not (decision["structure_valid_long"] or decision["structure_valid_short"]):
                reasons.append("structure invalid")
            if decision["displacement"] < 1.2:
                reasons.append("weak displacement")
            if decision["rr_ratio"] < 1.5:
                reasons.append("RR below threshold")
            print(f"ICT gate: FAILED -> {', '.join(reasons) if reasons else 'unknown failure'}")
            print(f"Flat reason: {', '.join(reasons) if reasons else 'no explicit flat reason'}")
        print("========================")

        metrics = run_backtest(
            enriched,
            model,
            initial_capital=risk_cfg["initial_capital"],
            risk_per_trade=risk_cfg["risk_per_trade"],
            max_position_pct=risk_cfg["max_position_pct"],
            probability_threshold=model_cfg["probability_threshold"],
        )

        latest_signal = int(enriched["signal"].iloc[-1])
        if latest_signal != 0 and len(self.order_manager.get_open_positions()) < self.order_manager.max_positions:
            side = "buy" if latest_signal > 0 else "sell"
            stop_distance = abs(float(enriched["Close"].iloc[-1]) * 0.002)
            qty = self.order_manager.compute_position_size(self.broker.get_account_balance(), stop_distance)
            order = self.order_manager.add_order(side, market_cfg["symbol"], float(enriched["Close"].iloc[-1]), qty, stop_loss=float(enriched["Close"].iloc[-1]) - stop_distance, take_profit=float(enriched["Close"].iloc[-1]) + stop_distance)
            if order.status == "filled":
                self.broker.place_order(side, float(enriched["Close"].iloc[-1]), quantity=qty, stop_loss=order.stop_loss, take_profit=order.take_profit)

        trade_log = getattr(self.broker, "get_trade_log", lambda: [])()

        print("=== AI Trader Summary ===")
        print(f"Symbol: {market_cfg['symbol']}")
        print(f"Samples: {len(enriched)}")
        print(f"Final capital: ${metrics['final_capital']:.2f}")
        print(f"Total return: {metrics['total_return'] * 100:.2f}%")
        print(f"Max drawdown: {metrics['max_drawdown'] * 100:.2f}%")
        print(f"Trades: {metrics['trades']}")
        print(f"Win rate: {metrics['win_rate'] * 100:.2f}%")
        print(f"Macro regime bias: {enriched['macro_bias'].iloc[-1]}")
        print(f"Current session: {enriched['session'].iloc[-1]}")
        print(f"Broker orders: {len(self.broker.get_orders())}")
        print(f"Demo trade entries: {len(trade_log)}")
        return metrics
