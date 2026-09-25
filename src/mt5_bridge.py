from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


@dataclass
class MT5OrderRequest:
    symbol: str
    side: str
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "AI Trader MT5"
    position_id: Optional[int] = None


class MT5ExecutionBridge:
    def __init__(self, *, path: str | None = None, login: str | None = None, password: str | None = None, server: str | None = None, symbol: str = "XAUUSD"):
        self.path = path
        self.login = login
        self.password = password
        self.server = server
        self.symbol = symbol
        self.connected = False
        self.account_info: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    def connect(self) -> bool:
        if mt5 is None:
            self.last_error = "MetaTrader5 package is not installed. Install it with: pip install MetaTrader5"
            self.connected = False
            return False

        if self.path:
            initialized = mt5.initialize(path=self.path, login=int(self.login) if self.login else 0, password=self.password or "", server=self.server or "")
        else:
            initialized = mt5.initialize(login=int(self.login) if self.login else 0, password=self.password or "", server=self.server or "")

        if not initialized:
            self.last_error = mt5.last_error() if hasattr(mt5, "last_error") else "MT5 initialize failed"
            self.connected = False
            return False

        self.connected = True
        self.account_info = mt5.account_info()
        return True

    def disconnect(self) -> bool:
        if mt5 is not None:
            mt5.shutdown()
        self.connected = False
        return True

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self.connected or mt5 is None:
            return []
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
            return [
                {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": pos.type,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "profit": pos.profit,
                }
                for pos in positions
            ]
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)
            return []

    def place_market_order(self, request: MT5OrderRequest) -> Dict[str, Any]:
        if not self.connected or mt5 is None:
            return {"status": "not_connected", "error": self.last_error or "Not connected to MT5"}

        mt5.symbol_select(request.symbol, True)
        tick = mt5.symbol_info_tick(request.symbol)
        order_type = mt5.ORDER_TYPE_BUY if request.side.lower() == "buy" else mt5.ORDER_TYPE_SELL
        trade_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": float(request.volume),
            "type": order_type,
            "price": request.price if request.price is not None else (tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid),
            "deviation": 10,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        if request.stop_loss is not None:
            trade_request["sl"] = float(request.stop_loss)
        if request.take_profit is not None:
            trade_request["tp"] = float(request.take_profit)

        if request.position_id is not None:
            trade_request["position"] = int(request.position_id)
            trade_request["position_by"] = 0

        result = mt5.order_send(trade_request)
        if result is None:
            last_error = mt5.last_error() if hasattr(mt5, "last_error") else (None, "Unknown")
            return {"status": "error", "error": "order_send returned None", "last_error": last_error}
        return {
            "status": "ok" if result.retcode == mt5.TRADE_RETCODE_DONE else "rejected",
            "retcode": result.retcode,
            "ticket": getattr(result, "order", None),
            "comment": getattr(result, "comment", None),
            "request": trade_request,
        }

    def get_account_state(self) -> Dict[str, Any]:
        if not self.connected or mt5 is None:
            return {"status": "not_connected"}
        info = mt5.account_info()
        if info is None:
            return {"status": "unknown"}
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "currency": info.currency,
        }
