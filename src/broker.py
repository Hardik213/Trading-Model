from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from src.mt5_bridge import MT5ExecutionBridge


@dataclass
class Order:
    side: str
    symbol: str
    price: float
    quantity: float
    status: str = "filled"
    ticket: int | None = None


class BrokerAdapter:
    def __init__(self, symbol: str = "XAUUSD", paper: bool = True):
        self.symbol = symbol
        self.paper = paper
        self.orders: List[Order] = []
        self.account_balance = 10000.0
        self.connected = True

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def place_order(self, side: str, price: float, quantity: float = 0.1, *, stop_loss: float | None = None, take_profit: float | None = None) -> Order:
        order = Order(side=side.lower(), symbol=self.symbol, price=float(price), quantity=float(quantity), status="filled", ticket=len(self.orders) + 1)
        self.orders.append(order)
        return order

    def get_orders(self) -> List[Dict[str, Any]]:
        return [
            {"side": order.side, "symbol": order.symbol, "price": order.price, "quantity": order.quantity, "status": order.status, "ticket": order.ticket}
            for order in self.orders
        ]

    def get_account_balance(self) -> float:
        return float(self.account_balance)

    def get_positions(self) -> List[Dict[str, Any]]:
        return [{"symbol": order.symbol, "side": order.side, "volume": order.quantity, "price": order.price} for order in self.orders if order.status == "filled"]


class DemoExecutionEngine:
    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = float(account_balance)
        self.trade_log: List[Dict[str, Any]] = []
        self.open_positions: Dict[str, Dict[str, Any]] = {}

    def execute_order(self, side: str, symbol: str, price: float, quantity: float, stop_loss: float | None = None, take_profit: float | None = None) -> Dict[str, Any]:
        trade = {
            "side": side.lower(),
            "symbol": symbol,
            "price": float(price),
            "quantity": float(quantity),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "demo_fill",
            "ticket": len(self.trade_log) + 1,
        }
        self.trade_log.append(trade)
        self.open_positions[symbol] = {
            "side": side.lower(),
            "symbol": symbol,
            "volume": float(quantity),
            "entry": float(price),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        return trade

    def get_trade_log(self) -> List[Dict[str, Any]]:
        return list(self.trade_log)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return list(self.open_positions.values())


class MT5DemoBrokerAdapter(BrokerAdapter):
    def __init__(self, symbol: str = "XAUUSD", *, host: str = "localhost", port: int = 5000, username: str = "demo", password: str = "demo", server: str | None = None, path: str | None = None, login: int | None = None, demo: bool = True):
        super().__init__(symbol=symbol, paper=demo)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.server = server
        self.path = path
        self.login = login
        self.demo = demo
        self.connection_log: List[str] = []
        self.execution = DemoExecutionEngine(account_balance=self.account_balance)
        self.bridge = MT5ExecutionBridge(path=path, login=str(login) if login is not None else None, password=password, server=server, symbol=symbol)

    def connect(self) -> bool:
        connected = self.bridge.connect()
        self.connected = connected
        self.connection_log.append(f"MT5 bridge connected={connected}")
        return connected

    def place_order(self, side: str, price: float, quantity: float = 0.1, *, stop_loss: float | None = None, take_profit: float | None = None) -> Order:
        if not self.connect():
            self.connection_log.append(f"MT5 bridge unavailable. Falling back to demo execution for {side} {self.symbol}")
            order = Order(side=side.lower(), symbol=self.symbol, price=float(price), quantity=float(quantity), status="simulated", ticket=len(self.orders) + 1)
            self.orders.append(order)
            self.execution.execute_order(side, self.symbol, price, quantity, stop_loss=stop_loss, take_profit=take_profit)
            return order

        result = self.bridge.place_market_order(
            __import__('src.mt5_bridge', fromlist=['MT5OrderRequest']).MT5OrderRequest(
                symbol=self.symbol,
                side=side,
                volume=float(quantity),
                price=float(price),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        )
        order = Order(side=side.lower(), symbol=self.symbol, price=float(price), quantity=float(quantity), status=result.get("status", "simulated"), ticket=result.get("ticket"))
        self.orders.append(order)
        self.execution.execute_order(side, self.symbol, price, quantity, stop_loss=stop_loss, take_profit=take_profit)
        self.connection_log.append(f"MT5 order result: {result}")
        return order

    def get_positions(self) -> List[Dict[str, Any]]:
        return self.bridge.get_positions() if self.connected else self.execution.get_open_positions()

    def get_trade_log(self) -> List[Dict[str, Any]]:
        return self.execution.get_trade_log()


class BrokerFactory:
    @staticmethod
    def create(config: Dict[str, Any], symbol: str):
        broker_type = (config.get("type") or "paper").lower()
        if broker_type in {"mt5", "mt5-demo", "metatrader", "metatrader5"}:
            login_raw = config.get("login")
            try:
                login = int(login_raw) if login_raw not in (None, "", "None", "null", "0") else None
            except (TypeError, ValueError):
                login = None

            return MT5DemoBrokerAdapter(
                symbol=symbol,
                host=config.get("host", "localhost"),
                port=int(config.get("port", 5000)),
                username=config.get("username", "demo"),
                password=config.get("password", "demo"),
                server=config.get("server"),
                path=config.get("path"),
                login=login,
                demo=bool(config.get("demo_account", False)),
            )
        return BrokerAdapter(symbol=symbol, paper=True)
