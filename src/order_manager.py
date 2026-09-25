from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Order:
    side: str
    symbol: str
    price: float
    quantity: float
    stop_loss: float | None = None
    take_profit: float | None = None
    status: str = "pending"


@dataclass
class Position:
    side: str
    quantity: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None


class OrderManager:
    def __init__(self, max_positions: int = 2, max_risk_per_trade: float = 0.01):
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.orders: List[Order] = []
        self.positions: Dict[str, Position] = {}

    def compute_position_size(self, account_balance: float, stop_distance: float, risk_per_trade: float = None) -> float:
        risk = risk_per_trade if risk_per_trade is not None else self.max_risk_per_trade
        if stop_distance <= 0:
            return 0.0
        return max(0.0, (account_balance * risk) / stop_distance)

    def add_order(self, side: str, symbol: str, price: float, quantity: float, stop_loss: float = None, take_profit: float = None) -> Order:
        if len(self.positions) >= self.max_positions:
            return Order(side=side, symbol=symbol, price=price, quantity=quantity, stop_loss=stop_loss, take_profit=take_profit, status="rejected")

        order = Order(side=side, symbol=symbol, price=price, quantity=quantity, stop_loss=stop_loss, take_profit=take_profit, status="filled")
        self.orders.append(order)
        self.positions[symbol] = Position(side=side, quantity=quantity, entry_price=price, stop_loss=stop_loss, take_profit=take_profit)
        return order

    def close_position(self, symbol: str) -> Position | None:
        pos = self.positions.pop(symbol, None)
        if pos is not None:
            self.orders.append(Order(side="close", symbol=symbol, price=pos.entry_price, quantity=pos.quantity, status="closed"))
        return pos

    def get_open_positions(self) -> List[Position]:
        return list(self.positions.values())
