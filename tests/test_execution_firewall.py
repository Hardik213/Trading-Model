import pytest

from src.broker import MT5DemoBrokerAdapter
from src.execution_firewall import ExecutionPolicy


def test_research_cannot_execute():
    p = ExecutionPolicy(mode="research")
    with pytest.raises(PermissionError): p.authorize(destination="demo")
    with pytest.raises(PermissionError): p.authorize(destination="live")


def test_demo_requires_explicit_enablement():
    with pytest.raises(PermissionError):
        ExecutionPolicy(mode="paper", allow_demo=False).authorize(destination="demo")
    ExecutionPolicy(mode="paper", allow_demo=True).authorize(destination="demo")


def test_live_requires_explicit_enablement():
    with pytest.raises(PermissionError):
        ExecutionPolicy(mode="live", allow_live=False).authorize(destination="live")
    ExecutionPolicy(mode="live", allow_live=True).authorize(destination="live")


def test_mt5_order_path_requires_explicit_policy_gate():
    adapter = MT5DemoBrokerAdapter(
        symbol="XAUUSD",
        policy=ExecutionPolicy(mode="research", allow_demo=False, allow_live=False),
    )

    with pytest.raises(PermissionError):
        adapter.place_order("buy", 1950.0, quantity=0.1)
