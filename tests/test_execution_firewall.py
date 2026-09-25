import pytest
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
