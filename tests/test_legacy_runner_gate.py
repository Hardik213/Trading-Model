import pytest

import main
import portal
from src.paper_trader import PaperTrader


def test_default_paper_trader_does_not_connect_or_read_market_data():
    trader = PaperTrader(config_path="config.yaml")

    assert trader.broker is None
    assert trader.order_manager is None
    snapshot = trader.build_dashboard_snapshot()
    assert snapshot["status"] == "DISABLED"
    assert "probability" not in snapshot
    assert "metrics" not in snapshot

    with pytest.raises(RuntimeError, match="legacy probabilistic runner is disabled"):
        trader.run()


def test_dashboard_snapshot_does_not_require_a_market_data_connection():
    snapshot = PaperTrader().build_dashboard_snapshot()

    assert snapshot["status"] == "DISABLED"
    assert snapshot["market_source"] == "unavailable"


def test_default_main_and_portal_only_expose_disabled_snapshot():
    with pytest.raises(SystemExit, match="legacy probabilistic runner is disabled"):
        main.main()

    snapshot = portal.get_dashboard_snapshot()
    assert snapshot["status"] == "DISABLED"
    assert "probability" not in snapshot
    assert "metrics" not in snapshot
