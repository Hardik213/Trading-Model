from __future__ import annotations

from src.paper_trader import PaperTrader


def main() -> None:
    try:
        PaperTrader().run()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
