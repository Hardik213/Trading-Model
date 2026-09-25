from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.paper_trader import PaperTrader

ROOT = Path(__file__).resolve().parent
PORT = int(__import__("os").getenv("PORT", "8000"))


def get_dashboard_snapshot():
    trader = PaperTrader()
    return trader.build_dashboard_snapshot()


def write_status_file(path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(get_dashboard_snapshot(), fh, indent=2)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/status.json":
            write_status_file(ROOT / "status.json")
            payload = json.dumps(get_dashboard_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()


if __name__ == "__main__":
    write_status_file(ROOT / "status.json")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"Dashboard available at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
