"""Minimal local HTTP server for the Grocery Coach adapter (standard library only, no extra dependencies).

Start it from a terminal (it runs until stopped):  python -m src.coach_server --port 8000

GET  /health     -> {"status": "ok", "schema_version": "1.0"}
GET  /contract   -> priorities, strategies, selection rules, supported scenario, example request
POST /recommend  -> JSON request body (see docs/GROCERY_COACH_INTEGRATION.md); always answers with the response JSON.
                    HTTP 200 for status ok / unsupported / error (read `status`), 400 for a body that is not valid JSON.
Development server: it sends `Access-Control-Allow-Origin: *` so a browser page on another port can call it; do not expose it publicly.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.coach_api import BALANCED_FILL_TOLERANCE, EXAMPLE_REQUEST, PPO_SCENARIO, PRIORITIES, RULES, SCHEMA_VERSION, STRATEGIES, CoachAdapter

LOCK = threading.Lock()
NOT_FOUND = {"status": "error", "errors": [{"code": "not_found", "message": "unknown path"}]}


def make_handler(adapter: CoachAdapter):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(200, {})

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "schema_version": SCHEMA_VERSION})
            elif self.path == "/contract":
                self._send(200, {"schema_version": SCHEMA_VERSION, "priorities": list(PRIORITIES), "strategies": list(STRATEGIES), "rules": RULES,
                                 "balanced_fill_tolerance": BALANCED_FILL_TOLERANCE, "ppo_supported_scenario": PPO_SCENARIO,
                                 "example_request": EXAMPLE_REQUEST})
            else:
                self._send(404, NOT_FOUND)

        def do_POST(self):
            if self.path != "/recommend":
                return self._send(404, NOT_FOUND)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
            except ValueError:
                return self._send(400, {"status": "error", "errors": [{"code": "invalid_json", "message": "the request body is not valid JSON"}]})
            with LOCK:                                   # the adapter reuses cached environments: one request at a time
                self._send(200, adapter.recommend(body))

        def log_message(self, *args):                   # keep the console quiet
            pass

    return Handler


def make_server(port: int = 8000, adapter: CoachAdapter | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(adapter or CoachAdapter()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    port = ap.parse_args().port
    print(f"Grocery Coach adapter on http://127.0.0.1:{port}  (POST /recommend, GET /health, GET /contract)")
    make_server(port).serve_forever()


if __name__ == "__main__":
    main()
