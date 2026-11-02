"""A local HTTP server for testing the real SDK adapters without a key or network."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.headers = []
        self.paths = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                outer.paths.append(self.path)
                status, payload = outer.responses.pop(0)
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_api():
    created = []

    def make(responses):
        api = FakeAPI(responses)
        created.append(api)
        return api
    yield make
    for api in created:
        api.close()
