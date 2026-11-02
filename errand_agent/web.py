"""Local demo UI: the agent's chat and its live trace, side by side.

`python -m errand_agent.web [--live] [--port 8765]`. Standard library only. Binds to localhost.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .factory import DEMO_SCENARIOS, build_agent, live_available
from .followup import FollowUpMonitor

USER = "user-1"
STATIC = Path(__file__).parent / "static"
MONITOR_SECONDS = 2.0


class App:
    """Owns the running agent, so the UI can switch scenario or simulate a process restart."""

    def __init__(self, data_dir: str, trace_dir: str, live: bool, model: str | None, scenario: str):
        self.data_dir, self.trace_dir, self.live, self.model = data_dir, trace_dir, live, model
        self.scenario = scenario
        self.lock = threading.RLock()
        self.events: list[dict] = []
        self.chat: list[dict] = []
        self.delivered = 0
        self.agent = None
        self.monitor = None
        self._start_agent()
        threading.Thread(target=self._monitor_loop, daemon=True, name="followup-monitor").start()

    def _start_agent(self) -> None:
        self.agent = build_agent(self.data_dir, live=self.live, model=self.model, scenario=self.scenario,
                                 trace_dir=self.trace_dir, on_trace_event=self.events.append)
        self.monitor = FollowUpMonitor(self.agent)

    def _monitor_loop(self) -> None:
        stop = threading.Event()
        while not stop.wait(MONITOR_SECONDS):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the monitor must outlive a bad tick
                self.events.append({"type": "monitor_error", "error": str(exc)})

    def tick(self) -> None:
        with self.lock:
            for event in self.monitor.tick(USER):
                if event["text"]:
                    self.chat.append({"role": "agent", "text": event["text"], "tag": f"proactive, {event['channel']}"})

    def send(self, text: str) -> dict:
        self.chat.append({"role": "user", "text": text})
        reply = self.agent.handle_message(USER, text)
        self.chat.append({"role": "agent", "text": reply.text, "tag": f"{reply.outcome}, state {reply.state or 'none'}"})
        return {"outcome": reply.outcome}

    def provider_cancel(self) -> dict:
        with self.lock:
            active = [b for b in self.agent.world.bookings(USER) if b["status"] == "confirmed"]
            if not active:
                return {"error": "No confirmed booking to cancel."}
            self.monitor.tick(USER)
            self.agent.world.simulate_provider_cancellation(active[-1]["id"])
            self.events.append({"type": "simulator", "note": f"provider cancelled {active[-1]['id']}"})
            return {"cancelled": active[-1]["id"]}

    def restart(self) -> dict:
        """Drop all in-memory state and reopen from disk, as a crash and relaunch would."""
        with self.lock:
            self.agent.close()
            self.events.append({"type": "simulator", "note": "process restarted; only disk state survives"})
            self._start_agent()
        return {"restarted": True}

    def reset(self, scenario: str) -> dict:
        with self.lock:
            self.agent.close()
            shutil.rmtree(self.data_dir, ignore_errors=True)
            self.scenario = scenario
            self.events.clear()
            self.chat.clear()
            self._start_agent()
        return {"scenario": scenario}

    def state(self) -> dict:
        world = self.agent.world
        return {
            "scenario": self.scenario,
            "scenarios": sorted(DEMO_SCENARIOS),
            "model": "live: " + (self.agent.model_name or "") if self.live else "offline stand-in (not a real LLM)",
            "chat": list(self.chat),
            "events": [_compact(e) for e in self.events[-400:]],
            "bookings": [{k: b[k] for k in ("id", "service", "provider_name", "slot", "price", "status")} for b in world.bookings(USER)],
            "memories": [{"key": m.key, "value": m.value, "kind": m.kind} for m in self.agent.memory.inspect(USER)],
        }


def _compact(event: dict) -> dict:
    """The context payload is large, and the page only needs its size. The file on disk keeps all of it."""
    return {k: v for k, v in event.items() if k != "payload"}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/state":
                self._json(app.state())
            elif path == "/":
                body = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if path == "/api/send":
                self._json(app.send(str(body.get("text", ""))))
            elif path == "/api/provider-cancel":
                self._json(app.provider_cancel())
            elif path == "/api/restart":
                self._json(app.restart())
            elif path == "/api/reset":
                scenario = body.get("scenario", "default")
                if scenario not in DEMO_SCENARIOS:
                    self._json({"error": "unknown scenario"}, 400)
                else:
                    self._json(app.reset(scenario))
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Errand agent demo UI.")
    parser.add_argument("--live", action="store_true", help="use the real model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--model")
    parser.add_argument("--scenario", default="timeout-after-success", choices=sorted(DEMO_SCENARIOS))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", default="data/web")
    parser.add_argument("--traces", default="traces")
    args = parser.parse_args()
    if args.live and not live_available():
        print("--live needs ANTHROPIC_API_KEY to be set.", file=sys.stderr)
        return 2
    app = App(args.data, args.traces, args.live, args.model, args.scenario)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Errand agent demo: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
