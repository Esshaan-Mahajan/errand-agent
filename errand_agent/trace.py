"""Per-run JSONL traces. One file per run; every line is one event."""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path

from .sanitize import redact


class Tracer:
    def __init__(self, directory: str | Path, clock, user_id: str, kind: str = "message", on_event=None):
        self.run_id = f"run-{clock.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.path = Path(directory) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.seq = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.cost_known = False
        self._lock = threading.Lock()
        self._on_event = on_event
        self.event("run_start", user_id=user_id, kind=kind)

    def event(self, event_type: str, **fields) -> None:
        with self._lock:
            self.seq += 1
            record = {
                "seq": self.seq,
                "run_id": self.run_id,
                "ts": self.clock.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "type": event_type,
                **redact(fields),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        if self._on_event:
            self._on_event(record)

    def model_call(self, *, model: str, turn: int, latency_ms: float, input_tokens: int, output_tokens: int,
                   cost_usd: float | None, stop_reason: str | None, response: list[dict]) -> None:
        self.model_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if cost_usd is not None:
            self.cost_usd += cost_usd
            self.cost_known = True
        self.event("model_call", model=model, turn=turn, latency_ms=round(latency_ms, 1),
                   input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=cost_usd,
                   stop_reason=stop_reason, response=response)

    def finish(self, outcome: str, reply: str, **fields) -> None:
        self.event("run_end", outcome=outcome, reply=reply, model_calls=self.model_calls,
                   tool_calls=self.tool_calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                   estimated_cost_usd=round(self.cost_usd, 6) if self.cost_known else None, **fields)


def read_trace(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scrub_traces(directory: str | Path, value: str) -> int:
    """Remove a forgotten value from stored traces. Returns the number of files rewritten."""
    pattern = re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
    changed = 0
    for path in Path(directory).glob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        scrubbed, count = pattern.subn("[forgotten]", text)
        if count:
            path.write_text(scrubbed, encoding="utf-8")
            changed += 1
    return changed
