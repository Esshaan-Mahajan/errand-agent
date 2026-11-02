"""Produce the end-to-end trace deliverable: one booking conversation with a timeout-after-success.

    python -m evals.export_trace          offline stand-in model
    python -m evals.export_trace --live   the real model (needs OPENAI_API_KEY)

All the runs in the conversation are concatenated, in order, into one raw JSONL file.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from errand_agent.agent import Agent
from errand_agent.clock import FakeClock
from errand_agent.factory import LIVE_KEY_HELP, live_available, make_llm
from errand_agent.mocks import WorldConfig

USER = "user-1"
CONVERSATION = ["Find a plumber for my kitchen tap this week, mornings only, under 1,500.", "Indiranagar", "yes"]
OUTPUT = Path("docs/trace_timeout_after_success.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", help="model ID to use with --live")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.live and not live_available(args.model):
        print(LIVE_KEY_HELP, file=sys.stderr)
        return 2
    llm, name = make_llm(args.live, args.model)
    work = Path(tempfile.mkdtemp(prefix="errand-trace-"))
    agent = Agent(work, llm, clock=FakeClock(), user_id=USER, model_name=name, trace_dir=work / "traces",
                  world_config=WorldConfig(book_script=["timeout_booked"]))
    paths: list[str] = []
    for text in CONVERSATION:
        reply = agent.handle_message(USER, text)
        paths.append(reply.trace_path)
        print(f"user> {text}\nagent [{reply.outcome}]> {reply.text}\n")
    bookings = agent.world.bookings(USER)
    book_calls = agent.world.call_count("book")
    agent.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for path in dict.fromkeys(paths):
            handle.write(Path(path).read_text(encoding="utf-8"))
    shutil.rmtree(work, ignore_errors=True)
    print(f"Wrote {args.output}; model={name}; book calls={book_calls}; bookings={len(bookings)}")
    return 0 if len(bookings) == 1 and book_calls == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
