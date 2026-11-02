"""Terminal chat. `python -m errand_agent.cli [--live] [--scenario NAME]`"""
from __future__ import annotations

import argparse
import sys

from .factory import DEMO_SCENARIOS, build_agent, live_available
from .followup import FollowUpMonitor

USER = "user-1"
HELP = """Commands:
  /bookings                list bookings in the mock world
  /provider-cancel [ID]    simulate a provider cancelling a booking, then run the monitor
  /trace                   show the path of the last run's trace
  /quit                    exit
Anything else goes to the agent. Example: Find a plumber for my kitchen tap this week, mornings only, under 1,500."""


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the errand agent.")
    parser.add_argument("--live", action="store_true", help="use the real model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--model", help="model ID to use with --live")
    parser.add_argument("--scenario", default="default", choices=sorted(DEMO_SCENARIOS))
    parser.add_argument("--data", default="data/cli", help="directory for persistent state")
    args = parser.parse_args()

    if args.live and not live_available():
        print("--live needs ANTHROPIC_API_KEY to be set.", file=sys.stderr)
        return 2
    agent = build_agent(args.data, live=args.live, model=args.model, scenario=args.scenario)
    monitor = FollowUpMonitor(agent)
    print(f"Errand agent ({'live model' if args.live else 'offline stand-in model'}, scenario: {args.scenario}). /help for commands.")
    last_trace = None
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/quit":
            return 0
        if text == "/help":
            print(HELP)
        elif text == "/trace":
            print(last_trace or "No run yet.")
        elif text == "/bookings":
            for booking in agent.world.bookings(USER):
                print(f"  {booking['id']}  {booking['service']:<12} {booking['provider_name']:<14} {booking['slot']}  {booking['price']}  {booking['status']}")
        elif text.startswith("/provider-cancel"):
            active = [b for b in agent.world.bookings(USER) if b["status"] == "confirmed"]
            parts = text.split()
            target = parts[1] if len(parts) > 1 else (active[-1]["id"] if active else None)
            if target is None:
                print("No confirmed booking to cancel.")
                continue
            monitor.tick(USER)
            agent.world.simulate_provider_cancellation(target)
            for event in monitor.tick(USER):
                print(f"\nagent [{event['channel'] or 'silent'}]> {event['text'] or '(decided not to notify: ' + event['decision'].reason + ')'}")
        else:
            reply = agent.handle_message(USER, text)
            last_trace = reply.trace_path
            print(f"\nagent> {reply.text}")
            print(f"       [{reply.outcome} | state: {reply.state}]")


if __name__ == "__main__":
    sys.exit(main())
