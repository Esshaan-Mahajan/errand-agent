# AI assistant usage

This is a summary, written by the assistant from its own account of the session. It is not the transcript. The transcript is a separate file: see the README's deliverables table.

**Assistant used:** Claude Code (Claude Opus 5.5), in one session. It wrote the code, tests and documents.

## How the work went

1. Read the assignment and chose Python, SQLite and pytest. The repo started empty.
2. Built the mocks, approval gate, state machine, executor, memory, context builder, traces, LLM adapters and agent loop.
3. Ran a first smoke test, which exposed the mock's nested-transaction bug (Bug 3).
4. Wrote the eval suite. The first run passed 63 of 69 scenarios. That state was committed as it stood, and the failures were fixed in the next commit (Bug 2 and two smaller bugs).
5. Reviewed the design for gaps the suite couldn't see, and wrote each one as a failing scenario before fixing it (Bug 1 and two smaller bugs).
6. Drove the UI in a browser, which exposed the trace date corruption by eye.
7. Checked a fresh clone: it installs and passes all tests.

## What the assistant could not do

- **Run the real model.** No API key was available, so the live path was tested only against local servers speaking each API's format. The live default moved from Anthropic to OpenAI because the author's credit is with OpenAI.
- **Record the demo.** Screen recording and upload are left to the author. See [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## Where it needed correcting

The test suite and the author's own checks caught these. They are recorded so the summary isn't only a success story:

- One commit (`6984c97`) went in with a test failing, because a shell pipe hid pytest's exit code. Later commits were gated on the real exit code.
- A global invariant was too broad twice, flagging legitimate area names as leaks, and was narrowed to the memory block.
- A regression test banned the `[phone]` mask outright, which wrongly failed on a provider's real number. It now checks that dates survive and phone numbers are masked.
