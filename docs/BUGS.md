# Three real bugs hit while building

All three happened in this repo. The git history shows the state before and after each fix, and the [AI transcript](AI_TRANSCRIPT.md) shows them being found. Regression tests are in [`tests/test_regressions.py`](../tests/test_regressions.py).

## Bug 1: saying "cancel it" booked the plumber

- **What broke:** with a booking prompt pending, the reply `cancel it` (meaning "stop") recorded an approval and made the booking. `book it` on a *cancellation* prompt cancelled the booking.
- **Why:** the approval grammar was one flat set of phrases, including `cancel it` for approving cancellations, and it never looked at which action the prompt was for. All 69 scenarios at the time passed around it, because none of them replied with the wrong action word. I found it by reviewing my own design, then wrote the failing scenario first: `['no approval', 'no booking', 'treated as a rejection']` all failed, so the booking really happened.
- **Fix:** `parse_reply` now takes the proposal's action. Action words count only for their own action. `cancel it` on a booking or reschedule prompt is a rejection. Other mismatches fall through to the model, which can't approve anything.
- **Regression tests:** `test_cancel_it_on_a_booking_prompt_does_not_book`, `test_action_words_only_approve_their_own_action`, and scenarios `cancel_it_does_not_approve_a_booking`, `book_it_does_not_approve_a_cancellation`.

## Bug 2: inferred facts reached the model through chat history

- **What broke:** "unconfirmed inferred facts must never be included" was violated. The scenario `inferred_fact_withheld_until_confirmed` failed with `['model never saw it', 'I5 no forgotten or unconfirmed fact reached the model']`.
- **Why:** the memory layer filtered correctly, and the inspection reply bypassed the model correctly. But that reply was then stored as an ordinary assistant message, so the *next* turn's history replayed the unconfirmed fact to the model. Each component was right; the leak was in how they composed.
- **Fix:** messages gained a `private` flag. Memory listings are stored with it set, and context assembly filters on it, so the user sees them and the model never does.
- **Regression tests:** `test_inspection_reply_never_enters_model_context`, `test_inspection_reply_stays_private_after_the_fact_is_confirmed`, the global invariant I5 (run after every scenario), and scenario `inference_from_booking_history`.

## Bug 3: the mock never committed a scripted booking, and the agent's honesty hid it

- **What broke:** in the first smoke run, a scripted `timeout_booked` produced no booking. `book` was called three times and the user was told nothing was booked.
- **Why:** `book()` ran inside a transaction and called a counter helper that opened its own, so SQLite raised `cannot start a transaction within a transaction`. The tool runner correctly classified that as an unexpected tool error, and the executor correctly treated it as ambiguous and checked the source of truth, which correctly said "not booked". Every safety layer worked, which is exactly what made the broken mock look like a plausible failure path. A direct reproduction against the mock exposed the `OperationalError`.
- **Fix:** `Database.transaction()` is re-entrant: an inner block joins the outer transaction, and the outer one commits or rolls back the whole unit.
- **Regression tests:** `test_transaction_blocks_can_nest`, `test_nested_transaction_rolls_back_as_one_unit`, `test_scripted_timeout_after_success_really_commits`, plus the mock contract tests.

## Also found

These are real too, but smaller. Each has a regression test or scenario.

| Bug | How it was found | Fix |
|---|---|---|
| The success-claim guard rewrote honest messages such as "Nothing has been booked" | 3 scenarios failed | Match per sentence and honour negation; filler like "No worries" doesn't count |
| A budget of `1,500` tokenised to `1`, making `misc_fact_1` look relevant, so irrelevant memories reached the model | Context scenario failed | Bare numbers are no longer relevance tokens |
| Phone redaction turned every ISO date in every trace into `[phone]` | Seen by eye in the live trace pane; no test read trace contents | Count digits, not characters, and never start on an ISO date |
| A `yes` to a later, unrelated question approved the earlier pending proposal | Design review, then a failing scenario | The prompt must be the latest thing shown, or it's shown again |
| A `yes` after any old booking got "already done" instead of answering the new question | Design review, then a failing scenario | A repeat only counts if it answers that execution's own result |
