# DESKTOP-R5 — Full Continuous Soak Report

**Runner:** `scripts/run_desktop_full_soak.py --seconds 3600 --sample-interval 60`
**Target duration:** 3,600 seconds (one hour) — actual duration
recorded in `docs/desktop/desktop_full_soak_evidence.json`
`actual_duration_seconds` field.
**Release bundle under test:** `release-assets/aeon-desktop-p2-proxy/`
(the same bundle the desktop chat runtime loads).

Evidence live-updates during the run into
`docs/desktop/desktop_full_soak_evidence.json` — one sample every
60 seconds plus a baseline and a final sample.

---

## 1. Actions exercised

Each ~30 s cycle draws uniformly at random from:

* `generate` (weight 3) — new generation, 4–8 tokens, greedy
* `idle` (weight 1) — 2–5 s sleep
* `cancel` (weight 1) — start a long generation, wait 200–500 ms, cancel
* `reset` (weight 1) — `reset_session` on the current session
* `new_session` (weight 1) — close current, create a new one
* `status` (weight 1) — `diagnostics()` call

## 2. Sampled fields per §9

Every sample records:

* `elapsed_s`
* `pid` (host process)
* `child_process_count` (constant 0 under the in-process design)
* `rss_mb` (from `resource.getrusage(RUSAGE_SELF).ru_maxrss`)
* `thread_count` (from `threading.active_count()`)
* `generation_count` (cumulative)
* `cancel_count` (cumulative)
* `runtime_state` (RuntimeState value)
* `last_error` (last exception seen, if any)
* `last_action`

## 3. Acceptance gates

From §9:

| Gate                                              | Method                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------- |
| No monotonic unbounded memory growth              | `ending_rss - starting_rss < 500 MB`                                |
| No child-process accumulation                     | constant 0 (in-process design)                                      |
| No thread accumulation                            | `final.thread_count <= baseline.thread_count + 4`                   |
| No deadlock                                       | soak completes; runtime state observable at final                   |
| No silent restart                                 | pid constant across samples                                         |
| No corrupted session                              | no `SESSION_NOT_FOUND` in `last_error`                              |
| No architecture violation                         | R2 trace already covered — not re-run here                          |
| No network attempt                                | soak does not install the R4 socket guard; §8's dynamic denial test is the covering evidence |
| No unrecoverable UI freeze                        | headless runner has no UI thread; the chat_ui `_drain_events` interval (50 ms) is the UI-thread evidence |

## 4. Final numbers

Filled in on soak completion by the runner's `acceptance` block. See
`docs/desktop/desktop_full_soak_evidence.json`:

* `starting_rss_mb`, `ending_rss_mb`, `max_rss_mb`
* `linear_memory_trend_mb` (ending − starting)
* `generation_count`, `cancel_count`, `reset_count`, `new_session_count`
* `acceptance.no_monotonic_unbounded_memory_growth` (bool)
* `acceptance.no_thread_accumulation` (bool)

## 5. What this soak does NOT cover

* It does not test the FROZEN runtime — that's WINDOWS-1.
* It does not test with the OS-level network firewall blocking egress
  — that's WINDOWS-3.
* It does not exercise the actual Tk UI event loop — the runner uses
  the headless `AeonDesktopRuntime` API directly. Chat UI tests
  covering the button state machine + drain interval are in
  `tests/test_desktop_3_4_5.py`.

## 6. If the soak fails any gate

Per §9's *"When a defect occurs"* rule: preserve evidence, fix the
root cause, restart the affected tranche from zero. Do NOT count
pre-fix trials.
