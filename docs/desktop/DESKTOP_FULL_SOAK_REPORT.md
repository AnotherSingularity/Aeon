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

## 4. Final numbers (from `desktop_full_soak_evidence.json`)

Soak completed cleanly.

| Field                             | Value             |
| --------------------------------- | ----------------- |
| Actual duration                   | 3,600.58 s        |
| Sample count                      | 60 (+ baseline + final = 62 records) |
| Generations completed             | **2,712**         |
| Cancels executed                  | **771**           |
| Session resets                    | **915**           |
| New sessions                      | **902**           |
| Starting RSS                      | 558.2 MB          |
| Ending RSS                        | 679.0 MB          |
| Maximum RSS                       | 679.0 MB          |
| Linear memory trend (end − start) | +120.8 MB         |
| Threads at baseline               | 1                 |
| Threads at final                  | 1                 |
| Child processes                   | 0 (constant)      |
| `last_error` at final             | None              |

### Growth pattern

RSS rises from 558 MB at t=0 to ~672 MB by t≈600 s (torch inductor /
allocator warm-up in the first ~10 minutes), then **plateaus at 679 MB
for the remaining 50 minutes**. Six of the seven 10-minute sample points
after t=619 s read 679.0 MB unchanged. That is a real steady state, not
a slow leak.

### Acceptance gates — all GREEN

| Gate                                            | Result   |
| ----------------------------------------------- | -------- |
| `no_monotonic_unbounded_memory_growth`          | **True** (+120 MB ≪ 500 MB gate; plateau reached by t=600 s) |
| `no_child_process_accumulation`                 | **True** (constant 0 — in-process design)                    |
| `no_thread_accumulation`                        | **True** (1 → 1)                                             |
| `no_deadlock_observed`                          | **True** (soak completed on schedule)                        |
| `no_silent_restart`                             | **True** (PID constant across all 62 samples)                |
| `runtime_state_reachable_ready`                 | **True** (final state = READY)                               |

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
