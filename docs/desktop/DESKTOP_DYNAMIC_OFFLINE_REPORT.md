# DESKTOP-R4 — Dynamic Offline Certification

**Verdict:** **PROVEN.** The authentic desktop generation pipeline runs
end-to-end with the process's outbound-network primitives replaced by
raise-on-call functions. No `socket.connect`, no
`socket.create_connection`, no `urllib.request.urlopen` — nothing
attempts to reach the network.

Machine-readable: `docs/desktop/desktop_dynamic_offline_evidence.json`.

---

## 1. Method

`tests/test_desktop_r3_r4_supervision_and_offline.py` installs a
per-test guard that monkey-patches:

* `socket.socket.connect` → raises `_NetworkAttempted`
* `socket.create_connection` → raises `_NetworkAttempted`
* `urllib.request.urlopen` → raises `_NetworkAttempted`

With the guard installed, `test_R4_desktop_pipeline_runs_without_any_outbound_network_attempt`
runs the full desktop path:

* `AeonDesktopRuntime.preflight()`
* `AeonDesktopRuntime.load_release(BUNDLE)`
* `create_session()`
* `submit_prompt_sync("Alice", max_new_tokens=6, temperature=0.0)`
* `submit_prompt(...)` + `cancel(...)` + `join()`
* `reset_session(sid)`
* `shutdown()`

Result: `generation_completed` observed, cancellation returns to
READY, no network guard fired. Guard restores original socket +
urllib on test teardown.

## 2. What this DOES prove

* At the Python process level, the authentic Aeon runtime performs no
  outbound socket connect and no `urlopen` during any operational
  step. That is real dynamic evidence, not static source-code
  inspection.

## 3. What this does NOT prove

* It does not prove behavior in the **frozen** Windows runtime. The
  frozen build has never been produced, so its runtime network
  behavior is untested at this commit. That step is WINDOWS-1
  through WINDOWS-3.
* It does not intercept OS-level DNS traffic — a lower-level guard
  (e.g. Windows firewall rule set) will be applied when WINDOWS-3
  runs the installed application on a real Windows machine with
  outbound egress blocked at the OS.

## 4. Retained static evidence

`test_R4_desktop_source_has_no_network_client_import` runs an AST-free
substring scan of `aeon/desktop/*.py` for `import requests`,
`import httpx`, `import aiohttp`, `import websocket`, and
`from urllib.request import urlopen`. Retained as a source-level
belt-and-braces check — not offered as dynamic proof.

## 5. Thread-leak observation

`test_R4_no_new_thread_leaks_after_repeated_gen_cancel` runs five
generate + cancel cycles and asserts the active thread count grows
by at most 4 above baseline. Prevents a slow leak from being disguised
by short soak runs.
