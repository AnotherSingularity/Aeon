# DESKTOP-0 — Repository and Runtime Inventory

**Desktop branch:** `claude/aeon-desktop-7m-validation`
**Starting head:** `377914b` (research campaign closure) → desktop branch commit `1b39361` (ACIS accounting evidence correction)
**Authoritative model basis:**
* Config: `configs/latent_bypass/aeon_lbc1_proxy.yaml`
* Protected checkpoint: `runs/aeon_lbc1_P2/final.pt`
  — sha256:`962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c`
  — K=16, useful_tokens=1,048,576, n_params=7,015,366, vocab_size=16,000, seed=20260731, stage=P2
* Tokenizer: `research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model`
  — sha256:`064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481`
  — 16,000 vocab, sentencepiece unigram, byte-fallback

---

## 1. Existing infrastructure (classified per §6)

### 1.1 Aeon model (all `aeon/**`)

| Component                              | Path                                | Classification   | Notes                                                       |
| -------------------------------------- | ----------------------------------- | ---------------- | ----------------------------------------------------------- |
| Authoritative model                    | `aeon/hybrid.py`                    | **FUNCTIONAL**   | HybridModel — Transformer + Substrate + Recursion, K=16     |
| Transformer stream                     | `aeon/transformer.py`               | **FUNCTIONAL**   | Aeon-owned; no external model                               |
| Substrate stream                       | `aeon/substrate/**`                 | **FUNCTIONAL**   | Autonomous gate + port + feedback + cell variants           |
| Recursion joiner                       | `aeon/recursion.py`                 | **FUNCTIONAL**   | fp32, σ-certificate, Cayley solve                           |
| Tokenizer wrapper                      | `aeon/tokenizer.py`                 | **FUNCTIONAL**   | Loads AeonTokenizer (sentencepiece)                         |
| Protected checkpoint                   | `aeon/protected_checkpoint.py`      | **FUNCTIONAL**   | HMAC-authenticated                                          |
| ACIS shuttle package                   | `aeon/shuttle/**`                   | **FUNCTIONAL**   | Default OFF; every invariant locked by tests                |
| Bypass telemetry / interventions       | `aeon/bypass/**`                    | **RESEARCH_ONLY**| Do not import from desktop hot path                         |

### 1.2 Windows packaging (all `aeon/**` + `packaging/**`)

| Component                              | Path                                    | Classification | Notes                                                    |
| -------------------------------------- | --------------------------------------- | -------------- | -------------------------------------------------------- |
| Unified entry point                    | `aeon/entry.py`                         | **FUNCTIONAL** | Dispatches GUI / worker / verify / validate / diagnose / recover |
| Windows path resolution                | `aeon/windows_paths.py`                 | **FUNCTIONAL** | is_frozen, installed_resource_root, user_data_root       |
| Release identity                       | `aeon/version.py` (RELEASE_METADATA)    | **FUNCTIONAL** | Referenced from launcher + installer                     |
| Integrity + runtime policy             | `aeon/integrity.py`, `runtime_policy.py`| **FUNCTIONAL** | Runtime manifest verify + policy checks                  |
| Launcher (Tkinter)                     | `aeon/launcher/gui.py` (724 LOC)        | **FUNCTIONAL** but **TRAINING-ORIENTED** | Currently supervises TRAINING runs |
| Worker lifecycle                       | `aeon/job/{worker,manager,identity,lock}.py`| **FUNCTIONAL** | Multi-process training worker + reattach |
| Atomic generation checkpoint chain     | `aeon/job/generation.py`                | **FUNCTIONAL** | Per-generation atomic promote (W10-4)                    |
| PyInstaller spec                       | `packaging/windows/Aeon.spec`           | **FUNCTIONAL** | Onedir; hidden imports pinned                            |
| Windows build script                   | `packaging/windows/build.ps1`           | **FUNCTIONAL** | Requires Windows + Python 3.11                           |
| Inno Setup installer                   | `packaging/windows/AeonInstaller.iss`   | **FUNCTIONAL** | + build_installer.ps1                                    |
| Runtime hook                           | `packaging/windows/runtime_hook.py`     | **FUNCTIONAL** | Frozen bootstrap                                         |
| Runtime manifest generator             | `packaging/windows/generate_runtime_manifest.py` | **FUNCTIONAL** | Verified by aeon.integrity        |
| Signing script                         | `packaging/windows/sign.ps1`            | **PLACEHOLDER**| Unsigned research preview per §33                        |
| Bundle verifier                        | `packaging/windows/verify_bundle.py`    | **FUNCTIONAL** |                                                          |
| Tier A workflow (CI)                   | `.github/workflows/windows-release.yml` | **FUNCTIONAL** |                                                          |
| Tier B workflow (CI)                   | `.github/workflows/windows-certification.yml`| **FUNCTIONAL**|                                                     |

### 1.3 Existing generation / inference

| Component                              | Path                    | Classification | Notes                                              |
| -------------------------------------- | ----------------------- | -------------- | -------------------------------------------------- |
| Greedy CLI generator                   | `scripts/infer.py`      | **RESEARCH_ONLY** — 130 LOC | Single-shot, no streaming, no cancellation, no session |
| Corpus training scripts                | `scripts/train*.py`, `scripts/run_pipeline_stage.py` | **RESEARCH_ONLY** | Must never load from desktop |
| Bypass telemetry runner                | `scripts/run_l3_l4_l5.py` | **RESEARCH_ONLY** | Never callable from desktop path       |

### 1.4 Gaps for a functional chat desktop

| Gap                                          | Status              |
| -------------------------------------------- | ------------------- |
| Inference-only model export tool             | **MISSING**         |
| Release bundle + release manifest schema     | **MISSING**         |
| Export/checkpoint equivalence gate           | **MISSING**         |
| AeonDesktopRuntime chat runtime              | **MISSING**         |
| Streaming event schema for chat generation   | **MISSING**         |
| Mid-generation cancellation                  | **MISSING**         |
| Chat UI (existing launcher is training-only) | **PARTIAL** (Tkinter shell exists) |
| Session isolation contract                   | **MISSING**         |
| Model-compatibility contract for 350M swap   | **MISSING**         |

---

## 2. What we will build

Additive under the DESKTOP-N tranches:

* **DESKTOP-1** — `scripts/export_aeon_desktop_model.py`, release bundle
  under `release-assets/aeon-desktop-p2-proxy/`, release manifest,
  inference-equivalence gate.
* **DESKTOP-2** — `aeon/desktop/runtime.py` — `AeonDesktopRuntime`,
  authentic model load, streaming generation, cancellation, session
  isolation. Reuses `aeon.hybrid.HybridModel` unchanged.
* **DESKTOP-3** — event schema + in-process request/response protocol.
  For the Tkinter shell (single Python process), IPC is intra-process
  event queues; no cross-process socket transport is introduced.
* **DESKTOP-4** — `aeon/desktop/chat_ui.py` — chat window (Tkinter),
  streams from the runtime, honors Stop / New Session / Clear.
* **DESKTOP-5** — session/cancellation/reset/recovery/diagnostics
  focused tests.
* **DESKTOP-6** — extend `packaging/windows/Aeon.spec` hidden-imports
  list with `aeon.desktop.*`; adjust `runtime_hook.py` if needed. Add
  the release bundle under `_internal/release-assets/` (or an
  installer-time payload).
* **DESKTOP-7** — soak + stability + memory-leak matrix runnable on
  the current CPU environment.
* **DESKTOP-8** — release-candidate closure evidence.

## 3. What we will NOT duplicate

Existing infrastructure is REUSED, not replaced:

* No second Electron shell — the Tkinter launcher is the shell.
* No second PyInstaller path — the existing `Aeon.spec` is extended.
* No second Python runtime — the existing frozen runtime hosts the desktop.
* No second installer path — the existing Inno Setup config is extended.
* No parallel training worker — the desktop chat runtime is a
  separate in-process class; the training-worker path is untouched.
* No second checkpoint loader — the export tool builds on
  `aeon.protected_checkpoint` and `aeon.hybrid.HybridModel`.

---

## 4. Regression accounting

* Command used across the campaign: a Python subprocess loop over
  `sorted(glob.glob('tests/test_*.py'))` running each file with
  `python <file>` and parsing `"N checks passed"`. `pytest` is not
  installed in the container; every test file is self-contained and
  has its own `_run_all()` driver.
* Baseline: **55 test files, 627 explicit checks, 0 failing**
  (from `377914b`).
* The DESKTOP tranches add new focused test files under `tests/` and
  keep the total ≥ 627 at every commit.

## 5. What DESKTOP-0 explicitly does not change

* No `aeon/` source modified.
* No config modified.
* No test modified.
* No research evidence altered (the ACIS accounting evidence
  correction at `1b39361` is additive: new paired-trial fields +
  explanatory note; original marginal-statistic fields preserved).
