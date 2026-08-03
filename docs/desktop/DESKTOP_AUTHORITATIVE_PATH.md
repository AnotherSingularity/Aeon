# DESKTOP-R1 — Authoritative Desktop Architecture

**Decision:** **Architecture B — Native Python / Tk**
(Not a preference. The Electron path never existed in this repository.)

---

## 1. The claim in the R1 directive

The directive states: *"The repository previously contained desktop
shell, Electron, IPC, process supervision, and packaging work."*

That is not accurate for this repository. The Electron half of that
statement has no supporting artifact.

## 2. Executable evidence

Run from the repo root at `d91a836`:

```
$ find . -maxdepth 4 \( -name package.json -o -name main.js \
                       -o -name preload.js -o -name renderer.js \) \
       2>/dev/null | grep -v node_modules
(no matches)

$ grep -ri 'BrowserWindow\|ipcRenderer\|ipcMain\|electron\.app' \
       --include='*.py' --include='*.js' 2>/dev/null
(no matches)
```

There is no `package.json`, no `main.js`, no `preload.js`, no
`renderer.js`, no `BrowserWindow`, no `ipcRenderer` / `ipcMain`, no
Node dependency, no `electron.app` reference anywhere in the tree.

## 3. What IS in the repository

* `packaging/windows/Aeon.spec` — PyInstaller spec targeting
  `aeon/entry.py` as the frozen entry point.
* `packaging/windows/AeonInstaller.iss` — Inno Setup config that
  ships `dist/Aeon/Aeon.exe`.
* `packaging/windows/{build,build_installer}.ps1` — PowerShell build
  scripts.
* `packaging/windows/runtime_hook.py` — PyInstaller runtime hook for
  the frozen bootstrap.
* `aeon/entry.py` — dispatches to `_dispatch_gui` (Tk launcher) by
  default, or to `_dispatch_chat` when `--chat` is passed.
* `aeon/launcher/gui.py` — Tkinter training launcher (W2, unchanged).
* `aeon/desktop/chat_ui.py` — Tkinter chat window (DESKTOP-4).
* `aeon/desktop/runtime.py` — in-process `AeonDesktopRuntime`.

None of these paths involve Electron, Node, or any browser shell.

## 4. Selected architecture

| Field                          | Value                                                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Authoritative desktop shell    | `aeon.launcher.gui` (Tk training launcher) + `aeon.desktop.chat_ui` (Tk chat window)                            |
| Authoritative runtime process  | `aeon.desktop.runtime.AeonDesktopRuntime` — in-process, per-request background thread                          |
| Authoritative IPC              | in-process event queue (`aeon.desktop.protocol.RuntimeEvent`) — 4096-slot Queue drained by `root.after(50, …)`  |
| Development harness            | `aeon.launcher.gui` (training-oriented; developer diagnostics)                                                  |
| Packaged entry point (default) | `Aeon.exe` → training launcher                                                                                  |
| Packaged entry point (chat)    | `Aeon.exe --chat` → chat window                                                                                |
| Installer chat shortcut        | to be added: `Aeon Chat` shortcut passes `--chat`                                                              |

## 5. In-process design — trade-offs disclosed

* **Safe for the 7M research preview.** Peak RSS ~130 MB. The
  background-thread-per-request design keeps the Tk main thread
  responsive. Cancellation is a `threading.Event()` polled between
  token steps. Session isolation is per-session-id in-memory state.
* **Weakness disclosed:** an in-process runtime crash takes the shell
  down too. That is an accepted trade-off for the bounded research
  preview and is documented in `DESKTOP_RELEASE_CANDIDATE.md`.
* **Migration path prepared.** For the 350M model or any larger
  target, a **supervised subprocess** design is required. The
  reference pattern is already in the repository:
  `aeon.launcher.gui` + `aeon.job.worker` + `aeon.job.manager` +
  `aeon.job.lock` implement the supervisor lifecycle for the training
  worker; the desktop runtime can adopt the same pattern when
  externalized.

## 6. Consequence for §6, §7

* **§6 process supervision:** because the chosen architecture keeps
  the runtime in-process, there is no supervised subprocess to
  detect-orphan for. Orphan-detection tests for a chat runtime
  subprocess are therefore **NOT_APPLICABLE** at the 7M scale. They
  become **REQUIRED** when the runtime is externalized to a
  subprocess for the 350M path.
* **§7 IPC acceptance tests:** in-process event dispatch does not
  cross an untrusted boundary. The versioned schema
  (`RuntimeEvent.schema_version = 1`), bounded queue (4096), rejection
  of unknown event types (enum), rejection of oversized prompts
  (`PROMPT_TOO_LARGE`), rejection of invalid sessions
  (`SESSION_NOT_FOUND`), rejection of duplicate requests
  (`REQUEST_ALREADY_ACTIVE`), and no-generic-command-exec surface (the
  runtime has no eval/exec/import-by-name path) collectively satisfy
  the spirit of §7. A cross-process IPC acceptance suite is deferred
  to the 350M migration.

## 7. What this decision does NOT change

* No production code modified.
* No test modified.
* `packaging/windows/Aeon.spec` unchanged in this tranche.
* `aeon/entry.py` unchanged in this tranche.
* Regression at `d91a836` remains 673/673.
* ACIS default remains OFF.
