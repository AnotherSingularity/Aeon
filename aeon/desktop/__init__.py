"""aeon.desktop — chat runtime + UI layer for the 7M research preview.

Everything here is EVALUATION-ONLY. The training path (aeon.job.worker,
scripts/train.py) is not reachable from any function in this package.
The desktop runtime executes the SAME aeon.hybrid.HybridModel that the
research campaign trained, via strict, verified release-manifest
loading.

Public entry points:

    aeon.desktop.runtime.AeonDesktopRuntime — the token-generation runtime
    aeon.desktop.protocol.*                 — event / request schemas
    aeon.desktop.chat_ui.run_chat_ui        — Tkinter chat interface
"""
from __future__ import annotations

DESKTOP_RUNTIME_VERSION = "aeon-desktop-runtime-1.0.0"
