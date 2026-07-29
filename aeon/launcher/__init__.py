"""aeon.launcher — W2 desktop launcher (Tkinter-based; no visible console).

The launcher never imports torch. Training runs in a separate worker process
spawned via aeon.launcher.controls."""
from aeon.launcher.gui import run_launcher

__all__ = ["run_launcher"]
