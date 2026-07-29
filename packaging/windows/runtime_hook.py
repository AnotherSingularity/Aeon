"""PyInstaller runtime hook — runs before user code inside the frozen bundle.

Responsibilities:
  * Set sys.frozen (PyInstaller already sets this; belt-and-suspenders).
  * Point AEON_DATA_DIR at %LOCALAPPDATA%\\Aeon on Windows when not overridden.
  * Prevent any accidental use of the CWD for resource resolution.
"""
import os
import sys


sys.frozen = getattr(sys, "frozen", True)

if os.name == "nt" and not os.environ.get("AEON_DATA_DIR"):
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        os.environ["AEON_DATA_DIR"] = os.path.join(localappdata, "Aeon")

# Aeon must not use CWD for resource resolution; the aeon.windows_paths
# functions ignore CWD by design (they use installed_resource_root()).
# The hook does not change CWD to avoid affecting any file dialogs the user
# may open; it only ensures the resource root is well-defined.
