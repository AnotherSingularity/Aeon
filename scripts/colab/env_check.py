"""scripts/colab/env_check.py — CUDA + torch environment probe.

Prints:
  * GPU model
  * VRAM (GB)
  * PyTorch version
  * CUDA version (from torch.version.cuda)
  * torch.cuda.is_available()

Halts non-zero (exit code 5) if CUDA is unavailable — free-Colab
fluency training REQUIRES a GPU runtime.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    import torch
    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version_torch_built_with": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info["device_0"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(p.total_memory),
            "total_memory_gb": round(p.total_memory / (1 << 30), 2),
            "capability_major": int(p.major),
            "capability_minor": int(p.minor),
            "multi_processor_count": int(p.multi_processor_count),
        }
    print(json.dumps(info, indent=2))

    if not info["cuda_available"]:
        print(("HALT: CUDA is unavailable. This notebook requires a "
                "GPU-backed Colab runtime. Choose Runtime > Change runtime "
                "type > T4 GPU (or any GPU) and re-run."), file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
