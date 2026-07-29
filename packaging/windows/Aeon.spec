# PyInstaller spec for the Aeon Windows onedir build.
#
# Build on Windows x64 with:
#     pyinstaller --clean packaging/windows/Aeon.spec
#
# Output: dist/Aeon/Aeon.exe (+ _internal/, configs/, docs/, manifests/, licenses/)
#
# Deliberate scope (§W5):
#   * onedir mode (not onefile) — production runtime uses the installed dir.
#   * Windowed subsystem — no console window pops up.
#   * Explicit collection of Aeon packages + minimal PyTorch CPU wheel.
#   * No UPX compression.
#   * NO test data, NO training corpus, NO development keys, NO .git.
#
# Do not add "collect_all" for large packages without documenting the size
# and necessity. Every hidden import listed below is justified inline.
# --------------------------------------------------------------------------

# NOTE: This file is EXEC'd by PyInstaller on Windows. Names like Analysis,
# PYZ, EXE, COLLECT come from PyInstaller at spec-eval time.

from PyInstaller.utils.hooks import collect_submodules, collect_data_files  # noqa: F401 (available at build time)
import os

block_cipher = None
project_root = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..', '..'))  # SPEC provided by PyInstaller  # noqa: F821


# --- explicit Aeon packages (avoid collect_submodules('aeon') to catch drift) ---
aeon_hidden = [
    'aeon',
    'aeon.entry', 'aeon.windows_paths', 'aeon.version', 'aeon.integrity',
    'aeon.hybrid', 'aeon.recursion', 'aeon.transformer',
    'aeon.substrate', 'aeon.substrate.matrix_cell', 'aeon.substrate.vector_cell',
    'aeon.substrate.port', 'aeon.substrate.conformance',
    'aeon.substrate.feedback',
    'aeon.checkpoint', 'aeon.protected_checkpoint',
    'aeon.audit', 'aeon.diagnostics',
    'aeon.observability', 'aeon.evidence',
    'aeon.runtime_policy', 'aeon.continuity',
    'aeon.provenance', 'aeon.corpus_manifest',
    'aeon.tokenizer', 'aeon.data',
    'aeon.adversarial', 'aeon.policies',
    'aeon.launcher', 'aeon.launcher.gui', 'aeon.launcher.controls',
    'aeon.job', 'aeon.job.manager', 'aeon.job.identity',
    'aeon.job.lock', 'aeon.job.worker',
    'aeon.config', 'aeon.config.schema', 'aeon.config.preflight',
]

# --- PyTorch CPU: pin the specific submodules Aeon actually touches ---
# Note: torch collect_submodules pulls in the whole distribution; we prefer
# explicit hidden imports for the CPU forward/backward + optimizer + Cayley SVD.
torch_hidden = [
    'torch', 'torch._C', 'torch._C._distributed_c10d',
    'torch.nn', 'torch.nn.functional',
    'torch.optim', 'torch.optim.adamw',
    'torch.linalg', 'torch.autograd',
    'torch.utils', 'torch.utils.data',
    'torch.random',
]

# --- Analysis ---
a = Analysis(
    [os.path.join(project_root, 'aeon', 'entry.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        # Certified configuration templates
        (os.path.join(project_root, 'configs'), 'configs'),
        # Local operations + recovery guidance (small text)
        (os.path.join(project_root, 'docs', 'OPERATIONS.md'), 'docs'),
        (os.path.join(project_root, 'docs', 'SECURITY_MODEL.md'), 'docs'),
        (os.path.join(project_root, 'docs', 'F3_PROTECTED_CHECKPOINT.md'), 'docs'),
        (os.path.join(project_root, 'docs', 'F4_RUNTIME_CONTAINMENT.md'), 'docs'),
        (os.path.join(project_root, 'docs', 'F9_DEFENSIVE_READINESS.md'), 'docs'),
        # Manifests (runtime + preservation)
        (os.path.join(project_root, 'docs', 'preservation.json'), 'manifests'),
        (os.path.join(project_root, 'docs', 'topology.json'), 'manifests'),
        (os.path.join(project_root, 'docs', 'runtime_policy.json'), 'manifests'),
        (os.path.join(project_root, 'docs', 'threat_model.json'), 'manifests'),
        (os.path.join(project_root, 'docs', 'asset_registry.json'), 'manifests'),
        (os.path.join(project_root, 'docs', 'boundary_registry.json'), 'manifests'),
        # Third-party licenses (torch/sentencepiece/pyyaml/numpy - operator to
        # populate 'licenses' folder before build; see build.ps1).
        (os.path.join(project_root, 'packaging', 'windows', 'licenses'), 'licenses'),
    ],
    hiddenimports=aeon_hidden + torch_hidden + [
        # sentencepiece + yaml are optional at GUI-launch time but required for
        # training workers; include them so the frozen bundle can service any
        # dispatch mode.
        'sentencepiece', 'yaml', 'numpy',
    ],
    hookspath=[os.path.join(project_root, 'packaging', 'windows')],
    runtime_hooks=[os.path.join(project_root, 'packaging', 'windows', 'runtime_hook.py')],
    excludes=[
        # Do NOT ship the whole test suite
        'tests', 'aeon.tests',
        # No CUDA in the CPU build
        'torch.cuda', 'torch.backends.cudnn',
        # No dev-only integrations
        'matplotlib', 'IPython', 'notebook',
        'pytest', 'pytest_asyncio',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],                          # keep binaries out of onefile-style dispatch
    exclude_binaries=True,       # onedir mode
    name='Aeon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # §W5: do NOT use UPX
    console=False,               # windowed subsystem — no console pop-up
    icon=None,                   # operator to supply Aeon.ico via packaging/windows/Aeon.ico
    version=os.path.join(project_root, 'packaging', 'windows', 'file_version_info.txt'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Aeon',
)
