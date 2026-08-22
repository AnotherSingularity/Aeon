# Windows Release Verification Evidence — WIN-PATCH tranche

Recorded at commit `ce7939e` → HEAD of the WIN-PATCH tranche.

## Environment where this tranche was authored

* Host: Linux container (this session's remote execution environment).
* No Windows runner attached to this session.
* No Inno Setup compiler attached to this session.
* PyInstaller was NOT executed during this tranche.
* `Aeon.exe` was NOT rebuilt during this tranche.
* `AeonSetup.exe` was NOT compiled during this tranche.

The tranche's purpose is to convert manual Windows fixes into
permanent, tested repository changes. Every fix is enforced by
static + unit tests that run on any platform. Where a Windows or
Inno Setup binary would be required to observe the fix end-to-end,
the corresponding evidence must be gathered separately on a real
Windows machine by running the documented commands from
`docs/WINDOWS_RELEASE_GUIDE.md`.

## What WAS verified in this checkout

Static and unit tests for the packaging tranche:

* `tests/test_windows_patch_a.py` — 48/48 pass
* `tests/test_w10_7_installer_correctness.py` — 8/8 pass
* `tests/test_w10_10_build_reproducibility.py` — 10/10 pass
* `tests/test_windows_packaging.py` — 15/15 pass

Full test suite: reported separately by the WIN-PATCH-D verification
step.

Architecture invariance:

| Item | Value |
| ---- | ----- |
| A₀ digest (pinned == live) | `sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9` |
| `total_parameters` | 7,015,366 |
| `K` | 16 |
| `MARGIN_H`, `MARGIN_C` | 0.02, 0.02 |
| Protected P2 SHA-256 (pinned == disk) | `sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c` |
| Protected tokenizer SHA-256 (pinned == disk) | `sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481` |
| Working tree at HEAD | clean |

Boundary discipline:

* `aeon/hybrid.py` — untouched
* `aeon/recursion.py` — untouched
* `aeon/substrate/**` — untouched
* `aeon/desktop/runtime.py` — untouched
* `release-assets/aeon-desktop-p2-proxy/model/**` — untouched
* `release-assets/aeon-desktop-p2-proxy/tokenizer/**` — untouched

## What COULD NOT be verified in this checkout

The following require a Windows runner and are documented as
follow-up evidence-collection work; **the tests above statically
enforce the fixes' structural correctness**, but do not prove a
successful Windows compilation:

1. Clean bundle build via `packaging\windows\build.ps1`.
2. Bundle verification via `packaging\windows\verify_bundle.py`
   against a freshly built `dist\Aeon`.
3. Inno Setup compilation via `packaging\windows\build_installer.ps1`
   (requires ISCC.exe).
4. Release ZIP production via `packaging\windows\build_release.ps1`.
5. Silent installation of `AeonSetup.exe` into an isolated
   temporary directory.
6. Installed `Aeon.exe --version` and `--verify-installation`
   against the installed copy.
7. Optional chat launch via `packaging\windows\test_release.ps1
   -LaunchChat`.
8. Uninstallation / cleanup of the isolated test installation.

None of the above touches existing user data under
`%LOCALAPPDATA%\Aeon`.

## Observation carried forward from the manual Windows build

Inno Setup 6.7.3's compiler banner reports
`Non-commercial use only`. Recorded in `docs/WINDOWS_RELEASE_GUIDE.md`
as a flag for legal / licence review before any commercial
distribution. This is a note, not a legal conclusion.
