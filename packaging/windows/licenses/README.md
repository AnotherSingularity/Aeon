# packaging/windows/licenses

This directory holds the real, verbatim third-party license and notice
text that the Windows release must ship. It is enforced by
`packaging/windows/build.ps1` — the build fails closed if any required
file is missing.

## Contract

The machine-readable contract is `LICENSES_MANIFEST.json` in this
directory. Every file listed under `required[]` must be present with
the exact filename declared there. When an entry carries a `sha256`,
the file's SHA-256 must match; a mismatch fails the build closed.

## How to populate (Windows operator, from a locked build venv)

After `build.ps1` has created `.build-venv` and pinned every runtime
dependency, copy the upstream license text from the wheel's dist-info
directory. For example, for the locked build documented at
commit `447f0dc`:

```powershell
$Venv = '.build-venv\Lib\site-packages'
$Dst  = 'packaging\windows\licenses'

Copy-Item "$Venv\torch-2.5.1.dist-info\LICENSE"                            "$Dst\torch-LICENSE.txt"
Copy-Item "$Venv\torch-2.5.1.dist-info\NOTICE"                             "$Dst\torch-NOTICE.txt"
Copy-Item "$Venv\pyinstaller-6.21.0.dist-info\licenses\COPYING.txt"        "$Dst\pyinstaller-COPYING.txt"
Copy-Item "$Venv\sentencepiece-*.dist-info\LICENSE"                        "$Dst\sentencepiece-LICENSE.txt"
Copy-Item "$Venv\safetensors-0.8.0.dist-info\licenses\LICENSE"             "$Dst\safetensors-LICENSE.txt"
Copy-Item "$Venv\numpy-2.5.1.dist-info\licenses\LICENSE.txt"               "$Dst\numpy-LICENSE.txt"
Copy-Item "$Venv\pyyaml-6.0.3.dist-info\licenses\LICENSE"                  "$Dst\pyyaml-LICENSE.txt"
```

If a wheel does not ship a license file, obtain the exact text from
the upstream repository at the tag matching the locked version, then
record its bytes in the working tree.

## What NOT to do

* Do not commit placeholders, summaries, or paraphrases.
* Do not commit `PLACEHOLDER.txt` — the build refuses it.
* Do not use a different version's license as a substitute; the
  locked version is what ships.
* Do not edit line endings or transcode the file — `sha256` compares
  bytewise.
* Do not swap `LICENSES_MANIFEST.json` for a smaller list to make the
  build pass.

## Provenance recording

After populating any file, update its `sha256` in
`LICENSES_MANIFEST.json` in the same commit. From this directory:

```powershell
Get-FileHash -Algorithm SHA256 torch-LICENSE.txt
```

Paste the lowercase hex digest into the corresponding `sha256` field
under `required[]`.

## Build enforcement

`build.ps1` reads `LICENSES_MANIFEST.json` and verifies every
`required[]` entry. When `sha256` is `null`, the build performs a
presence-only check for that entry (so a fresh operator can populate
locally without a repository update first). When `sha256` is set, the
build performs bytewise verification. Either way, absence of a
required file fails the build closed.
