# CODE EXECUTION DIRECTIVE

# Aeon W10 — Windows Runtime Integrity and Real-Training Correction

## Status

The prior Windows packaging work is not certified for release or English training.
A source audit identified critical gaps between the documented guarantees and
the actual GUI/worker implementation. W10 must correct those gaps before
GitHub Actions runner capacity is restored and before another Tier A installer
build is attempted.

Begin from the current remote head of `claude/funny-cori-a3k5cf`. The last
reported head was `96c017a7d7af22497a66b1452e2580e389eeb1c3`. Mechanically
verify the actual remote head before modifying files. If it differs, record
the difference and begin from the newest verified additive descendant. Do
not rewrite history.

The GitHub Actions runner-allocation failure is accepted as an external
platform blocker, but it is not the immediate work item. Do not spend
additional Actions minutes or resume the Windows build loop until W10 passes
locally.

## 1. Governing correction

The Windows launcher must become a truthful wrapper around Aeon's certified
training, checkpoint, provenance, security, and recovery systems.

It must not:

- Train on synthetic random token IDs while claiming to train English
- Describe an ordinary checksum checkpoint as authenticated
- Treat "Start" and "Resume" as the same operation
- Display fabricated throughput metrics
- Claim runtime integrity while omitting Aeon.exe
- Claim installer-payload verification when only a file-presence check occurs
- Permit an upgrade to terminate an active worker
- Mark incomplete functionality as MET
- Fall back silently when a protected requirement cannot be satisfied

English training remains prohibited until W10 proves that selected text
passes through the real tokenizer and corpus pipeline.

## 2. Preservation invariants

W10 must preserve:

- Independent transformer and substrate streams
- Recursion as the sole cross-stream integration point
- Existing single Recursion broadcast
- Fixed K=16
- Contractive certificate
- Recursion state in fp32
- Autonomous substrate gate
- Substrate state following parameter dtype
- All six V0.02.02 corrections
- Existing architecture-preservation tests
- Evidence-path canonicalization
- No model-directed shell or subprocess authority
- Local network-denied mode
- Additive Git history

Packaging and GUI convenience may not weaken any inherited invariant.

## 3. Required checkpoint sequence

Use additive commits:

- W10-0 — Audit reproduction and claim withdrawal
- W10-1 — Real tokenizer and corpus training path
- W10-2 — Protected checkpoint integration
- W10-3 — Distinct Start, Resume, and Recovery flows
- W10-4 — Atomic checkpoint-chain repair
- W10-5 — Frozen release provenance
- W10-6 — Complete runtime integrity
- W10-7 — Installer correctness and worker-safe upgrades
- W10-8 — Fail-closed frozen preflight
- W10-9 — Complete desktop operations
- W10-10 — Build reproducibility and licensing
- W10-11 — End-to-end certification and closure

Do not combine unrelated corrections into one large commit.

## 4. W10-0 — Audit reproduction and claim withdrawal

**Objective.** Reproduce each audit finding against the current branch before
implementing fixes.

**Required findings.** Create explicit tests or machine-readable reproductions
for:

1. GUI worker generates random token batches.
2. Configured tokenizer path is not used by training.
3. Configured corpus path is not used by training.
4. Worker uses ordinary atomic_save rather than the protected checkpoint envelope.
5. GUI labels ordinary checkpoint behavior as authenticated.
6. Resume Latest follows the Start path.
7. Configured checkpoint and validation intervals are ignored.
8. CPU-thread and memory limits are not enforced.
9. Runtime throughput metrics are hard-coded or zero placeholders.
10. Runtime manifest excludes top-level Aeon.exe.
11. Malformed manifest entries can be skipped.
12. Unexpected immutable executable files are not rejected.
13. Inno Setup relative source/output paths are ambiguous or wrong.
14. Installer pre-install check verifies presence rather than content.
15. Installer can proceed while a live worker is active outside CHECKPOINTING.
16. Frozen checkpoint provenance falls back to unknown.
17. .prev payload and associated verification metadata are not rotated as one recoverable generation.
18. Frozen preflight can return a nonblocked result without a usable tokenizer or corpus.
19. GUI Validate is incomplete.
20. GUI Recovery still requires terminal intervention.
21. Diagnose discards or hides useful output.
22. Dependency lock ranges permit unreviewed version drift.
23. Workflow actions are not pinned immutably.
24. Release license files may be placeholders.
25. Existing W9 documentation marks unsupported behavior as complete.

**Claim withdrawal.** Add visible banners to affected reports and documentation:

> WITHDRAWN PENDING W10 CORRECTION

Withdraw claims that the current GUI:

- Trains English
- Uses authenticated protected checkpoints
- Implements secure resume
- Verifies the complete installed runtime
- Provides terminal-free recovery
- Is ready for Tier A release certification

Preserve historical fields for auditability. Do not rewrite previous evidence
as though it never existed.

**Exit gate.** W10-0 passes only when every material finding is reproducible
or explicitly disproven with direct code and runtime evidence.

## 5. W10-1 — Real tokenizer and corpus training path

**Objective.** Replace synthetic random-token training with the real certified
data path.

**Requirements.** The worker must:

- Load the user-selected tokenizer through the canonical tokenizer loader.
- Validate tokenizer identity against configuration and provenance.
- Open the selected corpus or corpus manifest.
- Validate corpus identity and provenance.
- Apply the certified preprocessing and tokenization rules.
- Produce real token batches from corpus text.
- Record deterministic data position.
- Resume at the correct corpus position after checkpoint restoration.
- Distinguish training, validation, and test partitions.
- Refuse to train when required corpus or tokenizer data is unavailable.

Remove production fallback to `torch.randint(...)`. Synthetic tokens may
remain only in explicitly named unit-test fixtures that cannot be selected
through the normal GUI.

**Required tests.**

- Known text maps to expected token IDs.
- Worker consumes tokens from the selected corpus.
- Changing corpus content changes produced batches.
- Changing tokenizer identity causes incompatibility failure.
- Empty corpus blocks training.
- Unprovenanced corpus blocks certified training.
- Corrupt corpus fails closed.
- Resume continues from the saved data position.
- Validation does not consume or mutate the training-data position.
- No production code path calls random-token generation.

**Exit gate.** A bounded real-text training fixture must prove that actual
English text reaches model loss computation.

## 6. W10-2 — Protected checkpoint integration

**Objective.** Make the GUI worker use the existing protected checkpoint
lifecycle rather than the ordinary checksum-only path.

**Requirements.** Integrate the worker with the canonical protected APIs for:

- Protected save
- Protected load
- Authentication
- Optional confidentiality
- Provenance verification
- Anti-rollback
- Authorized recovery
- Certificate and dtype validation
- Known-good-state selection

The protected envelope must cover:

- Model state
- Optimizer state
- Scheduler state
- RNG state
- Global step
- Useful-token count
- Data position
- Tokenizer identity
- Corpus identity
- Architecture configuration
- Training configuration
- Runtime and security policy identities
- Source/release identity
- K=16
- Recursion dtype policy
- Substrate dtype policy
- Certificate configuration
- Six-patch manifest
- Architecture-preservation manifest

Do not label a SHA-256 sidecar alone as checkpoint authentication.

**Required behavior.**

- Altered checkpoint bytes are rejected.
- Altered protected metadata is rejected.
- Tokenizer mismatch is rejected.
- Corpus mismatch is rejected.
- Architecture mismatch is rejected.
- Policy mismatch is rejected.
- Unauthorized rollback is rejected.
- Explicitly authorized recovery remains possible.
- Missing key in confidentiality mode produces an explicit blocked state.
- No key material appears in command lines, logs, metrics, or repository content.

**Exit gate.** The GUI worker's Safe Stop must produce a checkpoint that
passes the protected authentication and provenance path.

## 7. W10-3 — Distinct Start, Resume, and Recovery flows

**Objective.** Eliminate the current aliasing of Resume to Start.

**Start New Training must:**

- Require a passing fresh-run preflight
- Refuse an output directory containing an incompatible active chain
- Create a new run identity
- Bind tokenizer, corpus, model, policy, and release identities
- Start from initialized model state unless an explicitly authorized initialization checkpoint is selected

**Resume Latest must:**

- Enumerate checkpoint candidates.
- Authenticate each candidate.
- Verify provenance and compatibility.
- Apply anti-rollback policy.
- Select the newest authorized known-good checkpoint.
- Display its identity and step.
- Require explicit user confirmation.
- Resume optimizer, scheduler, RNG, data position, and global step.
- Record a resume event.

The Resume button remains disabled when no eligible checkpoint exists.

**Recovery must:**

- Be distinct from normal resume
- Identify why current state is unusable
- List verified previous known-good generations
- Require an authorized recovery reason
- Record rollback authorization
- Preserve an immutable recovery audit event
- Revalidate architecture and certificate invariants before activation

**Exit gate.** Start, Resume, and Recovery must have separate code paths,
policies, tests, GUI states, and audit events.

## 8. W10-4 — Atomic checkpoint-chain repair

**Objective.** Make each checkpoint generation and its verification material
one recoverable unit.

**Requirements.** Rotate together:

- Checkpoint payload
- Protected metadata
- Authentication tag or MAC
- Digest
- Provenance record
- Generation identifier
- Authorized-state record where applicable

A previous generation must remain verifiable after a new save. Do not rotate
only the payload while leaving the old sidecar or metadata in place.

Use a generation directory or equivalent transactional design:

    checkpoints/
        generation-000001/
            state
            metadata
            authentication
            provenance
            COMPLETE
        generation-000002.tmp/

A generation becomes eligible only after every component is written,
contents are flushed, authentication is computed, internal verification
passes, a completion marker is written, the generation is atomically
promoted, and the authorized-state pointer is updated safely.

**Required tests.** Interruption after payload write; interruption after
metadata write; interruption before completion marker; corrupt current
generation with intact previous generation; missing authentication file;
mismatched payload and metadata generations; recovery from previous
authenticated generation; no incomplete generation becomes active.

## 9. W10-5 — Frozen release provenance

**Objective.** Ensure installed checkpoints retain the exact source and
release identity without requiring Git.

**Requirements.** When frozen, checkpoint provenance must use embedded
immutable release metadata containing source_commit, build identity, release
version, runtime-manifest identity, architecture-manifest identity,
security-policy identity, runtime-policy identity, build type, signing
status. Use Git only for source-tree development mode. Do not emit
`source_commit: unknown` for a valid frozen release. If embedded release
metadata is missing or malformed, protected training and protected resume
must fail closed.

**Tests.** Source-tree provenance; frozen provenance; missing embedded
metadata; modified embedded metadata; checkpoint built by one release
rejected by an incompatible release; authorized compatible-upgrade migration
path where supported.

## 10. W10-6 — Complete runtime integrity

**Objective.** Cover the complete immutable application, including Aeon.exe.

**Manifest scope.** The runtime manifest must cover the top-level Aeon.exe,
all executable helpers, `_internal`, configuration templates, architecture
manifests, runtime and security policies, required immutable documentation,
license files, version resources. Exclude only explicitly writable user-data
locations outside the installation bundle.

**Verifier requirements.** Fail closed on missing expected file, altered
file, malformed manifest entry, duplicate manifest path, unsafe relative
path, path traversal, unexpected executable file, unexpected DLL,
unexpected policy or configuration file, unsupported manifest version,
manifest identity mismatch. Do not silently skip malformed entries.

**Trusted root.** A manifest stored beside the files it verifies is not
sufficient against an adversary able to modify both. Implement one of:
signed manifest verification using an approved public verification key
embedded in the signed executable or protected release policy; manifest
digest embedded in signed release metadata; another explicitly reviewed
authenticated trust root. For unsigned development builds, state clearly
that integrity verification protects primarily against accidental corruption
and unsynchronized modification, not a privileged adversary able to replace
both executable and trust root.

**Tests.** Modify Aeon.exe; replace Aeon.exe; add unexpected .exe; add
unexpected .dll; modify policy file; modify manifest; remove manifest entry;
insert malformed entry; duplicate path; attempt traversal; verify valid
bundle.

## 11. W10-7 — Installer correctness and worker-safe upgrades

**Objective.** Correct Inno Setup path resolution and prevent upgrade-induced
worker termination.

**Path correction.** Set explicit source and output roots. The installer
script must resolve the repository build output deterministically regardless
of the current working directory or script location. Verify expected paths
mechanically before invoking ISCC.exe.

**Installer payload verification.** Remove the invalid assumption that the
original build tree exists beside AeonSetup.exe. The compiled installer must
verify its packaged payload through a viable design, such as signed
installer integrity plus post-install runtime-manifest verification, or an
embedded release manifest and digest, or installer-controlled post-extraction
verification before completing installation. Do not represent an external
manifest-presence test as payload verification.

**Active-worker safety.** Block upgrade while any live Aeon worker owns an
active job or checkpoint chain, including STARTING, RUNNING, STOP_REQUESTED,
CHECKPOINTING, RECOVERING. Do not use `CloseApplications=force` in a way
that can terminate the worker. Require Safe Stop first. Verify the final
checkpoint before allowing upgrade.

**Tests.** Compile-path resolution from different working directories;
installer build output path; standalone installer with no adjacent build
tree; active worker blocks upgrade; stale job record does not block
indefinitely; checkpointing blocks upgrade; safe-stopped worker permits
upgrade; upgrade preserves configuration; upgrade preserves checkpoints;
uninstall preserves user data by default.

## 12. W10-8 — Fail-closed frozen preflight

**Objective.** Prevent false READY or READY_WITH_WARNINGS results.

**Blocking conditions.** Certified training must report BLOCKED when
tokenizer is missing, tokenizer identity is invalid, corpus is missing,
corpus provenance is incomplete, corpus is empty or unreadable, training
partition is invalid, output path is unwritable, disk is insufficient for
declared checkpoint policy, memory is below the validated minimum, runtime
integrity is unavailable or fails, protected checkpoint policy is
unavailable, certificate verification fails, K is not 16, Recursion
precision is invalid, six-patch manifest is missing, security or runtime
policy is missing, conflicting worker is active, release provenance is
missing in frozen mode.

Warnings are permitted only for noncritical conditions that do not
invalidate safe training.

**Frozen-safe implementation.** Do not depend on source files that are
unavailable inside the PyInstaller bundle. Use packaged manifests, runtime
APIs, and embedded metadata.

**Resource checks.** Actually measure available memory, available disk,
writable space, CPU architecture, required filesystem access. Do not hard-
code passing results.

**Exit gate.** The Start button must remain disabled unless the preflight
returns READY under a complete real-training configuration.

## 13. W10-9 — Complete desktop operations

**Objective.** Make normal operation genuinely terminal-free.

**Validate.** The GUI must execute actual validation and show checkpoint
identity, evaluation partition, loss or other declared metrics, certificate
status, provenance result, output report location.

**Diagnose.** The GUI must select an authenticated checkpoint, run bounded
offline diagnostics, capture structured output, display success or failure,
link to the generated sanitized report. Never discard diagnostic output
silently.

**Recovery.** The GUI must perform the complete protected recovery flow
without instructing the user to open a terminal.

**First-run wizard.** Must validate tokenizer, corpus, provenance, storage,
CPU-thread policy, checkpoint location, logs and evidence locations,
runtime integrity, release metadata.

**Live metrics.** Replace placeholder zeros with actual values: step
duration, raw tokens per second, useful tokens per second, memory, loss,
learning rate, checkpoint state, certificate state. Do not fabricate
unavailable metrics. Display "not measured" when genuinely unavailable.

## 14. W10-10 — Build reproducibility and licensing

**Dependencies.** Replace loose ranges with fully resolved, reviewed pins
for the certified Windows build. Record package, exact version, source/
index, integrity hash where supported, reason required, license. Do not
silently resolve newer versions during release builds.

**Workflow actions.** Pin third-party GitHub Actions to immutable commit
SHAs where practical. Record the human-readable release corresponding to
each pinned SHA.

**Attestation.** Keep attestation outside the installer production critical
path. For private repositories where the selected GitHub plan does not
support attestations, record `ATTESTATION_NOT_AVAILABLE_FOR_CURRENT_PLAN`.
Do not fail unsigned Tier A development installer production solely because
attestation is unavailable.

**Licenses.** Remove placeholder license files. The build must fail if
required third-party notices are missing. Include actual notices for
Python, PyTorch, PyInstaller, Inno Setup as applicable, tokenizer
dependencies, all bundled runtime dependencies requiring attribution.

## 15. W10-11 — End-to-end certification

**Required local/source tests.** Run all inherited and W10 tests. Do not
merely preserve the former count. Report inherited checks, new W10 checks,
total checks, failures, skips.

**Required bounded end-to-end scenario.** Using a real small English
fixture: configure tokenizer; configure corpus; pass preflight; start new
training; verify actual corpus token IDs reach loss computation; cross
multiple K=16 boundaries; observe real metrics; request Safe Stop; produce
protected authenticated checkpoint; close worker; authenticate checkpoint;
resume from it; verify optimizer, RNG, data position, and step continuity;
run validation; run diagnostics; corrupt current checkpoint; reject
corrupted state; recover previous known-good generation; continue bounded
execution; verify audit continuity.

**Required negative scenarios.** Missing tokenizer; missing corpus;
synthetic fallback attempt; tokenizer mismatch; corpus mismatch; tampered
checkpoint; unauthorized rollback; modified Aeon.exe; modified runtime
manifest; unexpected executable; active worker during upgrade; incomplete
checkpoint generation; missing frozen release metadata; invalid certificate;
invalid K; invalid Recursion dtype.

**Documentation reconciliation.** Update every earlier report and definition-
of-done matrix. Each finding must end in one of: CORRECTED, SUPERSEDED,
NOT APPLICABLE with evidence, OPEN BLOCKER. Do not mark unsupported
behavior complete.

## 16. GitHub Actions policy during W10

Do not repeatedly trigger the unavailable runner pool. Disable or restrict
automatic branch-push release builds while runner allocation is unavailable.

Preferred temporary behavior:

- `windows-release.yml`: manual dispatch and release tags only
- Lightweight static workflow validation separated from installer build
- `actions-ping.yml`: retain only if required as audit evidence; otherwise
  disable manual noise and document the result

The correct external-blocker statement is:

> GitHub-hosted runner allocation is unavailable for this repository or
> account. A minimal workflow proves failure before runner provisioning.
> Billing or budget exhaustion is the leading hypothesis, but Actions
> policy, account status, and service status remain possible until verified
> through GitHub account settings or Support.

Do not claim the exact cause until directly confirmed.

## 17. Definition of done

W10 is complete only when:

- Production worker no longer generates synthetic random training tokens.
- Selected tokenizer is actually loaded.
- Selected corpus is actually consumed.
- Real English text reaches model loss computation.
- Data position resumes deterministically.
- GUI worker uses protected checkpoint APIs.
- Safe Stop produces an authenticated protected checkpoint.
- Start and Resume are separate flows.
- Recovery is a separate authorized flow.
- Unauthorized rollback is rejected.
- Checkpoint generations rotate atomically.
- Previous generation remains authentically recoverable.
- Frozen checkpoints use embedded release provenance.
- Aeon.exe is covered by runtime integrity.
- Malformed manifest entries fail closed.
- Unexpected executables and DLLs are rejected.
- Manifest trust limitations are stated accurately.
- Inno Setup paths are deterministic.
- Standalone installer verification design is valid.
- Active workers block upgrade.
- Installer cannot forcibly terminate active training.
- Frozen preflight blocks missing tokenizer or corpus.
- Disk and memory checks are real.
- Integrity and certificate failures block training.
- Validate works from the GUI.
- Diagnose works from the GUI.
- Recovery works from the GUI.
- Live metrics are real or explicitly unavailable.
- Windows dependencies are fully pinned.
- Workflow actions are pinned appropriately.
- Placeholder licenses are removed.
- Earlier inaccurate claims are withdrawn or corrected.
- Complete source regression passes.
- Real-text start-stop-resume scenario passes.
- Corruption and recovery scenario passes.
- No architectural invariant regresses.
- No offensive or unauthorized functionality is added.
- The final W10 ledger and evidence are pushed.
- Working tree is clean.
- No unresolved software blocker remains before Tier A.

## 18. Final execution instruction

Begin W10-0 now. Proceed autonomously through W10-11. Do not resume the
Tier A Windows installer loop merely because GitHub Actions becomes
available. Tier A remains blocked until W10 closes.

Do not begin the primary English-training campaign until the real-text
ingestion and protected start-stop-resume gates pass.

The intended result is a Windows launcher that truthfully trains Aeon on
real English data, preserves authenticated state, resumes safely, verifies
its complete runtime, and can later be packaged without overstating its
protection.
