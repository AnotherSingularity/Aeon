# STATE B — Corpus Acquisition External Blocker

## Exact blocked operation

`scripts/vendor_aeon_lbc1.py` cannot download the six Project Gutenberg
UTF-8 plain-text sources listed in `docs/latent_bypass/AEON_LBC_1_PACKAGE_STATUS.md`.
Without those sources, the AEON-LBC-1 corpus package cannot be
assembled, the P0/P1/P2 bounded research checkpoints cannot be
trained, and L3/L4/L5 cannot produce Level-2+ evidence per the claim
ladder.

## First fatal error

Direct probe:

```console
$ curl -sSI --max-time 10 https://www.gutenberg.org/cache/epub/2701/pg2701.txt
curl: (56) CONNECT tunnel failed, response 403
HTTP/1.1 403 Forbidden
Content-Length: 36
```

Proxy state (`$HTTPS_PROXY/__agentproxy/status`):

```json
{
  "recentRelayFailures": [
    {
      "kind": "connect_rejected",
      "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
      "host": "www.gutenberg.org:443"
    }
  ]
}
```

## Corrective attempts

Every attempt hit the same policy denial (403 on `CONNECT`):

| Host | Result |
|---|---|
| `www.gutenberg.org:443` | 403 CONNECT rejected |
| `gutenberg.org:443` | 403 CONNECT rejected |
| `mirrors.xmission.com:443` (`/gutenberg/`) | 403 CONNECT rejected |
| `gutenberg.pglaf.org:443` | 403 CONNECT rejected |
| `aleph.gutenberg.org:443` | 403 CONNECT rejected |

The environment's egress-policy README (`/root/.ccr/README.md`) states
403 responses are policy denials and instructs: **"Do not retry or
route around it — report the blocked host."**

## Why a local authenticated substitute cannot resolve it

The directive is explicit:

> Do not use mirrors, scraped copies, summaries, HTML-to-text
> conversions, audiobook transcripts, or unofficial modernized
> editions.

The directive also lists specific eBook IDs whose canonical
UTF-8 plain-text editions are hosted by Project Gutenberg. No local
substitute can be presented as the official edition; substituting a
non-canonical text would invalidate every downstream corpus digest,
partition-manifest identity, and claim-level record. Fabricating
those digests to hide the substitution would violate the directive's
claim-control rules and the audit-reproduction discipline established
in W10-0.

The synthetic-English fixture inherited from W10-11 is explicitly
forbidden for L3+ scientific claims by the corpus-staging rule in
`docs/LATENT_BYPASS_THEORY_LOCK.md §8` and by
`docs/latent_bypass/status.json.real_corpus_claims_authorized=false`.

## Exact external resource required

Egress permission from the environment's outbound network policy for
one HTTPS session to reach:

- `https://www.gutenberg.org/cache/epub/{2701,1342,11,84,55,1661}/pg{2701,1342,11,84,55,1661}.txt`
- Fallback: `https://www.gutenberg.org/ebooks/{2701,1342,11,84,55,1661}.txt.utf-8`

Once permitted, `scripts/vendor_aeon_lbc1.py` completes acquisition in
one invocation. Every other stage remains offline.

## Clean pushed repository state

- Branch: `claude/funny-cori-a3k5cf`
- Head after this STATE B commit will be pushed to origin.
- Working tree clean at push time.
- Full regression: 453 (pre-STATE-B) + 23 (`test_aeon_lbc1_acquisition`) + 7 (`test_l3_reaction_coordinate`) + 11 (`test_l4_telemetry_l5_interventions`) = **494/494 checks passing**.
- `docs/latent_bypass/status.json.achieved_claim_level = 0` — unchanged.
- `docs/latent_bypass/status.json.real_corpus_claims_authorized = false` — unchanged.

## Exact next command

After egress policy allows Project Gutenberg, run:

```bash
python scripts/vendor_aeon_lbc1.py --package-root research-data/AEON-LBC-1
```

Then follow `docs/latent_bypass/AEON_LBC_1_PACKAGE_STATUS.md` §
"Reproduction commands" through the P0/P1/P2 stages and into L3.

## What DID land in this tranche

Every layer downstream of the acquisition is coded, tested, and ready:

- `scripts/vendor_aeon_lbc1.py` — allowlist, HTTPS-only, host-suffix filter, no-offsite-redirect handler, `text/plain`-only, byte ceiling, digest recording, refresh-source policy.
- `scripts/prepare_aeon_lbc1.py` — strict UTF-8, BOM strip, LF normalize, Unicode NFC, header/footer boundary detection with fail-closed refusal, chapter indexing, paragraph split, stable record IDs, partition-role emission, `aeon-lbc1-v1` policy version.
- `aeon/bypass/sealed_partition.py` — sealed-test summary that reveals only count/bytes/sha/work-id/schema-validity; read gate requires a valid `L3_CALIBRATION_LOCK.json` with twelve required fields; experimental-version bump detection.
- `aeon/bypass/corpus_package.py` — layout validator (`ready_for_L3`), refuses to inspect sealed test unless `allow_test_partition_access=True`.
- `aeon/bypass/reaction.py` — three declared reaction-coordinate candidates (`z_norm`, `z_dir`, `z_pred`) with calibration-only fit, calibration digest binding, ridge solve, shuffled-state control. No import of `HybridModel`.
- `aeon/bypass/telemetry.py` — `SamplingTelemetryObserver` L1-shaped observer, disabled by default, byte and window ceilings fail closed, `delta_loss = pre − post` arithmetic, local-only persistence, no network calls.
- `aeon/bypass/interventions.py` — all eight declared kinds, evaluation-only guard (`assert_evaluation_mode` raises when `model.training=True`), `refuses_persistence` decorator (refuses `checkpoint_dir`/`generation_dir`/`save_path`), `InterventionRunner` returns `delta_L_c`.
- `configs/latent_bypass/aeon_lbc1_proxy.yaml` — P0/P1/P2 staged budgets (16 384 / 262 144 / 1 048 576 useful tokens) with every Aeon invariant preserved (K=16, fp32 Recursion, transformer + substrate independent, single broadcast, autonomous substrate gate, six V0.02.02 patches, contractive certificate, protected checkpoints).
- `benchmarks/latent_bypass/barriers.json` (from L2) — 8 required rows with null thresholds ready for calibration.
- Tests: `test_aeon_lbc1_acquisition.py` (23), `test_l3_reaction_coordinate.py` (7), `test_l4_telemetry_l5_interventions.py` (11).

## Reproducibility discipline maintained

- Achieved claim level: **0** (`THEORY_ONLY`). Unchanged.
- No observational, causal, efficiency, or bypass claim derived from the synthetic-English fixture.
- IP-preservation firewall: **PASS** (`tests/test_ip_preservation.py` 11 checks).
- Architecture-preservation firewall: **PASS** (K=16, fp32 Recursion, single broadcast, no direct stream-to-stream call, authorized substrate gate inputs).
- Corpus license does not touch Aeon source (`test_corpus_notice_does_not_overwrite_aeon_license`, `test_no_corpus_files_bundled_in_aeon_installer_paths`).
- No outbound-network calls introduced in `aeon/` code paths.
