"""aeon.en_train — English-acquisition training + evaluation infrastructure.

Reads MATHEMATICAL TRAINING AND VERIFICATION SPECIFICATION §1..§24 and
implements them against Aeon's native forward path. Never replaces
Aeon's architecture. Never uses external / teacher / LLM-generated
data or logits.

Public surface:

  aeon.en_train.data      — document-level splitter, exact + 5-gram-Jaccard
                              dedup, tokenizer r_UNK validation, intake
                              schema validator
  aeon.en_train.losses    — L_G (general), L_C (response-masked
                              conversational), effective-token accounting,
                              sequence-bucket construction, mixture sampler
  aeon.en_train.proof     — A₀ architecture fingerprint + Δarchitecture,
                              gradient-path proof, weight-delta proof,
                              native stability gate wrapper
  aeon.en_train.eval      — deterministic sealed-eval runner, R_readable,
                              R_one, R_two, continuity, R_repeat, R_fixation,
                              E_stream (renderer integrity)
  aeon.en_train.trainer   — resumable trainer, LR pilot, gated
                              learning-curve checkpoints, promotion gates
  aeon.en_train.attribution — swap-P2-back attribution test

Every training and inference call goes through
`aeon.hybrid.HybridModel.forward` — the same authoritative Aeon path
used by DESKTOP-R2 (byte-identical logit equivalence proven at
e7847bf).
"""
from __future__ import annotations

EN_TRAIN_SCHEMA_VERSION = 1

# Fixed identities pinned at EN-TRAIN-0.
PROTECTED_P2_SHA256 = "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c"
PROTECTED_TOKENIZER_SHA256 = "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481"
PROTECTED_A0_DIGEST = "sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9"
PROTECTED_TOTAL_PARAMETERS = 7015366
FIXED_K = 16
FIXED_VOCAB_SIZE = 16000

# §3 partition caps
MAX_SINGLE_BOOK_TOKEN_FRACTION = 0.005
MAX_SINGLE_AUTHOR_TOKEN_FRACTION = 0.02

# §4 unknown-token gate
MAX_UNK_RATE = 0.001

# §5..§6 mixture weights (Stage 1 / Stage 2)
STAGE1_MIX = {"D_G": 0.90, "D_C": 0.10, "D_A": 0.00}
STAGE2_MIX = {"D_G": 0.35, "D_C": 0.63, "D_A": 0.02}

# §9 token checkpoints
STAGE1_CHECKPOINTS = (1_000_000, 5_000_000, 10_000_000, 25_000_000)
STAGE1_INCREMENT_AFTER_25M = 25_000_000
STAGE1_CEILING = 100_000_000
STAGE2_CHECKPOINTS = (1_000_000, 5_000_000, 10_000_000, 20_000_000)

# §9 gate
MIN_RELATIVE_IMPROVEMENT = 0.005

# §10 pilot LRs
LR_PILOT_GRID = (1e-5, 3e-5, 1e-4, 3e-4)

# §11 batch
EFFECTIVE_BATCH_TOKENS_TARGET = 16_384

# §14 native diagnostic drift tolerance
NATIVE_DIAG_MAX_REL_DRIFT = 0.05

# §17..§21 promotion gates
GATE_R_READABLE = 0.90
GATE_R_ONE = 0.80
GATE_R_TWO = 0.70
GATE_CONTINUITY = 0.70
GATE_R_FIXATION_MAX = 0.02
GATE_LONG_REPEAT_MAX = 0.05
GATE_E_STREAM = 0.0
