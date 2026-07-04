"""
aeon/diagnostics.py — fault-isolation diagnostics for the substrate adaptive
feedback control loop.

The closed loop has five components that fail independently; this module ships
one diagnostic per component. Each diagnostic is split into a PURE DECISION
function (operates on measured arrays, no model) and a MEASUREMENT that drives a
model to produce those arrays. The split lets the decision logic be unit-tested
against controlled pass/fail scenarios ("sanity-test the tests") independently of
any trained checkpoint.

  1. Load sensor L(t)      — sensor_correlation:  L(t) tracks input complexity
  2. Signal gate g(L)      — gate_response:       gate is a real threshold, tuned
  3. Actuator W_stressed   — signal_divergence:   stressed ≠ scaled copy of normal
  4. Plant (transformer)   — plant_response:      output shifts under stressed cond.
  5. Loop closure          — loop_closure:        load drops after the gate fires

Verdicts are pass / fail / inconclusive. On a random (untrained) checkpoint only
the unlearned/structural parts are conclusive (sensor, divergence-at-init); the
gate/plant/loop tests depend on trained parameters (and the plant test needs the
write gate γ to have lifted) and report `inconclusive` rather than a false fail.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# ---- pass/fail thresholds (tunable; defaults follow the spec) --------------
T1_MIN_CORR = 0.5          # sensor: min corr(complexity, load)
T2_LOW_GATE_MAX = 0.10     # gate: g at L=0 must be below this
T2_SAT_GATE_MIN = 0.90     # gate: g at high L must be above this
T3_MAX_COSINE = 0.90       # divergence: mean cos(normal, stressed) must be below this
T4_MIN_KL = 1e-3           # plant: min KL(off || on) to count as a response
T4_STRUCTURE_FACTOR = 2.0  # plant: structured KL must exceed matched-noise KL by this
T5_MIN_DROP_MARGIN = 1e-3  # loop: post-fire load drop must beat the non-fire baseline by this


@dataclass
class DiagnosticResult:
    component: str
    name: str
    status: str            # "pass" | "fail" | "inconclusive"
    metric: float
    detail: str = ""

    def __str__(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "inconclusive": "----"}[self.status]
        return f"  [{mark}] {self.component}: {self.name} (metric={self.metric:.4f}) — {self.detail}"


def _pearson(x, y) -> float:
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    x = x - x.mean(); y = y - y.mean()
    denom = (x.norm() * y.norm())
    if denom < 1e-12:
        return 0.0
    return float((x @ y) / denom)


# ===========================================================================
# Component 1 — Load sensor:  sensor_correlation
# ===========================================================================
def sensor_correlation(complexity_levels, mean_loads, seq_lengths=None) -> DiagnosticResult:
    """PASS if L(t) correlates with input complexity (≥ T1_MIN_CORR) and does so
    MORE than it correlates with mere sequence length (guards the 'measures
    length not complexity' failure the spec names)."""
    corr = _pearson(complexity_levels, mean_loads)
    detail = f"corr(complexity, load)={corr:.3f}"
    length_ok = True
    if seq_lengths is not None:
        lcorr = _pearson(seq_lengths, mean_loads)
        detail += f", corr(length, load)={lcorr:.3f}"
        length_ok = corr > abs(lcorr)
    status = "pass" if (corr >= T1_MIN_CORR and length_ok) else "fail"
    return DiagnosticResult("sensor", "sensor_correlation", status, corr, detail)


# ===========================================================================
# Component 2 — Signal gate:  gate_response
# ===========================================================================
def gate_response(alpha, threshold, observed_loads=None) -> DiagnosticResult:
    """PASS if the gate is a real threshold: ~0 at L=0, saturating at high L, and
    (given observed loads) θ sits inside the range that actually occurs — not so
    high it never fires, not so low it always fires, not so flat (α≈0) it ignores L."""
    alpha = float(alpha); threshold = float(threshold)

    def g(L):
        return float(torch.sigmoid(torch.tensor(alpha * (L - threshold))))

    hi = threshold + 4.0 / max(abs(alpha), 1e-6)     # "well past" threshold for this α
    g_low, g_thr, g_hi = g(0.0), g(threshold), g(hi)
    detail = f"α={alpha:.2f} θ={threshold:.3f} g(0)={g_low:.2f} g(θ)={g_thr:.2f} g(hi)={g_hi:.2f}"

    shape_ok = (g_low < T2_LOW_GATE_MAX and g_hi > T2_SAT_GATE_MIN and alpha > 1e-2)
    range_ok = True
    if observed_loads is not None and len(observed_loads):
        lo = float(torch.quantile(torch.as_tensor(observed_loads, dtype=torch.float32), 0.05))
        hiq = float(torch.quantile(torch.as_tensor(observed_loads, dtype=torch.float32), 0.95))
        range_ok = lo <= threshold <= hiq
        detail += f" obs[5%,95%]=[{lo:.3f},{hiq:.3f}]"
    status = "pass" if (shape_ok and range_ok) else "fail"
    # metric: transition sharpness contrast (how much g moves from 0 to hi)
    return DiagnosticResult("gate", "gate_response", status, g_hi - g_low, detail)


# ===========================================================================
# Component 3 — Actuator:  signal_divergence
# ===========================================================================
def signal_divergence(normal_out, stressed_out) -> DiagnosticResult:
    """PASS if the stressed projection is DIRECTIONALLY distinct from the normal
    one (mean cosine < T3_MAX_COSINE) — not a scaled copy doing no corrective
    work — while staying bounded (magnitude ratio finite)."""
    n = torch.as_tensor(normal_out, dtype=torch.float32)
    s = torch.as_tensor(stressed_out, dtype=torch.float32)
    cos = torch.nn.functional.cosine_similarity(n, s, dim=-1).mean().item()
    n_mag = n.norm(dim=-1).mean().item()
    s_mag = s.norm(dim=-1).mean().item()
    ratio = s_mag / (n_mag + 1e-8)
    detail = f"mean_cos={cos:.3f}, |stressed|/|normal|={ratio:.3f}"
    bounded = ratio < 10.0 and torch.isfinite(s).all().item()
    status = "pass" if (cos < T3_MAX_COSINE and bounded) else "fail"
    return DiagnosticResult("actuator", "signal_divergence", status, cos, detail)


# ===========================================================================
# Component 4 — Plant:  plant_response
# ===========================================================================
def _mean_kl(logits_p, logits_q) -> float:
    lp = torch.log_softmax(logits_p.float(), dim=-1)
    lq = torch.log_softmax(logits_q.float(), dim=-1)
    kl = (lp.exp() * (lp - lq)).sum(dim=-1)          # KL(p||q) per position
    return float(kl.mean())


def plant_response(logits_off, logits_on, logits_noise=None, gamma=None) -> DiagnosticResult:
    """PASS if the transformer's output distribution shifts under stressed
    conditioning (KL(off||on) ≥ T4_MIN_KL) AND the STRUCTURED shift exceeds a
    matched-magnitude random-noise shift by T4_STRUCTURE_FACTOR (so the plant is
    responding to the stressed structure, not just to signal magnitude).

    Inconclusive (not fail) if γ≈0: the write path is closed, so the plant
    *cannot* respond yet — that is a pre-training state, not a broken plant."""
    kl_signal = _mean_kl(logits_off, logits_on)
    detail = f"KL(off||on)={kl_signal:.4f}"
    if gamma is not None and abs(float(gamma)) < 1e-6:
        return DiagnosticResult("plant", "plant_response", "inconclusive", kl_signal,
                                detail + " — γ≈0, write path closed (pre-training)")
    structure_ok = True
    if logits_noise is not None:
        kl_noise = _mean_kl(logits_off, logits_noise)
        structure_ok = kl_signal > T4_STRUCTURE_FACTOR * kl_noise
        detail += f", KL(off||noise)={kl_noise:.4f}"
    status = "pass" if (kl_signal >= T4_MIN_KL and structure_ok) else "fail"
    return DiagnosticResult("plant", "plant_response", status, kl_signal, detail)


# ===========================================================================
# Component 5 — Loop closure:  loop_closure
# ===========================================================================
def loop_closure(load_traj, gate_traj, horizon=32, fire_threshold=0.5) -> DiagnosticResult:
    """PASS if load falls after the gate fires, AND the post-fire fall exceeds the
    fall measured from matched NON-fire anchors (rules out the autocorrelation
    confound the spec names: language load falling on its own).

    load_traj/gate_traj are 1-D per-step sequences. A 'fire onset' is a step where
    the gate crosses fire_threshold upward. For each onset t we compare mean load
    over (t, t+horizon] to load at t; the baseline does the same from steps where
    the gate is low (no fire)."""
    L = torch.as_tensor(load_traj, dtype=torch.float32)
    G = torch.as_tensor(gate_traj, dtype=torch.float32)
    n = L.numel()
    if n < horizon + 2:
        return DiagnosticResult("loop", "loop_closure", "inconclusive", 0.0,
                                f"trajectory too short ({n} < horizon+2)")

    def _drop_after(anchors):
        drops = []
        for t in anchors:
            if t + horizon < n:
                drops.append(float(L[t] - L[t + 1: t + horizon + 1].mean()))
        return sum(drops) / len(drops) if drops else None

    fires = [t for t in range(1, n - horizon)
             if G[t] >= fire_threshold and G[t - 1] < fire_threshold]
    non_fires = [t for t in range(1, n - horizon)
                 if G[t] < fire_threshold and G[t - 1] < fire_threshold]
    if not fires:
        return DiagnosticResult("loop", "loop_closure", "inconclusive", 0.0,
                                "gate never fired on this trajectory")

    fire_drop = _drop_after(fires) or 0.0
    base_drop = _drop_after(non_fires) or 0.0
    margin = fire_drop - base_drop
    detail = (f"post-fire drop={fire_drop:.4f}, baseline drop={base_drop:.4f}, "
              f"margin={margin:.4f}, fires={len(fires)}")
    status = "pass" if margin >= T5_MIN_DROP_MARGIN else "fail"
    return DiagnosticResult("loop", "loop_closure", status, margin, detail)


# ===========================================================================
# Measurements — drive a model to produce the arrays the decisions consume
# ===========================================================================
def complexity_sequences(vocab_size, seq_len, device, seed=0):
    """Five input sequences of increasing variability (the property a
    rate-of-change sensor should track). Level 0 constant → level 4 regime-shift
    random. Swap in real natural-language vs adversarial sequences post-corpus."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = max(vocab_size, 8)
    seqs = {}
    seqs[0] = torch.full((1, seq_len), 3, dtype=torch.long)                       # constant
    per2 = torch.tensor([5, 6]).repeat(seq_len // 2 + 1)[:seq_len]
    seqs[1] = per2[None]                                                          # period-2
    blk = torch.randint(0, v, (16,), generator=g)
    seqs[2] = blk.repeat(seq_len // 16 + 1)[:seq_len][None]                       # period-16
    seqs[3] = torch.randint(0, v, (1, seq_len), generator=g)                      # uniform
    parts = []
    for b in range(seq_len // 32 + 1):
        lo = (b * 101) % max(v - 40, 1)
        parts.append(torch.randint(lo, min(lo + 40, v), (32,), generator=g))
    seqs[4] = torch.cat(parts)[:seq_len][None]                                    # regime-shift
    return {k: t.to(device) for k, t in seqs.items()}


def _feedback(model):
    fb = getattr(model.substrate, "feedback", None)
    if fb is None:
        raise ValueError("substrate has no adaptive feedback controller to diagnose")
    return fb


@torch.no_grad()
def measure_sensor(model, vocab_size, seq_len, device, seed=0, warmup=8):
    fb = _feedback(model)
    seqs = complexity_sequences(vocab_size, seq_len, device, seed)
    levels, loads, lengths = [], [], []
    fb.enable_history()
    for level, ids in sorted(seqs.items()):
        fb.clear_history()
        model(input_ids=ids)
        load_traj, _ = fb.history()
        traj = load_traj[warmup:] if len(load_traj) > warmup else load_traj
        levels.append(level)
        loads.append(sum(traj) / max(len(traj), 1))
        lengths.append(ids.shape[1])
    fb.disable_history()
    return levels, loads, lengths


@torch.no_grad()
def measure_divergence(model, vocab_size, seq_len, device, seed=0, runs=4):
    fb = _feedback(model)
    normals, stresseds = [], []
    for r in range(runs):
        ids = torch.randint(0, max(vocab_size, 8), (2, seq_len), device=device)
        model(input_ids=ids)
        base = fb.last_base()
        n, s = fb.normal_and_stressed(base)
        normals.append(n); stresseds.append(s)
    return torch.cat(normals), torch.cat(stresseds)


@torch.no_grad()
def measure_plant(model, vocab_size, seq_len, device, seed=0, noise_std=None):
    """Run the model with the gate forced OFF, forced ON, and (for the structure
    control) OFF + matched output noise; return the three logit tensors."""
    fb = _feedback(model)
    ids = torch.randint(0, max(vocab_size, 8), (2, seq_len), device=device)
    fb.force_gate = 0.0; fb.inject_noise_std = 0.0
    logits_off = model(input_ids=ids).logits
    fb.force_gate = 1.0
    logits_on = model(input_ids=ids).logits
    logits_noise = None
    if noise_std is not None:
        fb.force_gate = 0.0; fb.inject_noise_std = float(noise_std)
        torch.manual_seed(seed)
        logits_noise = model(input_ids=ids).logits
    fb.force_gate = None; fb.inject_noise_std = 0.0
    return logits_off, logits_on, logits_noise


@torch.no_grad()
def measure_loop(model, vocab_size, seq_len, device, seed=0):
    fb = _feedback(model)
    seqs = complexity_sequences(vocab_size, seq_len, device, seed)
    ids = seqs[4]                                    # the highest-load sequence
    fb.enable_history(); fb.clear_history()
    model(input_ids=ids)
    load_traj, gate_traj = fb.history()
    fb.disable_history()
    return load_traj, gate_traj


@torch.no_grad()
def run_all(model, vocab_size, seq_len=64, device="cpu", seed=0):
    """Run all five diagnostics on a (trained) model. Returns a list of
    DiagnosticResult, one per component, in loop order."""
    fb = _feedback(model)
    results = []

    levels, loads, lengths = measure_sensor(model, vocab_size, seq_len, device, seed)
    results.append(sensor_correlation(levels, loads, lengths))

    results.append(gate_response(fb.gate_alpha, fb.gate_threshold, observed_loads=loads))

    n, s = measure_divergence(model, vocab_size, seq_len, device, seed)
    div = signal_divergence(n, s)
    results.append(div)

    # matched noise magnitude for the plant structure-control = mean |stressed−normal|
    noise_std = float((s - n).abs().mean())
    gamma = float(model.transformer.gamma.detach()) if hasattr(model, "transformer") else None
    l_off, l_on, l_noise = measure_plant(model, vocab_size, seq_len, device, seed, noise_std)
    results.append(plant_response(l_off, l_on, l_noise, gamma=gamma))

    load_traj, gate_traj = measure_loop(model, vocab_size, seq_len, device, seed)
    results.append(loop_closure(load_traj, gate_traj))
    return results
