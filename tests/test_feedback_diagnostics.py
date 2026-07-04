"""
Sanity-test the feedback diagnostics themselves (spec implementation-order step 3):
each diagnostic must correctly identify BOTH a passing and a failing scenario, so
a green run means "the loop works", not "the test is blind".

For every component we feed the pure decision function controlled PASS and FAIL
inputs (including the confound cases the spec names), then exercise the two
model-measurable ones (sensor, divergence) end-to-end on a tiny model — including
the spec's explicit check: divergence FAILS when W_stressed is a scaled copy of
the normal path.

Requires torch; skips cleanly otherwise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _tiny_model(seed=0, **subcfg):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, tie_word_embeddings=True)
    sub = {"kind": "matrix", "d_in": 24, "d_state": 24, "n_head": 2, "head_size": 12}
    sub.update(subcfg)
    m = HybridModel(h_rec=24, K=4, transformer_config=tcfg, substrate=sub,
                    freeze_backbone=False, use_embedding_input=True, dtype=torch.float32)
    m.recursion.float()
    return m


# ---- Component 1: sensor_correlation --------------------------------------
def test_sensor_decision_and_endtoend():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.diagnostics import sensor_correlation, measure_sensor

    # decision: monotone load vs complexity → PASS
    assert sensor_correlation([0, 1, 2, 3, 4], [0.1, 0.2, 0.3, 0.5, 0.8],
                              [50] * 5).status == "pass"
    # decision: noise → FAIL
    assert sensor_correlation([0, 1, 2, 3, 4], [0.5, 0.1, 0.9, 0.2, 0.4],
                              [50] * 5).status == "fail"
    # decision: correlates with LENGTH not complexity → FAIL (confound guard)
    assert sensor_correlation([4, 3, 2, 1, 0], [1, 2, 3, 4, 5],
                              [1, 2, 3, 4, 5]).status == "fail"
    # end-to-end: the (unlearned) sensor tracks input variability on a real model
    m = _tiny_model()
    levels, loads, lengths = measure_sensor(m, 64, 48, "cpu")
    res = sensor_correlation(levels, loads, lengths)
    assert res.metric > 0.0, f"sensor corr not positive: {res}"


# ---- Component 2: gate_response -------------------------------------------
def test_gate_decision():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    from aeon.diagnostics import gate_response
    obs = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert gate_response(8.0, 0.5, obs).status == "pass"        # real threshold, in range
    assert gate_response(8.0, 100.0, obs).status == "fail"      # always-off (θ too high)
    assert gate_response(8.0, -100.0, obs).status == "fail"     # always-on (θ too low)
    assert gate_response(0.001, 0.5, obs).status == "fail"      # trivial (α≈0, ignores L)


# ---- Component 3: signal_divergence ---------------------------------------
def test_divergence_decision_and_copy_failure():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.diagnostics import signal_divergence, measure_divergence

    torch.manual_seed(0)
    n = torch.randn(16, 24)
    assert signal_divergence(n, torch.randn(16, 24)).status == "pass"   # distinct
    assert signal_divergence(n, 0.5 * n).status == "fail"               # scaled copy (cos=1)

    # end-to-end: random W_stressed at init is divergent → PASS
    m = _tiny_model()
    a, b = measure_divergence(m, 64, 32, "cpu")
    assert signal_divergence(a, b).status == "pass"
    # spec's explicit check: make W_stressed a (scaled) copy of the normal path
    # ⇒ the diagnostic must FAIL.
    with torch.no_grad():
        W = m.substrate.feedback.W_stressed.weight
        W.copy_(0.05 * torch.eye(W.shape[0]))          # stressed ≈ 0.05·base (a copy)
    a, b = measure_divergence(m, 64, 32, "cpu")
    assert signal_divergence(a, b).status == "fail", "divergence test blind to a copy"


# ---- Component 4: plant_response ------------------------------------------
def test_plant_decision():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.diagnostics import plant_response
    B, T, V = 2, 4, 10
    off = torch.zeros(B, T, V)
    on = torch.zeros(B, T, V); on[..., 0] = 6.0           # strong structured shift
    noise = off + 0.01 * torch.randn(B, T, V)             # tiny matched-noise shift
    assert plant_response(off, on, noise, gamma=0.1).status == "pass"
    assert plant_response(off, off.clone(), noise, gamma=0.1).status == "fail"   # no response
    assert plant_response(off, on, noise, gamma=0.0).status == "inconclusive"    # γ≈0: can't respond
    # structured shift no bigger than noise shift → FAIL (responds to magnitude, not structure)
    big_noise = off.clone(); big_noise[..., 1] = 6.0
    assert plant_response(off, on, big_noise, gamma=0.1).status == "fail"


# ---- Component 5: loop_closure --------------------------------------------
def test_loop_decision():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    from aeon.diagnostics import loop_closure
    H = 16
    n = 80
    # PASS: flat load pre-fire (no drop), gate fires at t=30, load then falls
    load = [0.5] * n
    gate = [0.0] * n
    gate[30] = 1.0; gate[31] = 1.0                       # a fire onset at t=30
    for k in range(1, H + 1):
        load[30 + k] = 0.8 - 0.5 * (k / H)               # falls after the fire
    load[30] = 0.8
    assert loop_closure(load, gate, horizon=H).status == "pass"

    # FAIL (unstable): load RISES after the fire
    load2 = [0.5] * n
    load2[30] = 0.3
    for k in range(1, H + 1):
        load2[30 + k] = 0.3 + 0.5 * (k / H)
    assert loop_closure(load2, gate, horizon=H).status == "fail"

    # FAIL (autocorrelation confound): load falls EVERYWHERE, fire explains nothing
    load3 = [1.0 - 0.5 * (t / n) for t in range(n)]
    gate3 = [0.0] * n
    gate3[10] = 1.0; gate3[40] = 1.0                     # fires, but drop == baseline drop
    assert loop_closure(load3, gate3, horizon=H).status == "fail"

    # INCONCLUSIVE: gate never fires
    assert loop_closure([0.5] * n, [0.0] * n, horizon=H).status == "inconclusive"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
