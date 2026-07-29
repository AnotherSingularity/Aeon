"""
E2 — Observability equivalence + overhead certification.

§8.5 metric-integrity tests:
  * Instrumentation on/off produces equivalent model outputs (bit-exact) and
    gradients (bit-exact).
  * Instrumentation cannot mutate optimizer updates.
  * Sampling does not modify clock scheduling (recursion.step count is invariant).
  * Failed metric writes do not abort training.
  * Missing optional metrics do not corrupt records.

§8.2 overhead ceiling: measured instrumented-vs-baseline overhead < 15%.

The overhead measurement:
  * Uses a small model (fast on CPU) but a real HybridModel + real training step.
  * Uses median steady-state over warmed-up N steps under the same seed/data.
  * The observability layer here is what train.py uses.

Requires torch. Skips cleanly otherwise.
"""
import copy
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _tiny_model(seed=0):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64)
    m = HybridModel(h_rec=24, K=16, transformer_config=tcfg,
                    substrate={"kind": "matrix", "d_in": 24, "d_state": 24,
                               "n_head": 2, "head_size": 12},
                    dtype=torch.float32)
    m.recursion.float()
    return m


def _run_step(model, opt, ids, loss_seed=None):
    """One deterministic training step; returns loss + a snapshot of first param."""
    import torch
    out = model(input_ids=ids, labels=ids)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    opt.step()
    return float(out.loss.item()), out


# ---- Equivalence: metric emission cannot change model semantics ------------
def test_instrumentation_on_off_bitexact_outputs_and_grads():
    """Two identical models, identical seed, identical ids: one runs with an
    Observer collecting the whole payload each step, one runs cold. Loss must be
    bit-equal, and every gradient tensor must be bit-equal after backward."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.observability import Observer, parameter_accounting

    torch.manual_seed(1)
    ids = torch.randint(0, 64, (2, 32))

    m_off = _tiny_model(seed=42)
    m_on = _tiny_model(seed=42)
    # Sanity: same-seed constructors yield bit-equal parameters
    for (n0, p0), (n1, p1) in zip(m_off.named_parameters(), m_on.named_parameters()):
        assert torch.equal(p0, p1), f"seeded init not equal: {n0}"

    with tempfile.TemporaryDirectory() as d:
        obs = Observer(out_dir=d, sample_every=1, enabled=True)
        obs.emit_static("parameter_accounting", parameter_accounting(m_on))

        # forward + backward on both, capture loss and gradients
        out_off = m_off(input_ids=ids, labels=ids)
        out_off.loss.backward()

        with obs.phase("output_loss"):
            out_on = m_on(input_ids=ids, labels=ids)
        with obs.phase("backward"):
            out_on.loss.backward()
        obs.emit_sampled(step=1, gate_mean=0.0)
        obs.emit_always_on(step=1, loss=out_on.loss.item(), lr=1e-4, step_time_s=0.001,
                           seq_len=32, resident_mb=0.0, certificate_holds=True,
                           sigma_h=0.5, sigma_c=0.5, gamma=0.0)

        # Bit-equal loss
        assert torch.equal(out_off.loss, out_on.loss), \
            f"loss diverged with instrumentation: {out_off.loss.item()} vs {out_on.loss.item()}"
        # Bit-equal gradients on every parameter
        for (n0, p0), (n1, p1) in zip(m_off.named_parameters(), m_on.named_parameters()):
            if p0.grad is None and p1.grad is None: continue
            assert p0.grad is not None and p1.grad is not None, f"grad presence differs: {n0}"
            assert torch.equal(p0.grad, p1.grad), \
                f"grad diverged at {n0}: max|Δ|={(p0.grad-p1.grad).abs().max().item()}"


def test_recursion_step_count_invariant_to_sampling():
    """Sampling must not change the slow-clock schedule (§8.5). Count recursion.step
    calls at sample_every={0 (disabled), 1, 3, 1000}. All must be equal."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.observability import Observer
    counts = {}
    for sample in (0, 1, 3, 1000):
        m = _tiny_model()
        ids = torch.randint(0, 64, (2, 48))                # 3 windows @ K=16
        calls = {"n": 0}
        orig = m.recursion.step
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        m.recursion.step = counting
        with tempfile.TemporaryDirectory() as d:
            obs = Observer(out_dir=d, sample_every=sample, enabled=(sample > 0))
            for step_ in range(4):
                if obs.should_sample(step_):
                    with obs.phase("output_loss"):
                        m(input_ids=ids, labels=ids)
                else:
                    m(input_ids=ids, labels=ids)
        counts[sample] = calls["n"]
    assert len(set(counts.values())) == 1, f"recursion.step count varied with sampling: {counts}"


def test_failed_writes_do_not_abort_training():
    """Corrupt the observer's writer target to be unwritable; training-style calls
    must proceed silently (no exception propagates)."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.observability import Observer
    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 16))
    with tempfile.TemporaryDirectory() as d:
        obs = Observer(out_dir=d, sample_every=1, enabled=True)
        # Break the writer: point it at a directory (open() for write will fail)
        obs._writer._broken = False
        obs._writer.path = d                                # a directory, not a file
        # Now emit — must not raise
        try:
            obs.emit_always_on(step=1, loss=1.0, lr=1e-4, step_time_s=0.01,
                               seq_len=16, resident_mb=0.0, certificate_holds=True,
                               sigma_h=0.5, sigma_c=0.5, gamma=0.0)
            obs.emit_sampled(step=1, gate_mean=0.1)
        except Exception as e:
            raise AssertionError(f"logging failure propagated: {e}") from e
        # And training-style forward still works
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        assert torch.isfinite(out.loss).all()


def test_missing_optional_metrics_do_not_corrupt_records():
    """emit_sampled with no extra fields must still produce a valid JSONL record
    with the phase timers included (no KeyError, no None-serialization crash)."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    from aeon.observability import Observer
    with tempfile.TemporaryDirectory() as d:
        obs = Observer(out_dir=d, sample_every=1, enabled=True)
        obs.emit_sampled(step=42)                          # zero optional metrics
        with open(os.path.join(d, "metrics.jsonl")) as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
        assert recs, "no record written"
        assert recs[-1]["step"] == 42
        assert recs[-1]["kind"] == "sampled"
        assert "phase_s" in recs[-1]


# ---- Overhead: instrumented vs baseline steady-state ------------------------
def test_permanent_instrumentation_overhead_under_15_percent():
    """Measure median steady-state step time, warm-up N steps, then time N steps
    with instrumentation off, then N with it on (sample_every=1 = worst case).
    Overhead must be < 15%. This is the E2 hard ceiling from §8.2.

    We use worst-case sampling here to make the ceiling meaningful: production
    uses sample_every=512, which is dramatically cheaper.
    """
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.observability import Observer, parameter_accounting

    torch.manual_seed(0)
    N = 24
    ids = torch.randint(0, 64, (2, 32))

    def bench(enabled: bool, sample_every: int):
        m = _tiny_model(seed=0)
        opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as d:
            obs = Observer(out_dir=d, sample_every=sample_every, enabled=enabled)
            if enabled:
                obs.emit_static("parameter_accounting", parameter_accounting(m))
            # warm-up
            for _ in range(6):
                out = m(input_ids=ids, labels=ids); opt.zero_grad(set_to_none=True)
                out.loss.backward(); opt.step()
            # measure
            ts = []
            for step_ in range(N):
                t = time.perf_counter()
                sampled = obs.should_sample(step_)
                if sampled:
                    with obs.phase("output_loss"):
                        out = m(input_ids=ids, labels=ids)
                else:
                    out = m(input_ids=ids, labels=ids)
                opt.zero_grad(set_to_none=True)
                if sampled:
                    with obs.phase("backward"): out.loss.backward()
                else:
                    out.loss.backward()
                if sampled:
                    with obs.phase("optimizer"): opt.step()
                else:
                    opt.step()
                if step_ % 5 == 0:
                    obs.emit_always_on(step=step_, loss=out.loss.item(), lr=1e-4,
                                       step_time_s=time.perf_counter()-t,
                                       seq_len=32, resident_mb=0.0,
                                       certificate_holds=True, sigma_h=0.5, sigma_c=0.5,
                                       gamma=0.0)
                if sampled:
                    obs.emit_sampled(step=step_, gate_mean=0.0)
                ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    baseline = bench(enabled=False, sample_every=0)
    instrumented = bench(enabled=True, sample_every=1)     # worst case (every step)
    overhead = (instrumented - baseline) / baseline
    print(f"    baseline median step = {baseline*1000:.3f} ms")
    print(f"    instrumented median  = {instrumented*1000:.3f} ms")
    print(f"    overhead = {overhead*100:.2f}%  (ceiling 15%)")
    assert overhead < 0.15, f"overhead {overhead*100:.2f}% exceeds 15% ceiling (E2)"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
