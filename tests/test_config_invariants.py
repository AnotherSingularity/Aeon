"""
E1 — Configuration invariants.

Directive §3.4/§3.5 fix K=16, Recursion=fp32, certificate on, single broadcast.
These must not silently drift through ordinary configuration. This suite scans
every YAML config in configs/ and asserts the protected constants.

Broader anti-drift: if someone adds a new config, it too must obey these rules,
or the E1 gate fails. That is what "cannot be changed accidentally" means.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _configs():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs")
    return sorted(glob.glob(os.path.join(root, "*.yaml")))


def test_every_config_pins_K_to_16():
    import yaml
    for path in _configs():
        cfg = yaml.safe_load(open(path))
        K = cfg.get("model", {}).get("K")
        assert K == 16, f"{path}: K={K!r} — MUST be 16 (P-K16)"


def test_every_config_declares_certificate_margins_in_safe_range():
    """margin_h, margin_c must be in (0, 1) — the certificate constants."""
    import yaml
    for path in _configs():
        cfg = yaml.safe_load(open(path))
        mh = cfg["model"]["margin_h"]
        mc = cfg["model"]["margin_c"]
        assert 0.0 < mh < 1.0, f"{path}: margin_h={mh} outside (0,1)"
        assert 0.0 < mc < 1.0, f"{path}: margin_c={mc} outside (0,1)"


def test_no_config_key_hints_at_adaptive_clock():
    """No K-related override key that would imply adaptive/learned/per-token clock."""
    import yaml
    banned = ("K_adaptive", "adaptive_K", "K_learned", "per_token_recursion",
              "K_min", "K_max", "K_schedule", "K_entropy", "K_dynamic")
    for path in _configs():
        cfg = yaml.safe_load(open(path))
        mcfg = cfg.get("model", {})
        for k in mcfg:
            assert k not in banned, f"{path}: banned config key {k!r} — §3.4"


def test_no_config_key_hints_at_dual_broadcast():
    """No config key that would enable a J_S/J_T dual-broadcast head (§3.3)."""
    import yaml
    banned = ("substrate_broadcast", "transformer_broadcast",
              "dual_broadcast", "bcast_sub", "bcast_trans",
              "J_S", "J_T", "separate_feedback_heads")
    for path in _configs():
        cfg = yaml.safe_load(open(path))
        # search anywhere in the config, not just top level
        def walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    assert k not in banned, f"{path}: banned key {k!r} (dual-head § 3.3)"
                    walk(v)
            elif isinstance(d, list):
                for x in d: walk(x)
        walk(cfg)


def test_dtype_key_is_one_of_permitted_compute_dtypes():
    """Compute dtype is bf16/f16/f32. The Recursion-fp32 rule is enforced by the
    training script post-cast (P-fp32-rec), so 'model.dtype' is the compute dtype
    for everything ELSE, not Recursion."""
    import yaml
    permitted = {"bfloat16", "float16", "float32"}
    for path in _configs():
        cfg = yaml.safe_load(open(path))
        dtype = cfg["model"].get("dtype", "bfloat16")
        assert dtype in permitted, f"{path}: dtype={dtype!r} not in {permitted}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
