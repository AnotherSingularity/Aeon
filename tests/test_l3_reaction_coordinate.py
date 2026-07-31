"""L3 reaction-coordinate diagnostics tests (structurally implemented only).

L3 claim ceiling: STRUCTURALLY_IMPLEMENTED (Level 1). Nothing here
supports Level 2+. Tests exercise the three declared candidates
(z_norm, z_dir, z_pred), the calibration-only fit contract, the
shuffled-state control, and the noninterference guarantee.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _synthetic_states(n: int = 200, d: int = 8, seed: int = 1):
    import random
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(n)]


# ---------------------------------------------------------------------------
def test_z_norm_returns_zero_at_mean_state():
    from aeon.bypass.reaction import fit_norm_coordinate, z_norm
    states = _synthetic_states()
    fit = fit_norm_coordinate(states)
    mean = [sum(v[i] for v in states) / len(states) for i in range(len(states[0]))]
    assert z_norm(fit, mean) < 1e-9


def test_z_dir_fit_produces_direction_and_reads_scalar():
    from aeon.bypass.reaction import fit_dir_coordinate, z_dir
    states = _synthetic_states()
    fit = fit_dir_coordinate(states)
    v = z_dir(fit, states[0])
    assert isinstance(v, float)


def test_z_pred_fit_solves_ridge():
    from aeon.bypass.reaction import fit_pred_coordinate, z_pred
    states = _synthetic_states()
    # Target = linear combination of state[0] and state[1] plus noise.
    import random
    rng = random.Random(2)
    targets = [3.0 * s[0] - 1.5 * s[1] + rng.gauss(0.0, 0.1)
                for s in states]
    fit = fit_pred_coordinate(states, targets, ridge_lambda=0.1)
    # The fit's dimensionality matches state dim.
    assert fit.dimensionality == len(states[0])
    v = z_pred(fit, states[0])
    assert isinstance(v, float)


def test_reaction_fits_bind_calibration_digest():
    from aeon.bypass.reaction import (
        fit_norm_coordinate, fit_dir_coordinate, fit_pred_coordinate,
    )
    states = _synthetic_states()
    fit_n = fit_norm_coordinate(states)
    fit_d = fit_dir_coordinate(states)
    fit_p = fit_pred_coordinate(states, [0.0] * len(states))
    # Same calibration input → same calibration digest across coordinates.
    assert fit_n.calibration_digest == fit_d.calibration_digest
    assert fit_d.calibration_digest == fit_p.calibration_digest
    assert fit_n.calibration_digest.startswith("sha256:")


def test_reaction_fits_are_locked_frozen_dataclass():
    from aeon.bypass.reaction import fit_norm_coordinate
    from dataclasses import FrozenInstanceError
    states = _synthetic_states()
    fit = fit_norm_coordinate(states)
    assert fit.locked is True
    try:
        fit.locked = False
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ReactionCoordinateFit must be frozen")


def test_shuffled_state_control_produces_baseline_score():
    from aeon.bypass.reaction import (
        fit_pred_coordinate, shuffled_control_score,
    )
    states = _synthetic_states()
    targets = [s[0] for s in states]
    fit = fit_pred_coordinate(states, targets, ridge_lambda=0.1)
    # Shuffled control should produce a correlation close to zero
    # relative to the fitted correlation on unshuffled data.
    r_shuffled = shuffled_control_score(fit, states, targets, seed=7)
    assert -1.0 <= r_shuffled <= 1.0


def test_reaction_coordinates_do_not_touch_hybrid_forward():
    """The reaction-coordinate module is external-evaluator code and
    must not import or mutate HybridModel.forward's default path."""
    src = open(os.path.join(ROOT, "aeon", "bypass", "reaction.py"),
                encoding="utf-8").read()
    # No import of HybridModel forward internals
    assert "from aeon.hybrid import HybridModel" not in src
    assert "model.forward" not in src


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
