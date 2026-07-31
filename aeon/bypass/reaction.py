"""L3 hidden reaction-coordinate diagnostics.

Three declared candidate coordinates operate on Recursion state r_b
captured by the L1 signal-trace observer. All fitting uses the
calibration partition only; evaluation runs on held-out data. No
coordinate ever enters HybridModel.forward — coordinates are computed
by external evaluators after a diagnostic pass writes RecursionWindow-
Events, and the results never feed back into the model.

Candidates:

  * z_norm(r_b) = ‖r_b - r̄‖_2
  * z_dir(r_b)  = v^T (r_b - r̄)      (v declared or PCA)
  * z_pred(r_b) = Ψ(r_b)              (regularized diagnostic model)

L3 claim ceiling: STRUCTURALLY_IMPLEMENTED (Level 1). Nothing here
supports Level 2+ — an observational claim requires held-out results
against a real corpus AND a calibration lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReactionCoordinateFit:
    """One fitted candidate — locks after fit_calibration()."""

    coordinate_name: str  # "norm" | "dir" | "pred"
    calibration_digest: str  # sha256 over the calibration state vectors
    dimensionality: int
    parameters: Tuple[Tuple[str, object], ...] = field(default_factory=tuple)
    locked: bool = True


def _l2_norm(v: Sequence[float]) -> float:
    return sum(float(x) * float(x) for x in v) ** 0.5


def _mean(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        raise ValueError("empty vector set")
    d = len(vectors[0])
    m = [0.0] * d
    for v in vectors:
        for i, x in enumerate(v):
            m[i] += float(x)
    return [x / len(vectors) for x in m]


def _subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. z_norm — centered L2 norm
# ---------------------------------------------------------------------------
def fit_norm_coordinate(
    calibration_states: Sequence[Sequence[float]],
) -> ReactionCoordinateFit:
    """Fit z_norm: computes calibration mean and returns a fit
    descriptor. Evaluation calls z_norm(fit, r_b)."""
    import hashlib
    mean = _mean(calibration_states)
    payload = "\n".join(",".join(f"{x:.9g}" for x in v)
                          for v in calibration_states).encode("utf-8")
    return ReactionCoordinateFit(
        coordinate_name="norm",
        calibration_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        dimensionality=len(mean),
        parameters=(("mean", tuple(mean)),),
    )


def z_norm(fit: ReactionCoordinateFit, r_b: Sequence[float]) -> float:
    mean = None
    for k, v in fit.parameters:
        if k == "mean":
            mean = v
            break
    if mean is None:
        raise ValueError("norm fit missing mean")
    return _l2_norm(_subtract(r_b, mean))


# ---------------------------------------------------------------------------
# 2. z_dir — linear direction (declared or PCA on calibration)
# ---------------------------------------------------------------------------
def fit_dir_coordinate(
    calibration_states: Sequence[Sequence[float]],
    *,
    declared_direction: Optional[Sequence[float]] = None,
) -> ReactionCoordinateFit:
    """Fit z_dir. If ``declared_direction`` is None, fit the first PCA
    direction using power iteration on the calibration covariance."""
    import hashlib
    mean = _mean(calibration_states)
    if declared_direction is not None:
        direction = list(declared_direction)
    else:
        centered = [_subtract(v, mean) for v in calibration_states]
        d = len(mean)
        v = [1.0] * d  # init
        for _ in range(50):
            new = [0.0] * d
            for x in centered:
                dot = _dot(x, v)
                for i in range(d):
                    new[i] += x[i] * dot
            n = _l2_norm(new) or 1.0
            v = [x / n for x in new]
        direction = v
    payload = "\n".join(",".join(f"{x:.9g}" for x in v)
                          for v in calibration_states).encode("utf-8")
    return ReactionCoordinateFit(
        coordinate_name="dir",
        calibration_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        dimensionality=len(mean),
        parameters=(("mean", tuple(mean)),
                     ("direction", tuple(direction))),
    )


def z_dir(fit: ReactionCoordinateFit, r_b: Sequence[float]) -> float:
    mean = direction = None
    for k, v in fit.parameters:
        if k == "mean":
            mean = v
        elif k == "direction":
            direction = v
    if mean is None or direction is None:
        raise ValueError("dir fit missing mean or direction")
    return _dot(direction, _subtract(r_b, mean))


# ---------------------------------------------------------------------------
# 3. z_pred — regularized diagnostic model
# ---------------------------------------------------------------------------
def fit_pred_coordinate(
    calibration_states: Sequence[Sequence[float]],
    calibration_targets: Sequence[float],
    *,
    ridge_lambda: float = 1.0,
) -> ReactionCoordinateFit:
    """Fit z_pred with ridge regression (least squares + λI). Small
    linear predictor of a declared visible improvement target from
    calibration r_b. Uses only visible calibration targets — no
    peeking at held-out y."""
    import hashlib
    d = len(calibration_states[0])
    # X^T X + λI
    xtx = [[0.0] * d for _ in range(d)]
    xty = [0.0] * d
    for x, y in zip(calibration_states, calibration_targets):
        xf = [float(v) for v in x]
        yf = float(y)
        for i in range(d):
            xty[i] += xf[i] * yf
            for j in range(d):
                xtx[i][j] += xf[i] * xf[j]
    for i in range(d):
        xtx[i][i] += float(ridge_lambda)
    # Solve xtx w = xty via Gaussian elimination.
    w = _solve(xtx, xty)
    payload = "\n".join(",".join(f"{x:.9g}" for x in v)
                          for v in calibration_states).encode("utf-8")
    return ReactionCoordinateFit(
        coordinate_name="pred",
        calibration_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        dimensionality=d,
        parameters=(("weights", tuple(w)),
                     ("ridge_lambda", ridge_lambda)),
    )


def z_pred(fit: ReactionCoordinateFit, r_b: Sequence[float]) -> float:
    weights = None
    for k, v in fit.parameters:
        if k == "weights":
            weights = v
            break
    if weights is None:
        raise ValueError("pred fit missing weights")
    return _dot(weights, r_b)


def _solve(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[pivot] = M[pivot], M[k]
        if abs(M[k][k]) < 1e-12:
            raise ValueError("singular matrix in reaction-coordinate solve")
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


# ---------------------------------------------------------------------------
# Shuffled-state control
# ---------------------------------------------------------------------------
def shuffled_control_score(
    fit: ReactionCoordinateFit,
    states: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    seed: int,
) -> float:
    """L3 requires reporting shuffled-state controls. Returns the
    coordinate's correlation with the target under a per-seed shuffle
    of the states, giving a null distribution for the coordinate."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(states)))
    rng.shuffle(idx)
    shuffled = [states[i] for i in idx]
    if fit.coordinate_name == "norm":
        scores = [z_norm(fit, r) for r in shuffled]
    elif fit.coordinate_name == "dir":
        scores = [z_dir(fit, r) for r in shuffled]
    elif fit.coordinate_name == "pred":
        scores = [z_pred(fit, r) for r in shuffled]
    else:
        raise ValueError(fit.coordinate_name)
    return _pearson(scores, targets)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)
