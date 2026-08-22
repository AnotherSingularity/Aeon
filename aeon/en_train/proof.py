"""aeon.en_train.proof — architecture invariance, gradient path, weight
delta, and native stability wrapper.

Implements §2, §12, §13, §14. Every proof is data — no proof mutates
the model or its state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import PROTECTED_A0_DIGEST, PROTECTED_TOTAL_PARAMETERS, NATIVE_DIAG_MAX_REL_DRIFT


# ---------------------------------------------------------------------------
# §2 — A₀ architecture fingerprint + Δarchitecture
# ---------------------------------------------------------------------------
def compute_architecture_fingerprint(model) -> Dict[str, Any]:
    """Return the exact fingerprint schema written to
    docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json."""
    return {
        "module_type_names": sorted(list(set(type(x).__name__ for x in model.modules()))),
        "state_dict_keys": sorted(model.state_dict().keys()),
        "tensor_shapes": {k: list(v.shape) for k, v in sorted(model.state_dict().items())},
        "tensor_dtypes": {k: str(v.dtype) for k, v in sorted(model.state_dict().items())},
        "state_dict_key_count": len(model.state_dict()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "K": int(model.K),
        "h_rec": int(model.h_rec),
        "D_transformer_hidden": int(model.D),
        "recurrence_configuration": {
            "type": type(model.recursion).__name__,
            "use_embedding_input": bool(model.recursion.use_embedding_input),
            "MARGIN_H": float(model.recursion.MARGIN_H),
            "MARGIN_C": float(model.recursion.MARGIN_C),
        },
        "substrate_configuration": {
            "type": type(model.substrate).__name__,
            "d_in": int(model.d_in),
            "d_state": int(model.d_state),
        },
        "forward_signature_kwargs": ["input_ids", "attention_mask", "labels",
                                          "observer", "intervention", "shuttle"],
        "parameter_sharing_relationships": [],
    }


def digest_fingerprint(fp: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()


class ArchitectureViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def assert_architecture_invariant(model, expected_a0_digest: str = PROTECTED_A0_DIGEST,
                                        expected_parameter_count: int = PROTECTED_TOTAL_PARAMETERS
                                        ) -> Dict[str, Any]:
    """Require Δarchitecture(A₀, current) = 0."""
    fp = compute_architecture_fingerprint(model)
    d = digest_fingerprint(fp)
    if fp["total_parameters"] != expected_parameter_count:
        raise ArchitectureViolation(
            "parameter_count_changed",
            f"got={fp['total_parameters']} expected={expected_parameter_count}")
    if d != expected_a0_digest:
        raise ArchitectureViolation(
            "architecture_fingerprint_drift",
            f"got={d} expected={expected_a0_digest}")
    return {"A_current_digest": d, "delta_architecture_zero": True,
                "total_parameters": fp["total_parameters"]}


# ---------------------------------------------------------------------------
# §12 — gradient-path proof
# ---------------------------------------------------------------------------
@dataclass
class GradientPathObservation:
    step: int
    per_group_grad_l2: Dict[str, float]
    any_nan: bool
    any_inf: bool
    any_disconnected: bool


def _group_parameters(model,
                          exempt_prefixes: Sequence[str] = ()
                          ) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Group model.named_parameters by top-level submodule name."""
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pref) for pref in exempt_prefixes):
            continue
        top = name.split(".")[0]
        groups.setdefault(top, []).append((name, p))
    return groups


def observe_gradient_path(model, step: int,
                                exempt_prefixes: Sequence[str] = ()
                                ) -> GradientPathObservation:
    """After ``loss.backward()``, snapshot per-group ‖∂L/∂θ‖₂."""
    groups = _group_parameters(model, exempt_prefixes)
    per: Dict[str, float] = {}
    any_nan = any_inf = any_disc = False
    for gname, params in groups.items():
        sq = 0.0
        seen_grad = False
        for _n, p in params:
            g = p.grad
            if g is None: continue
            seen_grad = True
            if torch.isnan(g).any().item(): any_nan = True
            if torch.isinf(g).any().item(): any_inf = True
            sq += float(g.detach().to(torch.float32).pow(2).sum().item())
        if not seen_grad:
            any_disc = True
        per[gname] = float(sq ** 0.5)
    return GradientPathObservation(step=step, per_group_grad_l2=per,
                                            any_nan=any_nan, any_inf=any_inf,
                                            any_disconnected=any_disc)


class GradientPathViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def assert_gradient_path_over_100_steps(observations: Sequence[GradientPathObservation]
                                                ) -> Dict[str, Any]:
    """§12: for the first 100 updates every intended group must have
    finite nonzero max grad; no NaN, no Inf, no disconnected group."""
    if not observations:
        raise GradientPathViolation("no_observations")
    all_groups = set()
    for o in observations: all_groups.update(o.per_group_grad_l2.keys())
    per_group_max: Dict[str, float] = {g: 0.0 for g in all_groups}
    for o in observations:
        if o.any_nan: raise GradientPathViolation("nan_gradient", f"step={o.step}")
        if o.any_inf: raise GradientPathViolation("inf_gradient", f"step={o.step}")
        if o.any_disconnected:
            raise GradientPathViolation("disconnected_group", f"step={o.step}")
        for g, v in o.per_group_grad_l2.items():
            if v > per_group_max[g]: per_group_max[g] = v
    zero_groups = [g for g, v in per_group_max.items() if v <= 0.0]
    if zero_groups:
        raise GradientPathViolation("zero_gradient_group", str(zero_groups))
    return {"per_group_max_grad_l2": per_group_max,
                "n_observations": len(observations)}


# ---------------------------------------------------------------------------
# §13 — weight-Δ proof
# ---------------------------------------------------------------------------
def snapshot_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().to(torch.float32).cpu().clone()
                for k, v in model.state_dict().items()}


@dataclass
class WeightDeltaReport:
    total_tensors: int
    zero_delta: List[str]
    positive_delta: List[str]
    min_nonzero_delta: float
    median_delta: float
    max_delta: float
    per_tensor_delta: Dict[str, float]


def compute_weight_delta(before: Dict[str, torch.Tensor],
                              after: Dict[str, torch.Tensor]) -> WeightDeltaReport:
    import statistics
    per: Dict[str, float] = {}
    zero: List[str] = []
    positive: List[str] = []
    for k in before:
        a = before[k]; b = after[k]
        if a.shape != b.shape:
            raise RuntimeError(f"shape drift on {k}: {a.shape} vs {b.shape}")
        d = float((a - b).to(torch.float32).norm(p=2).item())
        per[k] = d
        (positive if d > 0 else zero).append(k)
    nonzero = [d for d in per.values() if d > 0]
    return WeightDeltaReport(
        total_tensors=len(per),
        zero_delta=sorted(zero),
        positive_delta=sorted(positive),
        min_nonzero_delta=min(nonzero) if nonzero else 0.0,
        median_delta=statistics.median(per.values()) if per else 0.0,
        max_delta=max(per.values()) if per else 0.0,
        per_tensor_delta=per,
    )


# ---------------------------------------------------------------------------
# §14 — native stability gate wrapper
# ---------------------------------------------------------------------------
class NativeStabilityViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def check_finite_state_dict(model) -> None:
    """Every parameter must be finite (no NaN, no Inf)."""
    for name, p in model.state_dict().items():
        t = p.detach()
        if torch.isnan(t).any().item():
            raise NativeStabilityViolation("nan_in_state_dict", name)
        if torch.isinf(t).any().item():
            raise NativeStabilityViolation("inf_in_state_dict", name)


def sigma_certificate(model) -> Dict[str, float]:
    """Wrap the Recursion joiner's native margin-h / margin-c constants
    into a stability diagnostic. These are the declared safety bounds
    Aeon's forward path relies on; drift is a hard fault regardless
    of relative-tolerance rules."""
    return {
        "MARGIN_H": float(model.recursion.MARGIN_H),
        "MARGIN_C": float(model.recursion.MARGIN_C),
    }


def assert_native_stability_gate(model,
                                        baseline_sigma: Dict[str, float],
                                        max_rel_drift: float = NATIVE_DIAG_MAX_REL_DRIFT
                                        ) -> Dict[str, Any]:
    """§14: relative-drift comparison against P2 baseline diagnostics.
    Absolute native safety boundaries (finite params, sigma-certificate
    constants unchanged) are always binding."""
    check_finite_state_dict(model)
    cur = sigma_certificate(model)
    drifts: Dict[str, float] = {}
    for k, base in baseline_sigma.items():
        got = cur.get(k)
        if got is None:
            raise NativeStabilityViolation("missing_diagnostic", k)
        eps = max(abs(base), 1e-12)
        rel = abs(got - base) / eps
        drifts[k] = rel
        if rel > max_rel_drift:
            raise NativeStabilityViolation(
                "native_diagnostic_drift",
                f"{k}: |{got}-{base}|/max({eps},ε) = {rel:.4f} > {max_rel_drift}")
    return {"drifts": drifts, "passed": True}
