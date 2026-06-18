"""
aeon/substrate/conformance.py — the substrate conformance utility.

`verify_substrate(cell)` is **the conformance check to run when introducing a
new substrate** (a new cell of any kind). It runs the
full port contract against an arbitrary cell and returns a structured
`ConformanceReport` (pass / fail / skip per check, with diagnostics on failure).

It is the single source of truth for "does this thing satisfy the port?" — the
tests in tests/ are thin shells over it, not parallel copies.

Design:
  * Port-contract checks live here and are framework-free where possible. Static
    checks (capability negotiation, decay-is-read-only, contract attributes) run
    with no torch. Execution checks (shapes, bounded output, advertised reads)
    need to run the cell and are guarded — they `skip` when torch is
    unavailable or the cell is not a torch module, mirroring the test pattern.
  * Per-cell structural checks register via a `CONFORMANCE_CHECKS` class
    attribute on the cell (or the `extra_checks` argument), so cell-specific
    guards (e.g. the vector cell's anti-drift AST check) extend the suite without forking
    this utility. A registered check is `callable(cell)`; it passes by
    returning, fails by raising, and skips by raising `SkipCheck`.
"""
from __future__ import annotations

import ast
import importlib.util
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .port import (
    SubstratePort,
    CapabilityError,
    ALL_CAPABILITIES,
    MATRIX_READ,
    DECAY_CONTROL,
    ASSOC_WRITE,
    PER_LAYER_READ,
)

# capability -> the method that realises it
_CAP_METHOD = {
    MATRIX_READ: "read_matrix",
    DECAY_CONTROL: "read_decay",
    ASSOC_WRITE: "assoc_write",
    PER_LAYER_READ: "read_per_layer",
}


class SkipCheck(Exception):
    """Raised by a check to mark itself skipped (e.g. torch unavailable)."""


@dataclass
class CheckResult:
    name: str
    status: str            # "pass" | "fail" | "skip"
    detail: str = ""

    def __str__(self) -> str:
        mark = {"pass": "ok  ", "fail": "FAIL", "skip": "skip"}[self.status]
        tail = f"  — {self.detail}" if self.detail else ""
        return f"  [{mark}] {self.name}{tail}"


@dataclass
class ConformanceReport:
    cell: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status != "fail" for r in self.results)

    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        n_pass = sum(r.status == "pass" for r in self.results)
        n_fail = sum(r.status == "fail" for r in self.results)
        n_skip = sum(r.status == "skip" for r in self.results)
        head = (f"ConformanceReport({self.cell}): "
                f"{'OK' if self.ok else 'FAILED'} "
                f"[{n_pass} pass, {n_fail} fail, {n_skip} skip]")
        return "\n".join([head, *map(str, self.results)])


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


# ---------------------------------------------------------------------------
# reusable per-cell check factory (used by cells to register structural checks)
# ---------------------------------------------------------------------------
def make_ast_drift_check(
    module_name: str,
    forbidden: Iterable[str],
    required: Iterable[str],
    *,
    name: str | None = None,
) -> Callable[[Any], None]:
    """Build a torch-free structural check that parses ``module_name``'s source
    (located via importlib, *without importing it*, so no torch is pulled in)
    and asserts no `forbidden` identifiers appear and all `required` ones do.
    Comments/docstrings are ignored — only real identifiers are inspected."""
    forbidden = {f.lower() for f in forbidden}
    required = set(required)

    def check(cell: Any = None) -> None:
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin:
            raise SkipCheck(f"cannot locate source for {module_name}")
        with open(spec.origin) as fh:
            tree = ast.parse(fh.read())
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.alias):
                names.add(node.name)
                if node.asname:
                    names.add(node.asname)
        hit = forbidden & {n.lower() for n in names}
        assert not hit, f"forbidden identifiers in {module_name}: {sorted(hit)}"
        missing = required - names
        assert not missing, f"missing required identifiers in {module_name}: {sorted(missing)}"

    check.__name__ = name or f"ast_drift_check[{module_name}]"
    return check


# ---------------------------------------------------------------------------
# the conformance suite
# ---------------------------------------------------------------------------
def verify_substrate(
    cell: SubstratePort,
    *,
    batch_size: int = 2,
    extra_checks: Sequence[Callable[[Any], Any]] = (),
) -> ConformanceReport:
    """Run the full substrate port conformance suite against `cell`.

    THIS IS THE CHECK TO RUN WHEN INTRODUCING A NEW SUBSTRATE. Returns a
    `ConformanceReport`; `report.ok` is True iff no check failed (skips are not
    failures). Truthiness of the report mirrors `.ok`.
    """
    report = ConformanceReport(cell=type(cell).__name__)

    def record(name: str, fn: Callable[[], Any]) -> None:
        try:
            detail = fn()
            report.results.append(CheckResult(name, "pass", detail or ""))
        except SkipCheck as exc:
            report.results.append(CheckResult(name, "skip", str(exc)))
        except Exception as exc:  # noqa: BLE001 - diagnostics captured below
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            report.results.append(
                CheckResult(name, "fail", f"{type(exc).__name__}: {exc} | {tb}")
            )

    B = batch_size
    caps = cell.capabilities()

    # ---- static checks (torch-free) ---------------------------------------
    def _contract_attrs():
        for attr, typ in (("d_in", int), ("d_state", int)):
            assert isinstance(getattr(cell, attr), typ), f"{attr} must be {typ.__name__}"
        ob = cell.output_bound
        assert isinstance(ob, (int, float)) and ob > 0, "output_bound must be a positive number"
        return f"d_in={cell.d_in}, d_state={cell.d_state}, output_bound={ob}"

    def _capabilities_subset():
        assert set(caps) <= ALL_CAPABILITIES, f"unknown capabilities: {set(caps) - ALL_CAPABILITIES}"
        return f"advertises {sorted(caps) or '[]'}"

    def _has_require_agree():
        for cap in ALL_CAPABILITIES:
            assert cell.has(cap) == (cap in caps), f"has({cap}) disagrees with capabilities()"
            if cell.has(cap):
                cell.require(cap)  # must not raise
            else:
                try:
                    cell.require(cap)
                except CapabilityError:
                    pass
                else:
                    raise AssertionError(f"require({cap}) should raise for unadvertised cap")
        return "has()/require()/capabilities() agree"

    def _advertised_overridden():
        for cap in caps:
            method = _CAP_METHOD[cap]
            if getattr(type(cell), method) is getattr(SubstratePort, method):
                raise AssertionError(f"{cap} advertised but {method}() not overridden")
        return "advertised optional ports are implemented"

    def _gating_unadvertised():
        for cap in ALL_CAPABILITIES - set(caps):
            method = _CAP_METHOD[cap]
            fn = getattr(cell, method)
            args = (None, None) if cap == ASSOC_WRITE else ()
            try:
                fn(*args)
            except CapabilityError:
                continue
            except Exception as exc:  # wrong error type
                raise AssertionError(f"{method}() raised {type(exc).__name__}, expected CapabilityError")
            raise AssertionError(f"{method}() did not gate on unadvertised {cap}")
        return "unadvertised ports gate with CapabilityError"

    def _no_decay_mutator():
        assert not hasattr(SubstratePort, "set_decay"), "port exposes a set_decay mutator"
        assert "set_decay" not in dir(type(cell)), f"{type(cell).__name__} exposes set_decay"
        return "decay is read-only (no set_decay anywhere)"

    record("contract_attributes", _contract_attrs)
    record("capabilities_subset", _capabilities_subset)
    record("has_require_agree", _has_require_agree)
    record("advertised_ports_implemented", _advertised_overridden)
    record("unadvertised_ports_gated", _gating_unadvertised)
    record("decay_read_only", _no_decay_mutator)

    # ---- execution checks (need torch + a torch cell) ---------------------
    torch = _torch()
    runnable = torch is not None and isinstance(cell, torch.nn.Module)
    skip_reason = "torch unavailable or cell is not a torch.nn.Module"

    def _exec(fn):
        if not runnable:
            raise SkipCheck(skip_reason)
        return fn()

    def _required_tier():
        def go():
            cell.reset(B)
            out = cell.step(torch.randn(B, cell.d_in))
            assert tuple(out.shape) == (B, cell.d_state), f"step shape {tuple(out.shape)}"
            assert tuple(cell.read().shape) == (B, cell.d_state), "read shape"
            cell.write(torch.randn(B, cell.d_state))
            out2 = cell.step(torch.randn(B, cell.d_in))
            assert tuple(out2.shape) == (B, cell.d_state), "post-write step shape"
            return f"shapes (B={B}, d_state={cell.d_state}) hold through reset/step/read/write"
        return _exec(go)

    def _bounded_output():
        def go():
            cell.reset(B)
            mx = 0.0
            for _ in range(8):
                out = cell.step(torch.randn(B, cell.d_in) * 50.0)
                assert torch.isfinite(out).all(), "non-finite output"
                mx = max(mx, out.abs().max().item())
            assert mx <= cell.output_bound + 1e-5, f"max|out|={mx} > output_bound={cell.output_bound}"
            return f"max|out|={mx:.4f} <= output_bound={cell.output_bound}"
        return _exec(go)

    def _advertised_reads():
        def go():
            cell.reset(B)
            cell.step(torch.randn(B, cell.d_in))
            notes = []
            if cell.has(DECAY_CONTROL):
                d = cell.read_decay()
                assert d is not None, "read_decay returned None"
                notes.append("read_decay ok")
            if cell.has(MATRIX_READ):
                S = cell.read_matrix()
                assert S is not None and hasattr(S, "shape"), "read_matrix returned no tensor"
                assert S.shape[0] == B, "read_matrix batch dim"
                notes.append(f"read_matrix {tuple(S.shape)}")
            if cell.has(PER_LAYER_READ):
                layers = list(cell.read_per_layer())
                assert layers, "read_per_layer empty"
                notes.append(f"read_per_layer x{len(layers)}")
            return ", ".join(notes) or "no read-style optional ports"
        return _exec(go)

    record("required_tier_shapes", _required_tier)
    record("bounded_output", _bounded_output)
    record("advertised_reads_callable", _advertised_reads)

    # ---- per-cell registered checks (extension hook) ----------------------
    registered = list(getattr(type(cell), "CONFORMANCE_CHECKS", ())) + list(extra_checks)
    for chk in registered:
        name = getattr(chk, "__name__", "custom_check")
        record(name, (lambda c=chk: c(cell)))

    return report
