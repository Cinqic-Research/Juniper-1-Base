"""Out-of-process code checking, coding-contamination detection and coding-scorer audits (j1.codecheck.v1).

Three parts share one execution path:

* **I/O checker** (:func:`run_io`, :func:`score_python_io`). The candidate runs in the bubblewrap sandbox
  (:mod:`j1data.sandbox`) in a process that receives argument tuples and writes one JSON line per call
  (a tagged, closed value encoding, never pickle). Expected values come from literals in the specification
  or from the trusted reference run in a *separate* sandboxed process. Comparison happens in the host
  process, which never executes candidate code. The emitted values are untrusted observable output,
  not authenticated function returns: candidate frame inspection can modify the encoder or reporter.
  This is a functional diagnostic for non-adversarial programs, never an adversarial reward or
  qualification oracle. Expected outputs remain in the host and are never sent to the candidate.
* **Functional-twin detector** (:func:`functional_twins`). Task pairs are compared on statement text
  (word and character n-grams, numbers normalized), reference structure (canonical AST, identifiers renamed,
  constants normalized) and by sandboxed cross-execution of both references on each task's own test
  inputs and on type-preserving mutations of them. Finite cross-execution is evidence of overlap, not a
  proof of equivalence.
* **Scorer audit** (``audit`` CLI): deliberately wrong programs scored by the upstream asserts and by
  stronger I/O tests (EvalPlus inputs used for evaluation only, plus mutated inputs whose expected values
  come from the trusted reference).

Untrusted code (dataset references, model outputs, mutants) only ever runs through
:func:`j1data.sandbox.run_python`; there is no unsandboxed fallback.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import json
import math
import random
import re
import secrets
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from j1data import sandbox, sftscore

VERSION = "j1.codecheck.v1"
EXECUTION_POLICY = "j1.codecheck.emitted-output-diagnostic.v2"
LIMITATION = (
    "untrusted emitted-output diagnostic; not authenticated function returns; no adversarial qualification"
)


class UnqualifiedOracle(sftscore.SpecError):
    """This Python execution path is not qualified for the requested adversarial use."""


def _require_diagnostic(purpose: str) -> None:
    if purpose != "functional_diagnostic":
        raise UnqualifiedOracle(f"{purpose!r} refused: candidate interpreter can forge reports; {LIMITATION}")


# --- value encoding -------------------------------------------------------------------------------
#
# JSON keeps None/bool/int/float/str apart natively; every JSON array is a tagged container:
#   ["L", ...] list, ["T", ...] tuple, ["S", ...] set, ["F", ...] frozenset (elements sorted by their
#   canonical JSON), ["D", [k, v], ...] dict (pairs sorted by key JSON), ["B", hex] bytes,
#   ["C", re, im] complex, ["H", sha256, n] a value whose canonical JSON (n chars) is too long to inline.
# Values of any other type are not serializable and never compare equal. Subclasses of the built-in
# types are read through the base type's own slots, so an overridden __eq__/__iter__ has no effect.

INLINE_CHARS = 1500
NODE_BUDGET = 200_000
MAX_INT_BITS = 40_000


class Unserializable(Exception):
    """The value has a type outside the closed encoding."""


def _enc(v: Any, budget: list[int], depth: int = 0) -> Any:
    budget[0] -= 1
    if budget[0] < 0 or depth > 200:
        raise OverflowError("value too large")
    if v is None or v is True or v is False:
        return v
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        n = int.__index__(v)
        if n.bit_length() > MAX_INT_BITS:
            raise OverflowError("integer too large")
        return n
    if isinstance(v, float):
        return float.__float__(v)
    if isinstance(v, str):
        s = str.__str__(v)
        budget[0] -= len(s) // 16
        if budget[0] < 0:
            raise OverflowError("value too large")
        return s
    if isinstance(v, (list, tuple)):
        items = list(list.__iter__(v)) if isinstance(v, list) else list(tuple.__iter__(v))
        return ["L" if isinstance(v, list) else "T", *[_enc(x, budget, depth + 1) for x in items]]
    if isinstance(v, (set, frozenset)):
        elems = [
            _enc(x, budget, depth + 1)
            for x in (set.__iter__(v) if isinstance(v, set) else frozenset.__iter__(v))
        ]
        elems.sort(key=lambda e: json.dumps(e, separators=(",", ":")))
        return ["S" if isinstance(v, set) else "F", *elems]
    if isinstance(v, dict):
        pairs = [[_enc(k, budget, depth + 1), _enc(x, budget, depth + 1)] for k, x in dict.items(v)]
        pairs.sort(key=lambda p: json.dumps(p[0], separators=(",", ":")))
        return ["D", *pairs]
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        budget[0] -= len(b) // 8
        if budget[0] < 0:
            raise OverflowError("value too large")
        return ["B", b.hex()]
    if isinstance(v, complex):
        return ["C", complex.__complex__(v).real, complex.__complex__(v).imag]
    raise Unserializable(type(v).__name__)


def _dec(e: Any) -> Any:
    if not isinstance(e, list):
        return e
    tag, rest = e[0], e[1:]
    if tag == "L":
        return [_dec(x) for x in rest]
    if tag == "T":
        return tuple(_dec(x) for x in rest)
    if tag == "S":
        return {_dec(x) for x in rest}
    if tag == "F":
        return frozenset(_dec(x) for x in rest)
    if tag == "D":
        return {_dec(k): _dec(v) for k, v in rest}
    if tag == "B":
        return bytes.fromhex(rest[0])
    if tag == "C":
        return complex(rest[0], rest[1])
    raise ValueError(f"unknown tag {tag!r}")


def _loose(e: Any, unordered: bool = False) -> Any:
    """Lenient form used only for twin screening: list == tuple, set == frozenset, bool == int, floats to six
    significant digits (integral floats become ints); ``unordered`` also ignores the order of a top-level
    sequence (for tasks whose own asserts compare ``set(...)`` or ``sorted(...)``)."""
    if isinstance(e, bool):
        return int(e)
    if isinstance(e, float):
        if math.isfinite(e) and e == int(e) and abs(e) < 2**53:
            return int(e)
        return float(f"{e:.6g}") if math.isfinite(e) else e
    if not isinstance(e, list):
        return e
    tag, rest = e[0], e[1:]
    if tag in ("L", "T", "S", "F"):
        items = [_loose(x) for x in rest]
        if unordered or tag in ("S", "F"):
            items.sort(key=lambda x: json.dumps(x, separators=(",", ":")))
        return ["S" if tag in ("S", "F") else "L", *items]
    if tag == "D":
        pairs = [[_loose(k), _loose(v)] for k, v in rest]
        pairs.sort(key=lambda p: json.dumps(p[0], separators=(",", ":")))
        return ["D", *pairs]
    return e


def _canon(e: Any) -> str:
    return json.dumps(e, separators=(",", ":"), allow_nan=True)


def encode_value(v: Any) -> Any:
    """Tagged encoding of a trusted host value (raises :class:`Unserializable` / ``OverflowError``)."""
    return _enc(v, [NODE_BUDGET])


def decode_value(e: Any) -> Any:
    return _dec(e)


def safe_literal(node: ast.AST | str) -> Any:
    """Evaluate a literal expression: ``ast.literal_eval`` plus ``inf``/``nan`` names, ``float('inf')``,
    and ``set()``/``frozenset(...)``/``tuple(...)``/``list(...)`` of literals. Raises ``ValueError``."""
    if isinstance(node, str):
        node = ast.parse(node.strip(), mode="eval").body

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Tuple):
            return tuple(ev(x) for x in n.elts)
        if isinstance(n, ast.List):
            return [ev(x) for x in n.elts]
        if isinstance(n, ast.Set):
            return {ev(x) for x in n.elts}
        if isinstance(n, ast.Dict):
            if any(k is None for k in n.keys):
                raise ValueError("dict unpacking")
            return {ev(k): ev(v) for k, v in zip(n.keys, n.values, strict=True) if k is not None}
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            x = ev(n.operand)
            if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
                return -x if isinstance(n.op, ast.USub) else +x
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(b, complex) and isinstance(a, (int, float)):
                return a + b if isinstance(n.op, ast.Add) else a - b
        if isinstance(n, ast.Name) and n.id in ("inf", "nan"):
            return float(n.id)
        if (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id in ("math", "np", "numpy")
            and n.attr in ("inf", "nan", "pi", "e")
        ):
            return float(getattr(math, n.attr))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            args = [ev(a) for a in n.args]
            if (
                n.func.id == "float"
                and len(args) == 1
                and isinstance(args[0], str)
                and args[0].strip().lower().lstrip("+-") in ("inf", "infinity", "nan")
            ):
                return float(args[0])
            if n.func.id in ("set", "frozenset", "tuple", "list") and len(args) <= 1:
                ctor: Any = {"set": set, "frozenset": frozenset, "tuple": tuple, "list": list}[n.func.id]
                if not args or isinstance(args[0], (list, tuple, set, frozenset, str)):
                    return ctor(*args)
        raise ValueError(f"not a literal: {ast.dump(n)[:80]}")

    return ev(node)


# --- comparison (host side; candidate output is data) ----------------------------------------------


@dataclass(frozen=True)
class FloatTol:
    rel: float = 1e-9
    abs: float = 0.0

    @staticmethod
    def parse(spec: Any) -> FloatTol:
        if spec is None:
            return FloatTol()
        if isinstance(spec, (int, float)):
            return FloatTol(rel=0.0, abs=float(spec))
        return FloatTol(rel=float(spec.get("rel", 0.0)), abs=float(spec.get("abs", 0.0)))


def _num_equal(a: int | float, b: int | float, tol: FloatTol) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return abs(fa - fb) <= tol.abs + tol.rel * abs(fb)
    return a == b


def values_equal(got: Any, want: Any, tol: FloatTol | None = None, compare: str = "exact") -> bool:
    """Compare two encoded values with Python ``==`` semantics for numbers (bool/int/float by value, floats
    within ``tol``) and strict container types (a list never equals a tuple). ``compare`` = ``set`` or
    ``multiset`` makes the top-level sequence order-insensitive (as an upstream ``set(f(x)) == ...`` or
    ``sorted(f(x)) == ...`` assert does)."""
    tol = tol or FloatTol()
    if compare in ("set", "multiset"):
        if not (isinstance(got, list) and isinstance(want, list)):
            return False
        if got[:1] == ["H"] or want[:1] == ["H"] or got[0] not in "LTSF" or want[0] not in "LTSF":
            return got == want
        g = sorted(_canon(x) for x in got[1:])
        w = sorted(_canon(x) for x in want[1:])
        return (sorted(set(g)) == sorted(set(w))) if compare == "set" else g == w
    return _eq(got, want, tol, 0)


def _eq(a: Any, b: Any, tol: FloatTol, depth: int) -> bool:
    if depth > 400:
        return False
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num):
        return _num_equal(a, b, tol)
    if isinstance(a, list) and isinstance(b, list):
        if not a or not b or a[0] != b[0] or len(a) != len(b):
            return False
        tag = a[0]
        if tag == "H":
            return bool(a[1] == b[1])
        if tag == "D":
            ka = {_canon(p[0]): p[1] for p in a[1:]}
            kb = {_canon(p[0]): p[1] for p in b[1:]}
            return ka.keys() == kb.keys() and all(_eq(ka[k], kb[k], tol, depth + 1) for k in ka)
        if tag in ("S", "F"):
            return a == b
        return all(_eq(x, y, tol, depth + 1) for x, y in zip(a[1:], b[1:], strict=True))
    return type(a) is type(b) and bool(a == b)


# --- sandboxed runner -------------------------------------------------------------------------------

_RUNNER = """from __future__ import annotations
import sys, json, copy, signal, hashlib, os, math
sys.set_int_max_str_digits(0)
sys.setrecursionlimit({reclimit})
{helpers}
_NONCE = {nonce!r}
_FD = os.dup(1)
_WRITE = os.write


def _emit(i, rec):
    line = _NONCE + " " + str(i) + " " + json.dumps(rec, separators=(",", ":"), allow_nan=True) + "\\n"
    _WRITE(_FD, line.encode("utf-8", "surrogatepass"))


class _Sink:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _name(e):
    try:
        return str(type.__getattribute__(type(e), "__name__"))[:80]
    except BaseException:
        return "Exception"


def _digest(s):
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def _main():
    idx = {indices!r}
    inputs = [[_dec(a) for a in x] for x in json.loads({inputs!r})]
    mode = {mode!r}
    ns = {{"__name__": "candidate", "__builtins__": __builtins__}}
    sys.stdout = _Sink()
    sys.stderr = _Sink()
    signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, {load_seconds})
        exec(compile({code!r}, "<candidate>", "exec"), ns)
        fn = ns[{entry!r}]
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for i in idx:
            _emit(i, ["load", _name(e)])
        return
    for i, args in zip(idx, inputs):
        rec = None
        try:
            a = copy.deepcopy(args)
            signal.setitimer(signal.ITIMER_REAL, {call_seconds})
            r = fn(*a)
            e = _enc(r, [{node_budget}])
            signal.setitimer(signal.ITIMER_REAL, 0)
            s = json.dumps(e, separators=(",", ":"), allow_nan=True)
            if mode == "digest":
                lo = json.dumps(_loose(e), separators=(",", ":"), allow_nan=True)
                un = json.dumps(_loose(e, True), separators=(",", ":"), allow_nan=True)
                rec = ["ok", ["H", _digest(s)[:24], _digest(lo)[:24], _digest(un)[:24]]]
            elif len(s) > {inline}:
                rec = ["ok", ["H", _digest(s), len(s)]]
            else:
                rec = ["ok", e]
        except _Timeout:
            rec = ["timeout"]
        except Unserializable as u:
            rec = ["unser", str(u)[:80]]
        except OverflowError:
            rec = ["big"]
        except BaseException as ex:
            rec = ["exc", _name(ex)]
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException:
                pass
        _emit(i, rec)


_main()
"""

_HELPERS = "\n\n".join(
    [
        "class Unserializable(Exception):\n    pass",
        f"INLINE_CHARS = {INLINE_CHARS}\nNODE_BUDGET = {NODE_BUDGET}\nMAX_INT_BITS = {MAX_INT_BITS}",
        inspect.getsource(_enc),
        inspect.getsource(_dec),
        inspect.getsource(_loose),
    ]
)


@dataclass(frozen=True)
class Outcome:
    """Result of one call: kind is ``ok`` (value set), ``exc``, ``timeout``, ``unser``, ``big``, ``load``
    (the code did not load or define the entry point), ``crash`` (the process died before reporting) or
    ``dup`` (more than one report for the call)."""

    kind: str
    value: Any = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


@dataclass(frozen=True)
class RunConfig:
    call_seconds: float = 2.0
    load_seconds: float = 5.0
    recursion_limit: int = 3000
    limits: sandbox.Limits = field(
        default_factory=lambda: sandbox.Limits(cpu_seconds=30, wall_seconds=40.0, address_space_bytes=1 << 30)
    )
    chunk_chars: int = 6000


def _program(
    code: str, entry: str, idx: list[int], enc_inputs: list[Any], mode: str, cfg: RunConfig
) -> tuple[str, str]:
    nonce = secrets.token_hex(12)
    prog = _RUNNER.format(
        reclimit=cfg.recursion_limit,
        helpers=_HELPERS,
        nonce=nonce,
        indices=idx,
        inputs=json.dumps(enc_inputs, separators=(",", ":"), allow_nan=True),
        mode=mode,
        load_seconds=cfg.load_seconds,
        code=code,
        entry=entry,
        call_seconds=cfg.call_seconds,
        node_budget=NODE_BUDGET,
        inline=INLINE_CHARS,
    )
    return prog, nonce


def _parse(stdout: str, nonce: str, wanted: set[int]) -> dict[int, Outcome]:
    seen: dict[int, list[Outcome]] = defaultdict(list)
    for line in stdout.splitlines():
        at = line.find(nonce + " ")
        if at < 0:
            continue
        parts = line[at + len(nonce) + 1 :].split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            i = int(parts[0])
            rec = json.loads(parts[1])
        except (ValueError, RecursionError):
            continue
        if i not in wanted or not isinstance(rec, list) or not rec or not isinstance(rec[0], str):
            continue
        kind = rec[0]
        if kind == "ok" and len(rec) == 2:
            seen[i].append(Outcome("ok", rec[1]))
        elif kind in ("exc", "unser", "load") and len(rec) == 2:
            seen[i].append(Outcome(kind, None, str(rec[1])[:80]))
        elif kind in ("timeout", "big") and len(rec) == 1:
            seen[i].append(Outcome(kind))
    return {i: (v[0] if len(v) == 1 else Outcome("dup", None, f"{len(v)} reports")) for i, v in seen.items()}


def run_io(
    code: str,
    entry_point: str,
    inputs: Sequence[Sequence[Any]],
    *,
    mode: str = "value",
    config: RunConfig | None = None,
    encoded: bool = False,
    check: Callable[[int, Outcome], bool] | None = None,
    purpose: str = "functional_diagnostic",
) -> list[Outcome]:
    """Call ``entry_point`` defined by ``code`` on each argument tuple, in the sandbox, one process per chunk.

    Returns one :class:`Outcome` per input. ``mode="value"`` returns encoded values (long ones as a digest);
    ``mode="digest"`` returns ``["H", strict, loose, unordered]`` digests for each value (twin screening).
    A chunk whose process dies or whose output is cut is re-run on the missing calls (smaller chunks), so
    one bad call cannot hide the others; a call that still reports nothing is a ``crash``.

    ``check`` (host side) is called on each outcome; once it returns False the remaining calls are not run
    and are reported as ``skip`` (fail-fast grading)."""
    _require_diagnostic(purpose)
    cfg = config or RunConfig()
    enc = [list(x) if encoded else [encode_value(a) for a in x] for x in inputs]
    results: dict[int, Outcome] = {}
    pending = [list(range(len(enc)))] if enc else []
    sizes = [len(_canon(e)) for e in enc]
    line_chars = 80.0  # running estimate of one report line; the sandbox keeps only the output tail
    failed = False
    while pending:
        group = pending.pop()
        if failed:
            results.update({i: Outcome("skip") for i in group})
            continue
        cap = max(1, min(400, int(cfg.chunk_chars / line_chars)))
        chunk: list[int] = []
        budget = 0
        for i in group:
            if chunk and (len(chunk) >= cap or budget + sizes[i] > 4_000_000):
                break
            chunk.append(i)
            budget += sizes[i]
        if len(chunk) < len(group):
            pending.append(group[len(chunk) :])
        got = _run_chunk(code, entry_point, chunk, [enc[i] for i in chunk], mode, cfg)
        if got:
            observed = sum(len(_canon(o.value)) + 40 if o.ok else 60 for o in got.values()) / len(got)
            line_chars = max(40.0, 0.5 * line_chars + 0.5 * observed)
        results.update(got)
        if check is not None and not all(check(i, o) for i, o in sorted(got.items())):
            failed = True
        missing = [i for i in chunk if i not in got]
        if not missing:
            continue
        if len(chunk) == 1:
            results[chunk[0]] = Outcome("crash", None, "no report")
            failed = failed or check is not None
        else:
            # Calls run in order, so the first unreported call is the likely culprit (crash, hang, flood):
            # isolate it and re-run the rest without it.
            if missing[1:]:
                pending.append(missing[1:])
            pending.append(missing[:1])
    return [results[i] for i in range(len(enc))]


def _run_chunk(
    code: str, entry: str, idx: list[int], enc: list[Any], mode: str, cfg: RunConfig
) -> dict[int, Outcome]:
    prog, nonce = _program(code, entry, idx, enc, mode, cfg)
    try:
        result = sandbox.run_python(prog, cfg.limits)
    except sandbox.SandboxUnavailable:
        # Untrusted code can print a fake "bwrap:" error before dying; only a failing probe means no sandbox.
        if not sandbox.available():
            raise
        return {}
    if result.output_limited:
        # Only the output tail survives a flood; reports in it may be cut, so re-run smaller chunks.
        return {}
    # A timeout or non-zero exit after some calls still leaves their reports valid; discarding them
    # made run_io re-run every remaining call one chunk at a time (reports were never authenticated:
    # see EXECUTION_POLICY).
    return _parse(result.stdout, nonce, set(idx))


# --- python_io specifications ---------------------------------------------------------------------


def _spec_list(spec: dict[str, Any], key: str) -> list[Any] | None:
    if key in spec:
        return list(spec[key])
    if f"{key}_literal" in spec:
        return list(safe_literal(str(spec[f"{key}_literal"])))
    if f"{key}_enc" in spec:
        return [_dec(x) for x in spec[f"{key}_enc"]]
    return None


def expected_outputs(
    spec: dict[str, Any], config: RunConfig | None = None
) -> tuple[list[list[Any]], list[Outcome], list[int]]:
    """Inputs (as argument lists), expected outcomes and the indices of inputs excluded because the trusted
    reference raised, timed out or returned an unserializable value on them."""
    raw = _spec_list(spec, "inputs")
    if raw is None:
        raise sftscore.SpecError("python_io spec without inputs")
    inputs = [list(x) for x in raw]
    expected_values = _spec_list(spec, "expected")
    if expected_values is not None:
        if len(expected_values) != len(inputs):
            raise sftscore.SpecError("python_io inputs and expected differ in length")
        return inputs, [Outcome("ok", encode_value(v)) for v in expected_values], []
    reference = spec.get("reference")
    if not reference:
        raise sftscore.SpecError("python_io spec needs expected values or a reference")
    prefix = spec.get("prefix") or ""
    code = (
        (prefix + "\n" + str(reference))
        if prefix and not spec.get("reference_is_complete")
        else str(reference)
    )
    want = run_io(code, str(spec["entry_point"]), inputs, config=config)
    excluded = [i for i, w in enumerate(want) if not w.ok]
    return inputs, want, excluded


def score_python_io(
    spec: dict[str, Any],
    response: str,
    config: RunConfig | None = None,
    *,
    fail_fast: bool = True,
    purpose: str = "functional_diagnostic",
) -> tuple[bool, str]:
    """Grade a response against a ``python_io`` specification out of process.

    ``spec`` = ``{"type": "python_io", "entry_point", "prefix"?, "inputs": [...] | "inputs_literal",
    "expected": [...] | "expected_literal" | "reference": code, "float_tol"?: {"rel","abs"} | number,
    "compare"?: "exact" | "set" | "multiset"}``. The candidate block is chosen as by
    :func:`j1data.sftscore.select_code` and the prompt ``prefix`` is prepended (HumanEval style).
    Raises :class:`j1data.sandbox.SandboxUnavailable` when isolation is unavailable."""
    _require_diagnostic(purpose)
    _require_diagnostic(str(spec.get("purpose", "functional_diagnostic")))
    if spec.get("type") not in (None, "python_io"):
        raise sftscore.SpecError(f"not a python_io spec: {spec.get('type')!r}")
    entry = str(spec["entry_point"])
    inputs, want, excluded = expected_outputs(spec, config)
    keep = [i for i in range(len(inputs)) if i not in set(excluded)]
    if not keep:
        raise sftscore.SpecError("the reference fails on every input")
    code = sftscore.select_code(response, entry)
    prefix = spec.get("prefix") or ""
    if prefix:
        code = prefix + "\n" + code
    cleaned = sftscore.strip_main_blocks(code)
    if cleaned is None:
        return False, "candidate does not parse"
    tol = FloatTol.parse(spec.get("float_tol"))
    compare = str(spec.get("compare", "exact"))

    def correct(j: int, g: Outcome) -> bool:
        return g.ok and values_equal(g.value, want[keep[j]].value, tol, compare)

    got = run_io(
        cleaned, entry, [inputs[i] for i in keep], config=config, check=correct if fail_fast else None
    )
    failures = []
    for j, i in enumerate(keep):
        g = got[j]
        if g.kind == "skip":
            continue
        if not g.ok:
            failures.append((i, f"{g.kind} {g.detail}".strip()))
        elif not correct(j, g):
            failures.append((i, "wrong value"))
    note = (f"; {len(excluded)} inputs excluded (reference failed)" if excluded else "") + f"; {LIMITATION}"
    if failures:
        i, why = failures[0]
        done = sum(1 for g in got if g.kind != "skip")
        return False, f"failed input {i} ({why}); {done - len(failures)}/{done} run inputs correct{note}"
    return True, f"{len(keep)}/{len(keep)} inputs correct{note}"


# --- adversarial controls ------------------------------------------------------------------------------
#
# Candidate programs that defeat assert-based, in-process graders (j1.sftscore v3 notes the frame-inspection
# gap). These historical controls are finite examples, not an introspection-resistance proof.

EXPLOIT_TASK = {
    "inputs": [[[1, 2, 3], 2], [[], 5], [[-4, 4, 7], 1], [[10], 0]],
    "reference": "def total(xs, k):\n    return sum(xs) * k\n",
    "tests": "assert total([1, 2, 3], 2) == 12\nassert total([], 5) == 0\nassert total([-4, 4, 7], 1) == 7\n"
    "assert total([10], 0) == 0\n",
}

EXPLOIT_PROGRAMS: dict[str, str] = {
    "frame_inspection_forges_reports": (
        "import sys, os\n"
        "def total(xs, k):\n"
        "    f = sys._getframe()\n"
        "    while f.f_back is not None:\n"
        "        f = f.f_back\n"
        "    found = [c for c in f.f_code.co_consts if isinstance(c, str) and len(c.strip()) == 32]\n"
        "    nonce = f.f_globals.get('_NONCE') or (found[-1].strip() if found else '')\n"
        "    for i in range(10):\n"
        "        os.write(1, f'{nonce} {i} [\"ok\", 0]\\n'.encode())\n"
        "    os.write(1, ('\\n' + nonce + '\\n').encode())\n"
        "    os._exit(0)\n"
    ),
    "patch_assertion_error_and_builtins": (
        "import builtins\n"
        "builtins.AssertionError = type('Never', (Exception,), {})\n"
        "builtins.isinstance = lambda *a: True\n"
        "builtins.len = lambda *a: 0\n"
        "def total(xs, k):\n    return 0\n"
    ),
    "sys_exit_at_import": "import sys\nsys.exit(0)\ndef total(xs, k):\n    return 0\n",
    "sys_exit_in_call": "import sys\ndef total(xs, k):\n    sys.exit(0)\n",
    "os_exit_in_call": "import os\ndef total(xs, k):\n    os._exit(0)\n",
    "monkeypatch_runner_module": (
        "import sys, json\n"
        "m = sys.modules['__main__']\n"
        "m._enc = lambda *a, **k: 0\n"
        "m.values_equal = lambda *a, **k: True\n"
        "json.dumps = lambda *a, **k: '0'\n"
        "def total(xs, k):\n    return 0\n"
    ),
    "print_fake_marker": (
        "import os\n"
        "print('PASSED')\n"
        'os.write(1, b\'ok 0 ["ok", 6]\\nok 1 ["ok", 0]\\n\')\n'
        "def total(xs, k):\n    print('all tests passed')\n    return 1\n"
    ),
    "always_equal_object": (
        "class E:\n    def __eq__(self, o):\n        return True\n"
        "    def __ne__(self, o):\n        return False\n"
        "def total(xs, k):\n    return E()\n"
    ),
    "int_subclass_with_forged_eq": (
        "class I(int):\n    def __eq__(self, o):\n        return True\n    __hash__ = int.__hash__\n"
        "def total(xs, k):\n    return I(0)\n"
    ),
    "swallow_timer_and_spin": (
        "def total(xs, k):\n    while True:\n        try:\n            while True:\n                pass\n"
        "        except BaseException:\n            pass\n"
    ),
    "fake_sandbox_failure": (
        "import os\ndef total(xs, k):\n    os.write(2, b'bwrap: cannot create namespace')\n    os._exit(1)\n"
    ),
    "infinite_recursion": "def total(xs, k):\n    return total(xs, k)\n",
    "huge_output": "def total(xs, k):\n    return 'x' * (10 ** 8)\n",
    "generator_result": "def total(xs, k):\n    yield sum(xs) * k\n",
}


def exploit_table(config: RunConfig | None = None) -> list[dict[str, Any]]:
    """Each adversarial program graded by the in-process assert harness and by this checker."""
    spec = {
        "type": "python_io",
        "entry_point": "total",
        **{k: EXPLOIT_TASK[k] for k in ("inputs", "reference")},
    }
    rows = []
    for name, code in sorted(EXPLOIT_PROGRAMS.items()):
        response = f"```python\n{code}```"
        harness = {
            "type": "python_tests",
            "setup": "",
            "tests": EXPLOIT_TASK["tests"],
            "entry_point": "total",
        }
        try:
            old: bool | str = sftscore.score(harness, response)[0]
        except sandbox.SandboxUnavailable:
            old = "raises SandboxUnavailable (item becomes 'not executed')"
        new = score_python_io(spec, response, config)
        rows.append(
            {"exploit": name, "in_process_harness": old, "codecheck": new[0], "codecheck_detail": new[1]}
        )
    return rows


# --- task records ------------------------------------------------------------------------------------
#
# A task record is a plain dict: {"id", "source", "split", "text", "code" (complete reference program),
# "entry_point", "tests" (assert code), "setup", "callee" (name the tests call), "executable"}.

MBPP_SPLITS = ("train", "prompt", "validation", "test")
PROTECTED_SPLITS = frozenset({"test", "humaneval", "jbc"})


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def _normalize_code(code: str) -> str:
    from j1data.sftsources import normalize_code

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return normalize_code(code)
        except (SyntaxError, ValueError):
            return code.replace("\r\n", "\n")


def mbpp_records(mbpp_dir: Path) -> list[dict[str, Any]]:
    """Every MBPP ``full`` record (train, prompt, validation, test) from the pinned parquet directory."""
    from j1data.sftsources import mbpp_entry_point

    out = []
    for split in MBPP_SPLITS:
        for row in _read_parquet(mbpp_dir / f"{split}-00000-of-00001.parquet"):
            entry = mbpp_entry_point(row["test_list"]) or ""
            out.append(
                {
                    "id": f"mbpp:{row['task_id']}",
                    "source": "mbpp",
                    "split": split,
                    "text": row["text"],
                    "code": _normalize_code(row["code"]),
                    "entry_point": entry,
                    "callee": entry,
                    "setup": row.get("test_setup_code") or "",
                    "tests": "\n".join([*row["test_list"], *(row.get("challenge_test_list") or [])]),
                    "executable": bool(entry),
                }
            )
    return out


def _docstring_text(prompt: str, entry: str) -> str:
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return prompt
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            doc = ast.get_docstring(node) or ""
            lines, skip = [], False
            for line in doc.splitlines():
                stripped = line.strip()
                if stripped.startswith(">>>") or stripped.lower().startswith(("example", "for example")):
                    skip = True
                    continue
                if skip and stripped:
                    continue
                skip = False
                lines.append(line)
            return " ".join(" ".join(lines).split())
    return prompt


def humaneval_records(parquet: Path) -> list[dict[str, Any]]:
    out = []
    for row in _read_parquet(parquet):
        n = str(row["task_id"]).split("/")[-1]
        out.append(
            {
                "id": f"humaneval:{n}",
                "source": "humaneval",
                "split": "humaneval",
                "text": _docstring_text(row["prompt"], row["entry_point"]),
                "code": row["prompt"] + row["canonical_solution"],
                "prefix": row["prompt"],
                "entry_point": row["entry_point"],
                "callee": "candidate",
                "setup": "",
                "tests": row["test"],
                "executable": True,
            }
        )
    return out


def jbc_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """JuniperBench-Code records (HumanEval format); tasks that need packages outside the standard library
    (``requires``) and test-writing tasks are kept for text/structure comparison but not executed."""
    out = []
    for row in rows:
        out.append(
            {
                "id": f"jbc:{row['task_id']}",
                "source": "jbc",
                "split": "jbc",
                "text": _docstring_text(row["prompt"], row["entry_point"]),
                "code": row["prompt"] + row["canonical_solution"],
                "prefix": row["prompt"],
                "entry_point": row["entry_point"],
                "callee": "candidate",
                "setup": "",
                "tests": row.get("test") or "",
                "executable": not row.get("requires") and row.get("kind", "complete") == "complete",
            }
        )
    return out


def task_id_for_record(source_record: str) -> str | None:
    """``mbpp/full/train/601#fix`` -> ``mbpp:601``; HumanEval ``HumanEval/12`` -> ``humaneval:12``."""
    m = re.match(r"mbpp/full/(?:train|prompt|validation|test)/(\d+)", source_record)
    if m:
        return f"mbpp:{m.group(1)}"
    m = re.match(r"HumanEval/(\d+)$", source_record)
    return f"humaneval:{m.group(1)}" if m else None


def harvest_inputs(record: dict[str, Any]) -> list[list[Any]]:
    """Argument tuples of every call to the tested function in the task's own asserts whose arguments are
    all literals (calls with keywords, starred or computed arguments are skipped)."""
    callee = record.get("callee") or record.get("entry_point")
    try:
        tree = ast.parse(record.get("tests") or "")
    except SyntaxError:
        return []
    out, seen = [], set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee):
            continue
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        try:
            args = [safe_literal(a) for a in node.args]
            key = _canon([encode_value(a) for a in args])
        except (ValueError, TypeError, Unserializable, OverflowError, RecursionError):
            continue
        if key not in seen:
            seen.add(key)
            out.append(args)
    return out


# --- type-preserving input mutation -------------------------------------------------------------------

_MAX_LEN = 48
_MAX_STR = 160


def _mutate_value(v: Any, rng: random.Random, depth: int = 0) -> Any:
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        big = max(abs(v) * 2 + 10, 100)
        opts = [v + 1, v - 1, 0, -v, v * 2, v + rng.randint(2, 9), 1, rng.randint(-big, big)]
        return rng.choice(opts)
    if isinstance(v, float):
        return rng.choice([v + 0.5, v - 1.25, 0.0, -v, v * 1.5, round(v, 1), v + rng.uniform(-3, 3)])
    if isinstance(v, str):
        if not v:
            return rng.choice(["a", " ", "ab", "A1"])
        k = rng.randrange(len(v))
        texts: list[str] = [
            "",
            v[::-1],
            v[1:],
            v[:-1],
            (v + v[k])[:_MAX_STR],
            v.upper(),
            v.swapcase(),
            v[:k] + " " + v[k:],
            v[:k] + rng.choice(v) + v[k + 1 :],
            v[k],
        ]
        return rng.choice(texts)[:_MAX_STR]
    if isinstance(v, (list, tuple)):
        items = list(v)
        op = rng.choice(
            ["empty", "single", "reverse", "drop", "dup", "swap", "elem", "elem", "sort", "extend"]
        )
        if op == "empty":
            items = []
        elif op == "single" and items:
            items = [rng.choice(items)]
        elif op == "reverse":
            items.reverse()
        elif op == "drop" and items:
            items.pop(rng.randrange(len(items)))
        elif op == "dup" and items:
            j = rng.randrange(len(items))
            items.insert(j, items[j])
        elif op == "swap" and len(items) > 1:
            i, j = rng.sample(range(len(items)), 2)
            items[i], items[j] = items[j], items[i]
        elif op == "elem" and items and depth < 4:
            j = rng.randrange(len(items))
            items[j] = _mutate_value(items[j], rng, depth + 1)
        elif op == "sort":
            try:
                items.sort()
            except TypeError:
                items.reverse()
        elif op == "extend" and items and depth < 4:
            items.append(_mutate_value(rng.choice(items), rng, depth + 1))
        items = items[:_MAX_LEN]
        return tuple(items) if isinstance(v, tuple) else items
    if isinstance(v, (set, frozenset)):
        as_list = sorted(v, key=lambda x: _canon(encode_value(x)))
        mutated = _mutate_value(as_list, rng, depth)
        try:
            return type(v)(mutated)
        except TypeError:
            return v
    if isinstance(v, dict):
        d = dict(v)
        keys = list(d)
        op = rng.choice(["empty", "drop", "value", "value", "add"])
        if op == "empty":
            return {}
        if op == "drop" and keys:
            d.pop(rng.choice(keys))
        elif op == "value" and keys and depth < 4:
            k = rng.choice(keys)
            d[k] = _mutate_value(d[k], rng, depth + 1)
        elif op == "add" and keys and depth < 4:
            k = rng.choice(keys)
            with contextlib.suppress(TypeError):
                d[_mutate_value(k, rng, depth + 1)] = d[k]
        return d
    return v


def mutate_inputs(seeds: Sequence[Sequence[Any]], n: int, seed: str) -> list[list[Any]]:
    """Up to ``n`` distinct argument tuples, each derived from a seed tuple by mutating one argument while
    keeping its type (ints stay ints, tuples stay tuples, elements keep their kind). Deterministic."""
    rng = random.Random(f"{VERSION}:{seed}")
    seen = set()
    for s in seeds:
        try:
            seen.add(_canon([encode_value(a) for a in s]))
        except (Unserializable, OverflowError):
            continue
    out: list[list[Any]] = []
    attempts = 0
    usable = [list(s) for s in seeds if s]
    while usable and len(out) < n and attempts < n * 25:
        attempts += 1
        base = list(rng.choice(usable))
        j = rng.randrange(len(base))
        base[j] = _mutate_value(base[j], rng)
        if attempts % 3 == 0 and len(base) > 1:  # sometimes mutate a second argument too
            k = rng.randrange(len(base))
            base[k] = _mutate_value(base[k], rng)
        try:
            key = _canon([encode_value(a) for a in base])
        except (Unserializable, OverflowError):
            continue
        if key not in seen:
            seen.add(key)
            out.append(base)
    return out


def _coarse(v: Any) -> str:
    if isinstance(v, bool):
        return "b"
    if isinstance(v, (int, float)):
        return "n"
    if isinstance(v, str):
        return "s"
    if isinstance(v, (list, tuple)):
        return "q"
    if isinstance(v, dict):
        return "d"
    if isinstance(v, (set, frozenset)):
        return "e"
    return "z" if v is None else "o"


# --- task profiles (verified inputs and reference outputs) -------------------------------------------


@dataclass
class Profile:
    """A task's verified behaviour: inputs on which its reference returned a serializable value, identically
    in two separate sandboxed runs, with (strict, loose) output digests."""

    id: str
    arity: int | None = None
    sigs: set[tuple[str, ...]] = field(default_factory=set)
    own: list[list[Any]] = field(default_factory=list)
    own_out: list[tuple[str, str, str]] = field(default_factory=list)
    mut: list[list[Any]] = field(default_factory=list)
    mut_out: list[tuple[str, str, str]] = field(default_factory=list)
    unordered: bool = False
    notes: list[str] = field(default_factory=list)


SCREEN = RunConfig(call_seconds=0.5, load_seconds=5.0)


def _runnable(record: dict[str, Any]) -> str:
    setup = record.get("setup") or ""
    return str(record["code"]) + ("\n" + setup if setup else "")


def _order_insensitive(record: dict[str, Any]) -> bool:
    """The task's own asserts compare ``set(f(...))`` or ``sorted(f(...))``."""
    callee = re.escape(str(record.get("callee") or record.get("entry_point") or ""))
    return (
        bool(callee)
        and re.search(rf"\b(?:set|sorted)\(\s*{callee}\s*\(", record.get("tests") or "") is not None
    )


def profile(record: dict[str, Any], mutations: int = 24, config: RunConfig = SCREEN) -> Profile:
    p = Profile(record["id"], unordered=_order_insensitive(record))
    if not record.get("executable") or not record.get("entry_point"):
        p.notes.append("not executed")
        return p
    own = harvest_inputs(record)
    if not own:
        p.notes.append("no literal test inputs")
        return p
    mut = mutate_inputs(own, mutations, record["id"])
    cand = own + mut
    first = run_io(_runnable(record), record["entry_point"], cand, mode="digest", config=config)
    second = run_io(_runnable(record), record["entry_point"], cand, mode="digest", config=config)
    bad_own = 0
    for i, (x, a, b) in enumerate(zip(cand, first, second, strict=True)):
        stable = a.ok and b.ok and a.value[1] == b.value[1]
        if i < len(own):
            if not stable:
                bad_own += 1
                continue
            p.own.append(x)
            p.own_out.append((a.value[1], a.value[2], a.value[3]))
        elif stable:
            p.mut.append(x)
            p.mut_out.append((a.value[1], a.value[2], a.value[3]))
    if bad_own:
        p.notes.append(f"reference fails or is unstable on {bad_own}/{len(own)} own inputs")
    arities = {len(x) for x in p.own}
    p.arity = arities.pop() if len(arities) == 1 else None
    p.sigs = {tuple(_coarse(a) for a in x) for x in p.own}
    return p


def profiles(records: Sequence[dict[str, Any]], mutations: int = 24, workers: int = 12) -> dict[str, Profile]:
    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(lambda r: profile(r, mutations), records))
    return {p.id: p for p in done}


# --- statement and structure similarity ---------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _text_norm(text: str, numbers: bool) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    if numbers:
        t = _NUM_RE.sub("0", t)
    return " ".join(re.findall(r"\w+", t))


def statement_features(text: str) -> dict[str, frozenset[Any]]:
    """Word 3-grams without stopwords (the v1.2 measure), the same with numbers normalized, character
    5-grams with numbers normalized, and content-word sets."""
    from j1data.normalize import signature_words
    from j1data.sftbuild import _STOP

    words = [w for w in signature_words(text) if w not in _STOP]
    w3 = {tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2))} or {tuple(words)}
    nw = [w for w in _text_norm(text, True).split() if w not in _STOP]
    w3n = {tuple(nw[i : i + 3]) for i in range(max(0, len(nw) - 2))} or {tuple(nw)}
    flat = _text_norm(text, True)
    c5 = {flat[i : i + 5] for i in range(max(0, len(flat) - 4))} or {flat}
    return {"w3": frozenset(w3), "w3n": frozenset(w3n), "c5": frozenset(c5), "w1n": frozenset(nw)}


def _jaccard(a: frozenset[Any], b: frozenset[Any]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


STATEMENT_THRESHOLDS = {"w3": 0.6, "w3n": 0.6, "c5": 0.7}


def structure_keys(code: str) -> dict[str, str]:
    """``same_code``: the v1.2 identifier-renamed canonical code; ``same_shape``: additionally docstrings
    removed and every constant normalized (numbers to 0, strings to ''); ``size``: AST node count."""
    from j1data.sftbuild import canonical_code

    renamed = canonical_code(code)
    try:
        tree = ast.parse(renamed)
    except SyntaxError:
        return {"same_code": hashlib.sha256(renamed.encode()).hexdigest(), "same_shape": "", "size": "0"}

    class Norm(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            self.generic_visit(node)
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            v = node.value
            if isinstance(v, bool) or v is None:
                return node
            if isinstance(v, (int, float, complex)):
                return ast.copy_location(ast.Constant(0), node)
            if isinstance(v, (str, bytes)):
                return ast.copy_location(ast.Constant(""), node)
            return node

    shape = Norm().visit(tree)
    ast.fix_missing_locations(shape)
    try:
        shape_src = ast.unparse(shape)
    except (ValueError, AttributeError):
        shape_src = renamed
    size = sum(1 for _ in ast.walk(tree))
    return {
        "same_code": hashlib.sha256(renamed.encode()).hexdigest(),
        "same_shape": hashlib.sha256(shape_src.encode()).hexdigest(),
        "size": str(size),
    }


# --- functional twins ---------------------------------------------------------------------------------

TWIN, AMBIGUOUS, DISTINCT = "TWIN", "AMBIGUOUS", "DISTINCT"
EVIDENCE_NOTE = (
    "Finite cross-execution is evidence of behavioural overlap, not a proof of equivalence; DISTINCT means "
    "no overlap was found by these tests, not that none exists."
)


def _compatible(a: Profile, b: Profile) -> bool:
    if a.arity is None or a.arity != b.arity or not a.own or not b.own:
        return False
    return bool(a.sigs & b.sigs)


def _cross_run(
    jobs: dict[str, list[list[Any]]],
    records: dict[str, dict[str, Any]],
    workers: int,
) -> dict[str, dict[str, Outcome]]:
    """Run each task's reference on the given inputs; returns task id -> canonical input -> outcome."""

    def one(tid: str) -> tuple[str, dict[str, Outcome]]:
        uniq: dict[str, list[Any]] = {}
        for x in jobs[tid]:
            uniq.setdefault(_canon([encode_value(a) for a in x]), x)
        keys = list(uniq)
        rec = records[tid]
        outs = run_io(
            _runnable(rec), rec["entry_point"], [uniq[k] for k in keys], mode="digest", config=SCREEN
        )
        return tid, dict(zip(keys, outs, strict=True))

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, [t for t in jobs if jobs[t]]))


def _agree(o: Outcome | None, want: tuple[str, str, str], unordered: bool) -> str:
    if o is None or not o.ok:
        return "err"
    if o.value[1] == want[0]:
        return "strict"
    if o.value[2] == want[1] or (unordered and o.value[3] == want[2]):
        return "loose"
    return "differ"


def functional_twins(
    records_a: Sequence[dict[str, Any]],
    records_b: Sequence[dict[str, Any]],
    *,
    mutations: int = 24,
    workers: int = 12,
    profiles_cache: dict[str, Profile] | None = None,
    include_distinct: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Compare every task in ``records_a`` with every task in ``records_b`` (pairs of the same id skipped,
    unordered pairs counted once) and return one dict per pair with any evidence of overlap:
    ``{"a", "b", "verdict", "reasons", "evidence"}``.

    Each reference runs on its own literal test inputs and on type-preserving mutations of them (kept only
    where it returns the same value in two runs). Pairs with the same arity and argument kinds are screened
    by running each reference on the other's test inputs; pairs that pass at least half of the other's
    tests, or have similar statements or identical code, are also run on each other's mutated inputs.

    Verdicts: ``TWIN`` - each reference reproduces the other's outputs on all of the other's own test inputs
    and on every mutated input of both tasks (no disagreement, no input where only one errors, >= 6
    agreeing inputs, agreement also on non-majority outputs); ``AMBIGUOUS`` - one reference passes all of
    the other's tests and agrees on >= 80% of its mutated inputs, or they agree on >= 80% of >= 5 shared
    inputs, or agree wherever both are defined (partial domain overlap), or the references are identical up
    to identifier renaming, or the statements are near-duplicates under the v1.2 measure (word 3-gram
    Jaccard >= 0.6; the new character/number-normalized measures count only when behaviour could not be
    compared); ``DISTINCT`` otherwise. Agreement on booleans and other near-constant outputs counts only
    with agreement on the non-majority outputs. Agreement is judged leniently (list == tuple, bool == int,
    floats to six significant digits), which makes the detector conservative. See :data:`EVIDENCE_NOTE`."""
    say = progress or (lambda _m: None)
    records = {r["id"]: r for r in [*records_a, *records_b]}
    ids_a = list(dict.fromkeys(r["id"] for r in records_a))
    ids_b = list(dict.fromkeys(r["id"] for r in records_b))
    pairs = sorted({tuple(sorted((a, b))) for a in ids_a for b in ids_b if a != b})
    say(f"{len(records)} tasks, {len(pairs)} pairs")
    profs = dict(profiles_cache or {})
    todo = [records[t] for t in records if t not in profs]
    if todo:
        say(f"profiling {len(todo)} references")
        profs.update(profiles(todo, mutations, workers))
    feats = {t: statement_features(records[t].get("text") or "") for t in records}
    shapes = {t: structure_keys(records[t]["code"]) for t in records}

    stage: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in pairs:
        st = {k: round(_jaccard(feats[a][k], feats[b][k]), 3) for k in ("w3", "w3n", "c5", "w1n")}
        same_code = shapes[a]["same_code"] == shapes[b]["same_code"]
        same_shape = bool(shapes[a]["same_shape"]) and shapes[a]["same_shape"] == shapes[b]["same_shape"]
        ev: dict[str, Any] = {
            "statement": st,
            "structure": {"same_code": same_code, "same_shape": same_shape},
            "compatible": _compatible(profs[a], profs[b]),
        }
        stage[(a, b)] = ev
    say("screening own test inputs")
    jobs: dict[str, list[list[Any]]] = defaultdict(list)
    for (a, b), ev in stage.items():
        if ev["compatible"]:
            jobs[a].extend(profs[b].own)
            jobs[b].extend(profs[a].own)
    runs = _cross_run(jobs, records, workers)

    def score(src: str, dst: str, which: str) -> Counter[str]:
        p = profs[dst]
        xs, outs = (p.own, p.own_out) if which == "own" else (p.mut, p.mut_out)
        table = runs2 if which == "mut" else runs
        c: Counter[str] = Counter()
        modal = Counter(w[0] for w in outs).most_common(1)[0][0] if outs else None
        for x, want in zip(xs, outs, strict=True):
            got = _agree(table.get(src, {}).get(_canon([encode_value(v) for v in x])), want, p.unordered)
            c[got] += 1
            if want[0] != modal:  # agreement on a non-majority output is informative even for booleans
                c["nonmodal"] += 1
                c["nonmodal_agree"] += got in ("strict", "loose")
        return c

    runs2: dict[str, dict[str, Outcome]] = {}
    for (a, b), ev in stage.items():
        if not ev["compatible"]:
            continue
        ab, ba = score(a, b, "own"), score(b, a, "own")
        ev["own_a_on_b"] = dict(ab)
        ev["own_b_on_a"] = dict(ba)
    second = []
    for (a, b), ev in stage.items():
        if not ev["compatible"]:
            continue
        hit = [
            (c.get("strict", 0) + c.get("loose", 0))
            / max(1, c.get("strict", 0) + c.get("loose", 0) + c.get("differ", 0) + c.get("err", 0))
            for c in (ev["own_a_on_b"], ev["own_b_on_a"])
        ]
        st = ev["statement"]
        if (
            max(hit) >= 0.5
            or ev["structure"]["same_code"]
            or any(st[k] >= v for k, v in STATEMENT_THRESHOLDS.items())
        ):
            second.append((a, b))
    say(f"cross-executing mutated inputs for {len(second)} candidate pairs")
    jobs2: dict[str, list[list[Any]]] = defaultdict(list)
    for a, b in second:
        jobs2[a].extend(profs[b].mut)
        jobs2[b].extend(profs[a].mut)
    runs2.update(_cross_run(jobs2, records, workers))
    for a, b in second:
        ev = stage[(a, b)]
        ev["mut_a_on_b"] = dict(score(a, b, "mut"))
        ev["mut_b_on_a"] = dict(score(b, a, "mut"))

    out = []
    for (a, b), ev in stage.items():
        verdict, reasons = _verdict(ev)
        if verdict != DISTINCT or include_distinct or "mut_a_on_b" in ev:
            out.append({"a": a, "b": b, "verdict": verdict, "reasons": reasons, "evidence": ev})
    return out


def _verdict(ev: dict[str, Any]) -> tuple[str, list[str]]:
    """Behavioural verdict for one pair (rules in :func:`functional_twins`)."""
    reasons: list[str] = []

    def tally(*keys: str) -> Counter[str]:
        c: Counter[str] = Counter()
        for k in keys:
            c.update(ev.get(k, {}))
        return c

    def agreed(c: Counter[str] | dict[str, int]) -> int:
        return int(c.get("strict", 0) + c.get("loose", 0))

    def calls(c: Counter[str] | dict[str, int]) -> int:
        return agreed(c) + int(c.get("differ", 0) + c.get("err", 0))

    def passes(k: str) -> bool:
        c = ev.get(k, {})
        return calls(c) > 0 and agreed(c) == calls(c)

    def informative(c: Counter[str] | dict[str, int], need: int) -> bool:
        # agreement on outputs other than the most common one (a constant-ish predicate agrees by chance)
        n = int(c.get("nonmodal", 0))
        return n >= need and int(c.get("nonmodal_agree", 0)) >= 0.8 * n

    st = ev["statement"]
    if st["w3"] >= STATEMENT_THRESHOLDS["w3"]:
        reasons.append("statement_near_duplicate")  # the v1.2 rule: always at least AMBIGUOUS
    elif any(st[k] >= v for k, v in STATEMENT_THRESHOLDS.items()):
        reasons.append("statement_similar")  # new measures: AMBIGUOUS only when behaviour is not compared
    if ev["structure"]["same_code"]:
        reasons.append("same_code_up_to_renaming")
    compared = bool(ev.get("compatible")) and "mut_a_on_b" in ev
    if not ev.get("compatible"):
        return (AMBIGUOUS if reasons else DISTINCT), reasons
    mut = tally("mut_a_on_b", "mut_b_on_a")
    every = tally("own_a_on_b", "own_b_on_a", "mut_a_on_b", "mut_b_on_a")
    agree = agreed(every)
    if (
        passes("own_a_on_b")
        and passes("own_b_on_a")
        and calls(mut) >= 6
        and mut.get("differ", 0) == 0
        and mut.get("err", 0) == 0
        and agree >= 6
        and informative(every, 2)
    ):
        return TWIN, ["identical_on_all_inputs", *reasons]
    for own, mut_key, label in (
        ("own_a_on_b", "mut_a_on_b", "a_passes_b"),
        ("own_b_on_a", "mut_b_on_a", "b_passes_a"),
    ):
        if passes(own):
            m = ev.get(mut_key, {})
            # Weak upstream tests (e.g. three booleans) are passed by unrelated functions; require the
            # behaviour to carry over to the passed task's mutated inputs as well.
            if calls(m) >= 3 and agreed(m) >= 0.8 * calls(m) and informative(m, 2):
                reasons.append(f"{label}_tests")  # a's reference reproduces b on b's tests and mutations
            else:
                reasons.append("coincidental_test_pass")
    total = calls(every)
    if agree >= 5 and agree >= 0.8 * total and informative(every, 3):
        reasons.append(f"agrees_on_{agree}/{total}_inputs")
    elif agree >= 5 and every.get("differ", 0) == 0 and every.get("err", 0) > 0 and informative(every, 2):
        reasons.append("agrees_where_both_defined")
    weak = {"coincidental_test_pass"} | ({"statement_similar"} if compared else set())
    strong = [r for r in reasons if r not in weak]
    return (AMBIGUOUS if strong else DISTINCT), reasons


def clusters(pairs: Iterable[dict[str, Any]], verdicts: Iterable[str] = (TWIN, AMBIGUOUS)) -> list[list[str]]:
    """Connected components of the pair graph restricted to ``verdicts`` (union-find), largest first."""
    keep = set(verdicts)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if p["verdict"] in keep:
            ra, rb = find(p["a"]), find(p["b"])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = defaultdict(list)
    for x in parent:
        groups[find(x)].append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def is_protected(task_id: str, split_of: dict[str, str]) -> bool:
    return split_of.get(task_id) in PROTECTED_SPLITS


def quarantine(
    pairs: Iterable[dict[str, Any]],
    candidates: Iterable[str],
    protected: Iterable[str],
    *,
    transitive: str = "twin",
) -> dict[str, list[str]]:
    """v1.3 policy: a train/dev task is quarantined when it is TWIN or AMBIGUOUS with a protected (test or
    evaluation) task. ``transitive``: ``"none"`` direct links only; ``"twin"`` (default) also every task in
    the same TWIN cluster as a directly linked task (behaviourally identical tasks share one decision);
    ``"all"`` the whole TWIN+AMBIGUOUS component. Every derivation of a task (bug-fix variant, paraphrase)
    shares its task id and therefore its decision. Returns candidate task id -> protected task ids."""
    pairs = list(pairs)
    prot = set(protected)
    cand = set(candidates)
    linked: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        if p["verdict"] == DISTINCT:
            continue
        for x, y in ((p["a"], p["b"]), (p["b"], p["a"])):
            if y in prot:
                linked[x].add(y)
    if transitive != "none":
        verdicts = (TWIN,) if transitive == "twin" else (TWIN, AMBIGUOUS)
        for group in clusters(pairs, verdicts):
            hits = set().union(*(linked.get(t, set()) for t in group)) | prot.intersection(group)
            for t in group:
                linked[t] |= hits - {t}
    return {t: sorted(v) for t, v in sorted(linked.items()) if t in cand and t not in prot and v}


def load_twin_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    if report.get("version") != VERSION:
        raise ValueError(f"{path}: not a {VERSION} report")
    return report


def quarantined_tasks(report: dict[str, Any]) -> dict[str, list[str]]:
    """The quarantine decided by a committed twin report: task id -> protected task ids. A builder rejects
    every train/dev item whose :func:`task_id_for_record` is a key (all derivations of the task)."""
    q: dict[str, list[str]] = report["quarantine"]["tasks"]
    return q


# --- stronger tests (evaluation only) ------------------------------------------------------------------
#
# Expected values always come from the trusted reference executed in the sandbox. EvalPlus inputs are used
# only where the reference reproduces EvalPlus's recorded result (within EvalPlus's tolerance) and does so
# identically in two runs; mutated inputs only where the reference returns a value identically twice.
# Everything built here is protected evaluation material and stays on the offline verification path.

EVALPLUS_TOL = {
    "rel": 1e-7,
    "abs": 1e-6,
}  # EvalPlus `assertion`: np.allclose(rtol=1e-07, atol=1e-6) for floats
VERIFY = RunConfig(call_seconds=0.5, load_seconds=5.0)


def _assign_literal(tree: ast.AST, name: str) -> Any:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return safe_literal(node.value)
    raise KeyError(name)


def evalplus_tasks(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    """EvalPlus records keyed by our task id: inputs, recorded results (None where not literal), tolerance,
    order-insensitive comparison, EvalPlus reference. ``kind`` is ``humaneval`` or ``mbpp``."""
    out: dict[str, dict[str, Any]] = {}
    for row in _read_parquet(path):
        test = row["test"]
        try:
            tree = ast.parse(test)
            inputs = _assign_literal(tree, "inputs")
        except (SyntaxError, KeyError, ValueError, RecursionError):
            continue
        try:
            results = _assign_literal(tree, "results")
        except (KeyError, ValueError, RecursionError):
            results = None
        atol = 0.0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "assertion"
                and len(node.args) == 3
                and isinstance(node.args[2], ast.Constant)
                and isinstance(node.args[2].value, (int, float))
            ):
                atol = float(node.args[2].value)
        unordered = "out = set(out)" in test
        if kind == "humaneval":
            tid = "humaneval:" + str(row["task_id"]).split("/")[-1]
            reference = row["prompt"] + row["canonical_solution"]
            entry = row["entry_point"]
        else:
            tid = f"mbpp:{row['task_id']}"
            reference = "\n".join(row.get("test_imports") or []) + "\n" + row["code"]
            m = re.search(r"assertion\((\w+)\(\*inp\)", test)
            entry = m.group(1) if m else ""
        out[tid] = {
            "inputs": [list(x) for x in inputs],
            "results": results,
            "atol": atol,
            "unordered": unordered,
            "reference": reference,
            "entry_point": entry,
        }
    return out


def _compare_mode(record: dict[str, Any]) -> str:
    return "set" if _order_insensitive(record) else "exact"


def strong_suite(
    record: dict[str, Any],
    plus: dict[str, Any] | None,
    *,
    mutation_inputs: int = 60,
    plus_max: int = 400,
) -> dict[str, Any]:
    """A ``python_io`` spec for a protected task with tiers: own (the task's literal test inputs), plus
    (verified EvalPlus inputs), mut (verified mutated inputs) and holdout (more verified mutated inputs, used
    only to show that a program is wrong). Returns the spec plus verification counts. Protected material:
    write only to the offline evaluation store."""
    entry = record["entry_point"]
    code = _runnable(record)
    own = harvest_inputs(record)
    stats: Counter[str] = Counter()
    plus_inputs: list[list[Any]] = []
    plus_results: list[Any] = []
    if plus:
        plus_inputs = plus["inputs"][:plus_max]
        plus_results = (plus["results"] or [])[:plus_max]
    mut = mutate_inputs(own + plus_inputs[:50], mutation_inputs, record["id"])
    cand = own + plus_inputs + mut
    # alternate mutated inputs between the graded tier and a holdout used only to show that a program is
    # wrong, so the graded suite's miss rate is measured on inputs it never saw
    tiers = (
        ["own"] * len(own)
        + ["plus"] * len(plus_inputs)
        + [("mut", "holdout")[i % 2] for i in range(len(mut))]
    )
    first = run_io(code, entry, cand, config=VERIFY)
    second = run_io(code, entry, cand, config=VERIFY)
    # EvalPlus's (corrected) reference also has to agree on mutated inputs; this drops inputs where the
    # original reference is buggy or that fall outside EvalPlus's input contract
    ep_inputs = plus_inputs + mut
    ep_ref = run_io(plus["reference"], plus["entry_point"] or entry, ep_inputs, config=VERIFY) if plus else []
    tol = FloatTol(**EVALPLUS_TOL) if plus else FloatTol()
    mode = "set" if plus and plus["unordered"] else _compare_mode(record)
    keep_inputs, keep_expected, keep_tiers = [], [], []
    seen: set[str] = set()
    for i, (x, a, b, tier) in enumerate(zip(cand, first, second, tiers, strict=True)):
        if not (a.ok and b.ok and _canon(a.value) == _canon(b.value)):
            stats[f"{tier}_reference_failed_or_unstable"] += 1
            continue
        if tier == "plus":
            j = i - len(own)
            recorded: Any = None
            if j < len(plus_results):
                try:
                    recorded = encode_value(plus_results[j])
                except (Unserializable, OverflowError):
                    recorded = None
            agrees_recorded = recorded is not None and values_equal(a.value, recorded, tol, mode)
            agrees_ep = j < len(ep_ref) and ep_ref[j].ok and values_equal(a.value, ep_ref[j].value, tol, mode)
            if not (agrees_recorded or (plus_results == [] and agrees_ep)):
                stats["plus_disagrees_with_evalplus"] += 1
                continue
        elif tier in ("mut", "holdout") and plus:
            e = ep_ref[i - len(own)]
            if not (e.ok and values_equal(a.value, e.value, tol, mode)):
                stats[f"{tier}_disagrees_with_evalplus_reference"] += 1
                continue
        key = _canon([encode_value(v) for v in x])
        if key in seen:
            continue
        seen.add(key)
        keep_inputs.append(x)
        keep_expected.append(a.value)
        keep_tiers.append(tier)
        stats[f"{tier}_kept"] += 1
    floats = any(_has_float(v) for v in keep_expected)
    spec = {
        "type": "python_io",
        "entry_point": entry,
        "inputs_enc": [[encode_value(v) for v in x] for x in keep_inputs],
        "expected_enc": keep_expected,
        "tiers": keep_tiers,
        "compare": mode,
        "float_tol": EVALPLUS_TOL if floats else {"rel": 1e-9, "abs": 0.0},
    }
    if record.get("prefix"):
        spec["prefix"] = record["prefix"]
    return {"task": record["id"], "spec": spec, "stats": dict(stats)}


def _has_float(e: Any) -> bool:
    if isinstance(e, float):
        return True
    return isinstance(e, list) and e[:1] != ["H"] and any(_has_float(x) for x in e[1:])


def check_against_suite(
    suite: dict[str, Any], code: str, tiers: Iterable[str], *, purpose: str = "functional_diagnostic"
) -> tuple[bool, str]:
    """Run complete ``code`` (prefix already included) on the selected tiers; fail fast."""
    spec = suite["spec"]
    _require_diagnostic(purpose)
    _require_diagnostic(str(spec.get("purpose", "functional_diagnostic")))
    wanted = set(tiers)
    idx = [i for i, t in enumerate(spec["tiers"]) if t in wanted]
    if not idx:
        raise sftscore.SpecError("no inputs in requested suite tiers")
    sub = {
        "type": "python_io",
        "entry_point": spec["entry_point"],
        "inputs_enc": [spec["inputs_enc"][i] for i in idx],
        "expected_enc": [spec["expected_enc"][i] for i in idx],
        "compare": spec["compare"],
        "float_tol": spec["float_tol"],
    }
    return _score_code(sub, code)


TIERS = (
    ("own", ("own",)),
    ("plus", ("own", "plus")),
    ("all", ("own", "plus", "mut")),
    ("holdout", ("own", "plus", "mut", "holdout")),
)


def tiered_check(suite: dict[str, Any], code: str) -> dict[str, tuple[bool, str]]:
    """Results for the nested tiers own < +plus < +mut (the graded suite, "all") < +holdout, running each
    tier's new inputs only while the smaller tier passed (fail fast)."""
    out: dict[str, tuple[bool, str]] = {}
    previous: tuple[bool, str] = (True, "")
    for name, tiers in TIERS:
        if not previous[0]:
            out[name] = previous
            continue
        added = {tiers[-1]}
        result = check_against_suite(suite, code, added)
        previous = result if not result[0] else (True, f"{name}: passed")
        out[name] = previous
    return out


def _score_code(spec: dict[str, Any], code: str) -> tuple[bool, str]:
    """Like :func:`score_python_io` for an already-extracted program and pre-encoded expected values."""
    cleaned = sftscore.strip_main_blocks(code)
    if cleaned is None:
        return False, "candidate does not parse"
    inputs = spec["inputs_enc"]
    want = spec["expected_enc"]
    tol = FloatTol.parse(spec.get("float_tol"))
    compare = str(spec.get("compare", "exact"))

    def correct(i: int, g: Outcome) -> bool:
        return g.ok and values_equal(g.value, want[i], tol, compare)

    got = run_io(cleaned, spec["entry_point"], inputs, encoded=True, check=correct)
    for i, g in enumerate(got):
        if g.kind != "skip" and not correct(i, g):
            return False, f"input {i}: {g.kind if not g.ok else 'wrong value'} {g.detail}".strip()
    return True, f"{len(inputs)} inputs correct; {LIMITATION}"


# --- wrong-code controls ------------------------------------------------------------------------------


def _entry_def(tree: ast.Module, entry: str) -> ast.FunctionDef | None:
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == entry:
            return stmt
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            return node
    return None


def _literal_src(e: Any) -> str | None:
    """Python source for an encoded value, if it round-trips through :func:`safe_literal`."""
    if isinstance(e, list) and e[:1] == ["H"]:
        return None
    v = _dec(e)
    src = repr(v)
    if len(src) > 2000:
        return None
    try:
        back = encode_value(safe_literal(src))
    except (ValueError, SyntaxError, Unserializable, OverflowError, RecursionError):
        return None
    return src if _canon(back) == _canon(e) else None


def _default_for(e: Any) -> str:
    if isinstance(e, bool):
        return "False" if e else "True"
    if isinstance(e, int):
        return "0"
    if isinstance(e, float):
        return "0.0"
    if isinstance(e, str):
        return "''"
    if isinstance(e, list) and e:
        return {"L": "[]", "T": "()", "S": "set()", "F": "frozenset()", "D": "{}", "B": "b''"}.get(
            e[0], "None"
        )
    return "None"


class _OffByOne(ast.NodeTransformer):
    """Mutate the ``target``-th site: an integer constant +/-1, a strict/non-strict comparison, or a range()
    bound."""

    def __init__(self, target: int, delta: int) -> None:
        self.target, self.delta, self.seen = target, delta, -1

    def _hit(self) -> bool:
        self.seen += 1
        return self.seen == self.target

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, int) and not isinstance(node.value, bool) and self._hit():
            return ast.copy_location(ast.Constant(node.value + self.delta), node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        swap = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
        if type(node.ops[0]) in swap and self._hit():
            node.ops[0] = swap[type(node.ops[0])]()
        return node


def _sites(fn: ast.FunctionDef) -> int:
    m = _OffByOne(-1, 1)
    m.visit(ast.Module(body=[fn], type_ignores=[]))
    return m.seen + 1


def wrong_programs(
    record: dict[str, Any], first_output: Any, seed: str, per_kind: int = 2
) -> list[dict[str, str]]:
    """Deliberately wrong variants of a reference: off-by-one (constants, comparisons), constant return,
    returning the first test's expected value, swapped if/else branches, empty-input mishandling and the
    v1.2 bug-fix operator swaps. Returns ``[{"kind", "code"}]`` (complete programs)."""
    from j1data.sftsources import mutate, mutation_sites

    code, entry = record["code"], record["entry_point"]
    rng = random.Random(f"{VERSION}:wrong:{seed}")
    out: list[dict[str, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    fn = _entry_def(tree, entry)
    if fn is None:
        return out

    def variant(kind: str, new_fn: ast.FunctionDef) -> None:
        mod = ast.parse(code)
        target = _entry_def(mod, entry)
        assert target is not None
        target.body, target.args = new_fn.body, new_fn.args
        try:
            src = ast.unparse(ast.fix_missing_locations(mod))
        except (ValueError, AttributeError):
            return
        if src != ast.unparse(ast.parse(code)):
            out.append({"kind": kind, "code": src})

    def copy_fn() -> ast.FunctionDef:
        f = _entry_def(ast.parse(code), entry)
        assert f is not None
        return f

    n_sites = _sites(copy_fn())
    for site in rng.sample(range(n_sites), min(per_kind, n_sites)):
        f = copy_fn()
        _OffByOne(site, rng.choice([1, -1])).visit(f)
        variant("off_by_one", f)
    # whole-function replacements keep the signature (and any helpers) and return a fixed value
    head = f"def {entry}({ast.unparse(fn.args)}):\n    return "

    def stub(value_src: str) -> ast.FunctionDef:
        node = ast.parse(head + value_src + "\n").body[0]
        assert isinstance(node, ast.FunctionDef)
        return node

    variant("return_constant", stub(_default_for(first_output)))
    lit = _literal_src(first_output) if first_output is not None else None
    if lit is not None:
        variant("return_first_expected", stub(lit))
    ifs = [i for i, n in enumerate(ast.walk(copy_fn())) if isinstance(n, ast.If)]
    for k in rng.sample(ifs, min(per_kind, len(ifs))):
        f = copy_fn()
        node = list(ast.walk(f))[k]
        assert isinstance(node, ast.If)
        if node.orelse:
            node.body, node.orelse = node.orelse, node.body
        else:
            node.test = ast.UnaryOp(ast.Not(), node.test)
        variant("swapped_branches", f)
    if fn.args.args:
        a0 = fn.args.args[0].arg
        f = copy_fn()
        guard = ast.parse(f"if hasattr({a0}, '__len__') and len({a0}) == 0:\n    return None").body
        f.body = guard + f.body
        variant("empty_input_mishandled", f)
    try:
        sites = mutation_sites(code)
        for site in rng.sample(range(sites), min(per_kind, sites)):
            src = mutate(code, site)
            if src != ast.unparse(ast.parse(code)):
                out.append({"kind": "operator_swap", "code": src})
    except (SyntaxError, ValueError):
        pass
    uniq: dict[str, dict[str, str]] = {}
    for v in out:
        uniq.setdefault(v["code"], v)
    return list(uniq.values())


def upstream_spec(record: dict[str, Any]) -> dict[str, Any]:
    """The v1.2 evaluation specification (in-process assert harness) for a protected task."""
    if record["source"] == "humaneval":
        return {
            "type": "python_tests",
            "setup": "",
            "prefix": record["prefix"],
            "tests": f"{record['tests']}\ncheck({record['entry_point']})",
            "entry_point": record["entry_point"],
        }
    return {
        "type": "python_tests",
        "setup": record.get("setup") or "",
        "tests": record["tests"],
        "entry_point": record["entry_point"],
    }


def audit_task(record: dict[str, Any], suite: dict[str, Any]) -> list[dict[str, Any]]:
    spec = suite["spec"]
    first = spec["expected_enc"][0] if spec["expected_enc"] and spec["tiers"][0] == "own" else None
    rows = []
    for k, w in enumerate(wrong_programs(record, first, record["id"])):
        # upstream: the response is the whole program; HumanEval's prefix is joined as in v1.2
        old_ok, _ = sftscore.score(upstream_spec(record), f"```python\n{w['code']}\n```")
        tiers_ok = {name: res[0] for name, res in tiered_check(suite, w["code"]).items()}
        rows.append(
            {
                "task": record["id"],
                "variant": k,
                "kind": w["kind"],
                "upstream": old_ok,
                **{f"strong_{n}": v for n, v in tiers_ok.items()},
                "code_sha256": hashlib.sha256(w["code"].encode()).hexdigest(),
            }
        )
    return rows


# --- command line: twins report and v1.2 measurements ------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(path.read_text())


def _git_show(repo: Path, revision: str, file: str) -> bytes:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{file}"], check=True, capture_output=True
    ).stdout


def load_jbc(repo: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for f in cfg["files"]:  # later files supersede earlier ones per task id
        for line in _git_show(repo, cfg["revision"], f).decode().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["task_id"]] = row
    return jbc_records(rows.values())


def _sha256_file(path: Path) -> str:
    from j1data.hashing import sha256_file

    return sha256_file(path)


def load_all_tasks(data_root: Path, cfg: dict[str, Any], jbc_repo: Path | None) -> list[dict[str, Any]]:
    from j1data.registry import REPO_ROOT

    lock = _load_config(REPO_ROOT / "configs/sft_v1.toml")["sources"]["lock"]
    mdir = data_root / cfg["mbpp"]["dir"]
    for split in MBPP_SPLITS:
        f = mdir / f"{split}-00000-of-00001.parquet"
        if _sha256_file(f) != lock[f"mbpp/full/{split}-00000-of-00001.parquet"]:
            raise RuntimeError(f"{f}: bytes differ from the pinned lock")
    he = data_root / cfg["humaneval"]["file"]
    if _sha256_file(he) != cfg["humaneval"]["sha256"]:
        raise RuntimeError(f"{he}: bytes differ from the pinned SHA-256")
    tasks = mbpp_records(mdir) + humaneval_records(he)
    if jbc_repo is not None:
        tasks += load_jbc(jbc_repo, cfg["jbc"])
    return tasks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def v12_code_items(build: Path) -> list[dict[str, Any]]:
    """MBPP-derived train/dev records of a built dataset: episode id, split, family, task id."""
    out = []
    for split in ("train", "dev"):
        for r in _read_jsonl(build / f"{split}.jsonl"):
            tid = task_id_for_record(str(r.get("source_record", "")))
            if tid is not None:
                out.append(
                    {"episode_id": r["episode_id"], "split": split, "family": r["family"], "task": tid}
                )
    return out


def _summarize_pairs(pairs: list[dict[str, Any]], split_of: dict[str, str]) -> dict[str, Any]:
    def kind(p: dict[str, Any]) -> str:
        sa, sb = split_of[p["a"]], split_of[p["b"]]
        pa, pb = sa in PROTECTED_SPLITS, sb in PROTECTED_SPLITS
        return "protected-protected" if pa and pb else ("open-protected" if pa or pb else "open-open")

    by_verdict = Counter(p["verdict"] for p in pairs)
    by_kind = Counter(f"{kind(p)}:{p['verdict']}" for p in pairs)
    reasons = Counter(
        r.split("_on_")[0] if r.startswith("agrees_on") else r for p in pairs for r in p["reasons"]
    )
    return {
        "by_verdict": dict(by_verdict),
        "by_kind": dict(sorted(by_kind.items())),
        "reasons": dict(reasons),
    }


def cmd_twins(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config))
    tasks = load_all_tasks(Path(args.data_root), cfg, Path(args.jbc_repo) if args.jbc_repo else None)
    split_of = {t["id"]: t["split"] for t in tasks}
    t0 = time.time()
    pairs = functional_twins(
        tasks, tasks, mutations=int(cfg["twins"]["mutations"]), workers=args.workers, progress=_progress
    )
    elapsed = time.time() - t0
    protected = sorted(t for t in split_of if split_of[t] in PROTECTED_SPLITS)
    open_ids = sorted(t for t in split_of if split_of[t] not in PROTECTED_SPLITS)
    q_cluster = quarantine(pairs, open_ids, protected)
    q_direct = quarantine(pairs, open_ids, protected, transitive="none")
    q_twin = quarantine([p for p in pairs if p["verdict"] == TWIN], open_ids, protected, transitive="none")
    q_all = quarantine(pairs, open_ids, protected, transitive="all")
    groups = clusters(pairs, (TWIN,))
    report: dict[str, Any] = {
        "version": VERSION,
        "note": EVIDENCE_NOTE,
        "inputs": {
            "mbpp_revision": cfg["mbpp"]["revision"],
            "humaneval_sha256": cfg["humaneval"]["sha256"],
            "jbc": {"revision": cfg["jbc"]["revision"], "files": cfg["jbc"]["files"]}
            if args.jbc_repo
            else None,
            "tasks_by_split": dict(Counter(split_of.values())),
            "code": _code_identity(),
        },
        "parameters": {
            "mutations": cfg["twins"]["mutations"],
            "statement_thresholds": STATEMENT_THRESHOLDS,
            "screen_call_seconds": SCREEN.call_seconds,
        },
        "seconds": round(elapsed, 1),
        "summary": _summarize_pairs(pairs, split_of),
        "twin_clusters": {"count": len(groups), "largest": [len(g) for g in groups[:10]]},
        "quarantine": {
            "policy": "train/dev task TWIN or AMBIGUOUS with a protected task, or TWIN with such a task",
            "tasks": q_cluster,
            "component_all": sorted(q_all),
            "direct_only": sorted(q_direct),
            "twin_direct_only": sorted(q_twin),
        },
        "pairs": [p for p in pairs if p["verdict"] != DISTINCT],
        "distinct_pairs_with_partial_evidence": len([p for p in pairs if p["verdict"] == DISTINCT]),
    }
    if args.build:
        policies = {
            "twin_cluster": q_cluster,
            "direct": q_direct,
            "twin_direct": q_twin,
            "component_all": q_all,
        }
        report["v12"] = measure_build(Path(args.build), pairs, policies, split_of)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.full_out:
        full = {"version": VERSION, "pairs": pairs}
        Path(args.full_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.full_out).write_text(json.dumps(full, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("summary", "twin_clusters", "seconds")}, indent=1))
    if "v12" in report:
        print(json.dumps(report["v12"]["counts"], indent=1))
    return 0


def link_kind(pair: dict[str, Any], open_id: str) -> str:
    """Strongest kind of evidence linking ``open_id`` to the other task of a non-DISTINCT pair."""
    r = set(pair["reasons"])
    if pair["verdict"] == TWIN:
        return "twin"
    if any(x.startswith("agrees") for x in r):
        return "agreement"
    open_is_a = pair["a"] == open_id
    if ("a_passes_b_tests" in r and open_is_a) or ("b_passes_a_tests" in r and not open_is_a):
        return "open_reference_passes_protected_tests"
    if r & {"statement_near_duplicate", "statement_similar", "same_code_up_to_renaming"}:
        return "statement_or_code"
    return "protected_reference_passes_open_tests"  # the protected task generalizes the open one


def measure_build(
    build: Path,
    pairs: list[dict[str, Any]],
    policies: dict[str, dict[str, list[str]]],
    split_of: dict[str, str],
) -> dict[str, Any]:
    items = v12_code_items(build)
    counts: dict[str, Any] = {}
    for name, q in policies.items():
        hit = [it for it in items if it["task"] in q]
        counts[name] = {
            "items": len(hit),
            "by_split_family": dict(Counter(f"{it['split']}:{it['family']}" for it in hit)),
            "tasks": len({it["task"] for it in hit}),
            "protected_tasks_involved": sorted({p for it in hit for p in q[it["task"]]}),
        }
    counts["mbpp_items_total"] = dict(Counter(f"{it['split']}:{it['family']}" for it in items))
    present = {it["task"] for it in items}
    links = []
    for p in pairs:
        if p["verdict"] == DISTINCT:
            continue
        for x, y in ((p["a"], p["b"]), (p["b"], p["a"])):
            if x in present and split_of[y] in PROTECTED_SPLITS:
                links.append(
                    {
                        "open": x,
                        "protected": y,
                        "verdict": p["verdict"],
                        "kind": link_kind(p, x),
                        "reasons": p["reasons"],
                    }
                )
    order = ["twin", "agreement", "open_reference_passes_protected_tests", "statement_or_code"]
    best: dict[str, str] = {}
    for link in links:
        k = link["kind"]
        prev = best.get(link["open"])
        rank = order.index(k) if k in order else len(order)
        if prev is None or rank < (order.index(prev) if prev in order else len(order)):
            best[link["open"]] = k
    by_kind_tasks = Counter(best.values())
    by_kind_items = Counter(best[it["task"]] for it in items if it["task"] in best)
    counts["direct_by_strongest_link"] = {"tasks": dict(by_kind_tasks), "items": dict(by_kind_items)}
    counts["mbpp_test_tasks_with_twin_in_v12"] = sorted(
        {
            link["protected"]
            for link in links
            if link["kind"] == "twin" and split_of[link["protected"]] == "test"
        }
    )
    return {"counts": counts, "direct_links": sorted(links, key=lambda x: (x["protected"], x["open"]))}


def _protected_tasks(data_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = load_all_tasks(data_root, cfg, None)
    return [t for t in tasks if t["split"] in ("test", "humaneval")]


def _evalplus(data_root: Path, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plus: dict[str, dict[str, Any]] = {}
    for name, kind in (("humanevalplus", "humaneval"), ("mbppplus", "mbpp")):
        ep = cfg["evalplus"][name]
        path = data_root / ep["file"]
        if _sha256_file(path) != ep["sha256"]:
            raise RuntimeError(f"{path}: bytes differ from the pinned SHA-256")
        card = data_root / ep["card"]
        if _sha256_file(card) != ep["card_sha256"]:
            raise RuntimeError(f"{card}: license evidence differs from the pinned SHA-256")
        plus.update(evalplus_tasks(path, kind))
    return plus


def build_suites(data_root: Path, cfg: dict[str, Any], out: Path, workers: int) -> dict[str, dict[str, Any]]:
    """Build (or reload) the stronger evaluation-only suites for every MBPP test and HumanEval task."""
    if out.exists():
        return {s["task"]: s for s in _read_jsonl(out)}
    tasks = _protected_tasks(data_root, cfg)
    plus = _evalplus(data_root, cfg)
    a = cfg["audit"]

    def one(r: dict[str, Any]) -> dict[str, Any]:
        return strong_suite(
            r,
            plus.get(r["id"]),
            mutation_inputs=int(a["mutation_inputs"]),
            plus_max=int(a["evalplus_max_inputs"]),
        )

    with ThreadPoolExecutor(workers) as pool:
        suites = list(pool.map(one, [t for t in tasks if t.get("entry_point")]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for su in suites:
            f.write(json.dumps(su, sort_keys=True, allow_nan=True) + "\n")
    return {s["task"]: s for s in suites}


def _rate(
    rows: list[dict[str, Any]], key: str, where: Callable[[dict[str, Any]], bool] = lambda _r: True
) -> list[int]:
    sel = [r for r in rows if where(r)]
    return [sum(1 for r in sel if r[key]), len(sel)]


def audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Acceptance counts ``[accepted, total]``. ``combined`` = upstream asserts AND the graded I/O suite
    (the recommended grading); holdout-only rejections measure what the graded suite misses."""
    for r in rows:
        r["combined"] = bool(r["upstream"] and r["strong_all"])
    # demonstrably wrong: fails the upstream asserts or some verified input, including holdout inputs that
    # the graded suite ("all") never uses
    wrong = [r for r in rows if not r["strong_holdout"] or not r["upstream"]]
    summary: dict[str, Any] = {
        "variants": len(rows),
        "tasks": len({r["task"] for r in rows}),
        "accepted_all_variants": {
            k: _rate(rows, k) for k in ("upstream", "strong_own", "strong_plus", "strong_all", "combined")
        },
        "demonstrably_wrong": len(wrong),
        "accepted_demonstrably_wrong": {
            k: _rate(wrong, k) for k in ("upstream", "strong_own", "strong_plus", "strong_all", "combined")
        },
        "not_shown_wrong": sum(1 for r in rows if r["strong_holdout"] and r["upstream"]),
        "wrong_only_on_holdout": sum(1 for r in wrong if r["combined"]),
    }
    by: dict[str, Any] = {}
    for src in ("humaneval", "mbpp"):
        for kind in sorted({r["kind"] for r in rows}):
            sel = [r for r in wrong if r["task"].startswith(src) and r["kind"] == kind]
            if sel:
                by[f"{src}:{kind}"] = {
                    k: _rate(sel, k) for k in ("upstream", "strong_plus", "strong_all", "combined")
                }
    summary["demonstrably_wrong_by_source_kind"] = by
    with_plus = [r for r in wrong if r.get("has_plus")]
    summary["demonstrably_wrong_on_tasks_with_evalplus"] = {
        k: _rate(with_plus, k) for k in ("upstream", "strong_plus", "strong_all", "combined")
    }
    return summary


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config))
    root = Path(args.data_root)
    store = Path(args.store)
    _progress("building or loading the stronger suites")
    suites = build_suites(root, cfg, store / "strong_suites_v1.jsonl", args.workers)
    _progress(f"{len(suites)} suites; scoring wrong-code variants")
    tasks = {t["id"]: t for t in _protected_tasks(root, cfg)}
    plus_ids = set(_evalplus(root, cfg))
    t0 = time.time()

    def one(tid: str) -> list[dict[str, Any]]:
        rows = audit_task(tasks[tid], suites[tid])
        for r in rows:
            r["has_plus"] = tid in plus_ids and suites[tid]["stats"].get("plus_kept", 0) > 0
        return rows

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(one, sorted(suites)) for r in rs]
    with (store / "wrong_code_rows_v1.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    suite_stats: Counter[str] = Counter()
    for su in suites.values():
        suite_stats.update(su["stats"])
        suite_stats["tasks"] += 1
        suite_stats["tasks_with_plus"] += int(su["stats"].get("plus_kept", 0) > 0)
    report = {
        "version": VERSION,
        "code": _code_identity(),
        "seconds": round(time.time() - t0, 1),
        "suites": dict(sorted(suite_stats.items())),
        "wrong_code": audit_summary(rows),
        "store_sha256": {
            name: _sha256_file(store / name)
            for name in ("strong_suites_v1.jsonl", "wrong_code_rows_v1.jsonl")
        },
    }
    if args.pilot_scores:
        report["pilot"] = recheck_pilot(
            Path(args.pilot_scores), Path(args.bundle), suites, tasks, _evalplus(root, cfg)
        )
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1)[:6000])
    return 0


def reading_agreement(
    code: str, record: dict[str, Any], plus: dict[str, Any] | None, mutation_inputs: int = 60
) -> dict[str, list[int]]:
    """How often a program agrees with each reading of a task (original reference, EvalPlus reference) on
    the task's mutated inputs, counting only inputs where that reference returns a value. Used where the
    two references disagree, so neither alone can serve as the oracle."""
    own = harvest_inputs(record)
    plus_inputs = (plus["inputs"] if plus else [])[:50]
    xs = mutate_inputs(own + plus_inputs, mutation_inputs, record["id"])
    entry = record["entry_point"]
    cand = run_io(code, entry, xs)
    refs = {"original": run_io(_runnable(record), entry, xs, config=VERIFY)}
    if plus:
        refs["evalplus"] = run_io(plus["reference"], plus["entry_point"] or entry, xs, config=VERIFY)
    mode = "set" if plus and plus["unordered"] else _compare_mode(record)
    tol = FloatTol(**EVALPLUS_TOL)
    out = {}
    for name, ref in refs.items():
        usable = [(c, r) for c, r in zip(cand, ref, strict=True) if r.ok]
        agree = sum(1 for c, r in usable if c.ok and values_equal(c.value, r.value, tol, mode))
        out[name] = [agree, len(usable)]
    return out


def recheck_pilot(
    scores: Path,
    bundle: Path,
    suites: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    plus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Re-grade every MBPP/HumanEval item the pilot scored as passed with the stronger suites."""
    data = json.loads(scores.read_text())
    records = {r["episode_id"]: r for r in _read_jsonl(bundle / "test_records.jsonl")}
    answers = {r["id"]: r for r in _read_jsonl(bundle / "answers.jsonl")}
    rows = []
    for item in data["items"]:
        if item["family"] not in ("ext.mbpp", "ext.humaneval") or not item["strict"]:
            continue
        tid = task_id_for_record(records[item["id"]]["source_record"])
        assert tid is not None
        spec = answers[item["id"]]["spec"]
        code = sftscore.select_code(item["output"], spec["entry_point"])
        if spec.get("prefix"):
            code = spec["prefix"] + "\n" + code
        upstream_ok, _ = sftscore.score(spec, item["output"])
        row: dict[str, Any] = {"id": item["id"], "task": tid, "upstream_rescore": upstream_ok}
        for name, (ok, detail) in tiered_check(suites[tid], code).items():
            row[f"strong_{name}"] = ok
            if not ok:
                row[f"strong_{name}_detail"] = detail[:80]
        row["tier_sizes"] = dict(Counter(suites[tid]["spec"]["tiers"]))
        stats = suites[tid]["stats"]
        if any("disagrees" in k for k in stats):
            row["reference_conflict"] = {k: v for k, v in stats.items() if "disagrees" in k}
            row["reading_agreement"] = reading_agreement(code, tasks[tid], plus.get(tid))
        rows.append(row)
    return {
        "passes_rechecked": len(rows),
        "wrong_under_plus": sorted(r["task"] for r in rows if not r["strong_plus"]),
        "wrong_under_all": sorted(r["task"] for r in rows if not r["strong_all"]),
        "wrong_under_holdout": sorted(r["task"] for r in rows if not r["strong_holdout"]),
        # where the two references conflict, a program that matches neither reading on every input is wrong
        "wrong_under_every_reading": sorted(
            r["task"]
            for r in rows
            if "reading_agreement" in r and all(a < n for a, n in r["reading_agreement"].values())
        ),
        "wrong": sorted(
            r["task"]
            for r in rows
            if not r["strong_holdout"]
            or ("reading_agreement" in r and all(a < n for a, n in r["reading_agreement"].values()))
        ),
        "items": sorted(rows, key=lambda r: r["task"]),
    }


_PROGRAM_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _function_keys(code: str, min_nodes: int = 20) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and sum(1 for _ in ast.walk(node)) >= min_nodes:
            keys.add(structure_keys(ast.unparse(node))["same_code"])
    return keys


def cmd_recheck(args: argparse.Namespace) -> int:
    """Re-check v1.2 train+dev against every registered code benchmark with the stronger detectors."""
    cfg = _load_config(Path(args.config))
    root, build = Path(args.data_root), Path(args.build)
    report = load_twin_report(Path(args.twins))
    tasks = load_all_tasks(root, cfg, Path(args.jbc_repo) if args.jbc_repo else None)
    split_of = {t["id"]: t["split"] for t in tasks}
    pairs = report["pairs"]
    out: dict[str, Any] = {
        "version": VERSION,
        "code": _code_identity(),
        "twin_report_sha256": _sha256_file(Path(args.twins)),
    }
    rows = {s: _read_jsonl(build / f"{s}.jsonl") for s in ("train", "dev")}
    code_rows = [(s, r) for s in rows for r in rows[s] if r["category"] == "python"]
    mbpp_tasks = {task_id_for_record(str(r.get("source_record"))) for _s, r in code_rows} - {None}
    out["v12_python_items"] = dict(Counter(f"{s}:{r['family']}" for s, r in code_rows))

    # 1. the existing 10-gram layer decision as recorded in the build
    def layer(r: dict[str, Any]) -> str:
        return "self-match" if "self-match" in " ".join(r.get("notes") or []) else "clean"

    out["layer_index_notes"] = dict(Counter(f"{s}:{r['family']}:{layer(r)}" for s, r in code_rows))
    # 2. functional/statement/structure links of v1.2 MBPP tasks to protected and benchmark tasks
    linked: dict[str, set[str]] = defaultdict(set)
    touched: set[str] = set()  # any evidence at all, including weak statement similarity
    for p in pairs:
        for x, y in ((p["a"], p["b"]), (p["b"], p["a"])):
            if x in mbpp_tasks and split_of.get(y) in PROTECTED_SPLITS:
                if p["verdict"] != DISTINCT:
                    linked[y].add(x)
                if p["verdict"] != DISTINCT or p["reasons"]:
                    touched.add(y)
    groups = clusters(pairs, (TWIN,))
    cluster_of = {t: i for i, g in enumerate(groups) for t in g}
    cluster_linked = {
        t for t in split_of if split_of[t] in PROTECTED_SPLITS and t in cluster_of
        and any(m in mbpp_tasks for m in groups[cluster_of[t]])
    }  # fmt: skip
    for bench in ("test", "humaneval", "jbc"):
        prot = [t for t in split_of if split_of[t] == bench]
        out[f"protected_{bench}"] = {
            "tasks": len(prot),
            "linked_direct": len([t for t in prot if t in linked]),
            "linked_cluster": len([t for t in prot if t in cluster_linked]),
        }
    # 3. MBPP+ / HumanEval+ status
    plus = _evalplus(root, cfg)
    mplus = sorted(t for t in plus if t.startswith("mbpp:"))
    in_train = [t for t in mplus if t in mbpp_tasks]
    test_split = [t for t in mplus if split_of.get(t) == "test"]
    clean_v12 = [t for t in test_split if t not in linked and t not in cluster_linked and t not in touched]
    out["mbppplus"] = {
        "tasks": len(mplus),
        "tasks_by_mbpp_split": dict(Counter(split_of.get(t, "?") for t in mplus)),
        "tasks_present_in_v12_train_dev": len(in_train),
        "test_split_tasks": len(test_split),
        "test_split_tasks_linked_to_v12_train_dev": len([t for t in test_split if t in linked]),
        "test_split_tasks_with_any_evidence": len([t for t in test_split if t in touched]),
        "clean_subset_v12_size": len(clean_v12),
        "clean_subset_v12_sha256": hashlib.sha256("\n".join(clean_v12).encode()).hexdigest(),
        "clean_subset_v12": clean_v12,
    }
    hplus = [t for t in plus if t.startswith("humaneval:")]
    out["humanevalplus"] = {
        "tasks": len(hplus),
        "linked_to_v12_train_dev": len([t for t in hplus if t in linked or t in cluster_linked]),
    }
    # 4. program-output items against benchmark reference functions (structure) and AAA program hashes
    bench_keys: dict[str, str] = {}
    for t in tasks:
        if t["split"] in PROTECTED_SPLITS:
            for k in _function_keys(t["code"]):
                bench_keys.setdefault(k, t["id"])
    matches = []
    programs = []
    for s, r in code_rows:
        if r["family"] != "python.output":
            continue
        user = next(m for m in r["episode"]["messages"] if m["role"] == "user")["content"][0]["text"]
        m = _PROGRAM_RE.search(user)
        if not m:
            continue
        programs.append(m.group(1))
        hits = {bench_keys[k] for k in _function_keys(m.group(1)) if k in bench_keys}
        if hits:
            matches.append({"episode_id": r["episode_id"], "split": s, "benchmarks": sorted(hits)})
    out["python_output_function_matches"] = {"programs": len(programs), "matches": matches}
    if args.aaa_repo:
        aaa_hashes: set[str] = set()
        for f in cfg["aaa"]["files"]:
            keys = json.loads(_git_show(Path(args.aaa_repo), cfg["aaa"]["revision"], f))["keys"]
            aaa_hashes |= {
                v["source_sha256"] for v in keys.values() if isinstance(v, dict) and "source_sha256" in v
            }
        variants = {hashlib.sha256(p.encode()).hexdigest() for p in programs}
        variants |= {hashlib.sha256(p.rstrip("\n").encode()).hexdigest() for p in programs}
        out["aaa_program_hash_matches"] = {
            "aaa_programs": len(aaa_hashes),
            "matches": len(variants & aaa_hashes),
        }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "mbppplus"}, indent=1))
    print(json.dumps({k: v for k, v in out["mbppplus"].items() if k != "clean_subset_v12"}, indent=1))
    return 0


def _code_identity() -> dict[str, Any]:
    import subprocess

    from j1data.registry import REPO_ROOT

    def git(*a: str) -> str:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, check=False)
        return r.stdout.strip()

    return {
        "execution_policy": EXECUTION_POLICY,
        "purpose": "functional_diagnostic",
        "limitation": LIMITATION,
        "codecheck_sha256": _sha256_file(Path(__file__)),
        "git_head": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _write_json(path: Path, data: Any) -> int:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(json.dumps(data, indent=1)[:4000])
    return 0


def _progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    ap = argparse.ArgumentParser(prog="python -m j1data.codecheck", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("twins", help="functional-twin report over MBPP (all splits), HumanEval and JBC")
    t.add_argument("--config", default="configs/codecheck_v1.toml")
    t.add_argument("--data-root", required=True)
    t.add_argument("--jbc-repo", help="local clone of the JuniperBench-Code repository (optional)")
    t.add_argument("--build", help="built dataset directory to measure the quarantine on (train/dev jsonl)")
    t.add_argument("--out", required=True)
    t.add_argument("--full-out", help="all pairs with evidence (large; keep off the repository)")
    t.add_argument("--workers", type=int, default=12)
    t.set_defaults(func=cmd_twins)
    au = sub.add_parser("audit", help="stronger suites, wrong-code controls and pilot re-check")
    au.add_argument("--config", default="configs/codecheck_v1.toml")
    au.add_argument("--data-root", required=True)
    au.add_argument("--store", required=True, help="offline evaluation store for protected material (HDD)")
    au.add_argument("--out", required=True, help="summary JSON (no protected material)")
    au.add_argument("--pilot-scores", help="pilot cand_scores.json")
    au.add_argument("--bundle", help="evaluation bundle directory (answers/test_records)")
    au.add_argument("--workers", type=int, default=12)
    au.set_defaults(func=cmd_audit)
    rc_ = sub.add_parser("recheck", help="v1.2 train+dev against registered code benchmarks")
    rc_.add_argument("--config", default="configs/codecheck_v1.toml")
    rc_.add_argument("--data-root", required=True)
    rc_.add_argument("--build", required=True)
    rc_.add_argument("--twins", required=True, help="twin report written by `twins`")
    rc_.add_argument("--jbc-repo")
    rc_.add_argument("--aaa-repo")
    rc_.add_argument("--out", required=True)
    rc_.set_defaults(func=cmd_recheck)
    ex = sub.add_parser("exploits", help="adversarial programs: in-process harness vs this checker")
    ex.add_argument("--out", required=True)
    ex.set_defaults(
        func=lambda a: _write_json(
            Path(a.out), {"version": VERSION, "code": _code_identity(), "rows": exploit_table()}
        )
    )
    args = ap.parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
