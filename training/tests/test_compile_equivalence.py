"""compile_equivalence v2: registered protocol, identity validation, PASS/FAIL/INVALID."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compile_equivalence as E  # noqa: E402

PROTO = json.loads((ROOT / "docs/fable_2026-10-10/compile_equivalence_protocol_v2.json").read_text())
STOP = PROTO["stop_at"]
IDENT = {"kind": "token_stream_v1", "shards_json_sha256": "s" * 64, "tokenizer_json_sha256": "t" * 64, "build": "b",
         "limit_tokens": 3410100225, "seed": 1}  # fmt: skip
EVAL = {"documents": 2576, "scored_bytes": 12311466, "doc_index_sha256": "d" * 64}


def _proto(tmp_path, **over):
    p = dict(PROTO)
    p.update(over)
    f = tmp_path / "protocol.json"
    f.write_text(json.dumps(p))
    return f


def _run(d: Path, name: str, compile_: bool, *, scale=1.0, bpb=1.3, seed=1, code="c" * 64, ident=None, exit_code=0,
         fp_shift=0, dup=False, resumed=False, final_step=STOP, domain_scale=1.0, drop_milestone=False,
         stop_at=STOP, mode="default", nonfinite=False, zero_loss=False, zero_bpb=False,
         zero_domain=False):  # fmt: skip
    cfg = dict(PROTO["required_config"])
    cfg.update({"compile": compile_, "compile_mode": mode, "run_id": name, "seed": seed, "stop_at": stop_at,
                "metrics": f"{name}.jsonl", "summary": f"{name}.json"})  # fmt: skip
    by_source = {
        "big": {"bytes": 1_000_000, "bpb": 0.0 if zero_domain else 1.25 * domain_scale},
        "small": {"bytes": 1_000, "bpb": 1.9 * (1.1 if domain_scale != 1.0 else 1.0)},
    }
    evals = [
        {"event": "eval_detail", "step": s, "bpb": bpb + (0.5 - s / STOP) * 0.1, "by_source": by_source}
        for s in PROTO["milestones"]
        if not (drop_milestone and s == PROTO["milestones"][0])
    ]
    last = evals[-1] if evals else {"step": 0}
    summ = {
        "config": cfg, "last_eval_detail": {**last, "bpb": 0.0 if zero_bpb else bpb, "step": final_step},
        "tokens_per_s_median": 1.0,
        "data_identity": ident or IDENT, "code": {"package_sha256": code}, "model": {"n_layer": 2}, "n_params": 10,
        "eval_set": EVAL, "final_step": final_step, "steps_run": final_step, "first_step_s": 1.0,
        "startup": {"process_elapsed_s": 50.0},
    }  # fmt: skip
    p = d / f"{name}.json"
    p.write_text(json.dumps(summ))
    lines = []
    if resumed:
        lines.append(json.dumps({"event": "restored", "step": 3}))
    for s in range(1, STOP + 1):
        loss = (6 - s / 500) * scale
        if nonfinite and s == 7:
            loss = float("nan")
        if zero_loss and s == 7:
            loss = 0.0
        lines.append(json.dumps({"step": s, "loss": loss, "batch_fp": f"fp{s + (fp_shift if s == 9 else 0)}"}))
        if dup and s == 5:
            lines.append(lines[-1])
    lines += [json.dumps(r) for r in evals]
    p.with_suffix(".jsonl").write_text("\n".join(lines))
    p.with_suffix(".status").write_text(json.dumps({"name": name, "exit_code": exit_code}))
    return p


def _main(tmp_path, e, c, proto=None):
    out = tmp_path / "out.json"
    rc = E.main([str(e), str(c), "--protocol", str(proto or _proto(tmp_path)), "--out", str(out)])
    return rc, json.loads(out.read_text())


def test_pass_records_identity_and_protocol(tmp_path):
    e = _run(tmp_path, "e", False)
    rc, rec = _main(tmp_path, e, _run(tmp_path, "c", True, scale=1.002))
    assert rc == 0 and rec["status"] == "PASS" and rec["schema"] == E.SCHEMA
    assert rec["identity"]["shards_json_sha256"] == "s" * 64 and rec["identity"]["preset"] == "j1-wide-20x768"
    assert rec["identity"]["eval_scored_bytes"] == 12311466 and rec["identity"]["stop_at"] == STOP
    assert rec["tolerances"] == {"loss_rel": 0.005, "bpb_rel": 0.003} and len(rec["protocol"]["sha256"]) == 64
    assert rec["criteria"]["critical_domains_within"] and rec["domains"]["big"]["critical"]
    assert set(rec["milestones"]) == {str(s) for s in PROTO["milestones"]}


def test_fail_on_each_registered_criterion(tmp_path):
    e = _run(tmp_path, "e", False)
    rc, rec = _main(tmp_path, e, _run(tmp_path, "c1", True, scale=1.01))  # mean loss 1 % > 0.5 %
    assert rc == 1 and rec["status"] == "FAIL" and not rec["criteria"]["mean_rel_loss_within"]
    rc, rec = _main(tmp_path, e, _run(tmp_path, "c2", True, bpb=1.31))  # final bpb 0.77 % > 0.3 %
    assert rc == 1 and not rec["criteria"]["final_bpb_within"] and rec["criteria"]["mean_rel_loss_within"]
    rc, rec = _main(tmp_path, e, _run(tmp_path, "c3", True, domain_scale=1.01))  # critical domain 1 % > 0.5 %
    assert rc == 1 and not rec["criteria"]["critical_domains_within"]


def test_invalid_identity_or_completion(tmp_path):
    e = _run(tmp_path, "e", False)
    cases = {
        "seed": _run(tmp_path, "x1", True, seed=2),
        "code": _run(tmp_path, "x2", True, code="z" * 64),
        "data": _run(tmp_path, "x3", True, ident={**IDENT, "shards_json_sha256": "q" * 64}),
        "status": _run(tmp_path, "x4", True, exit_code=1),
        "fingerprint": _run(tmp_path, "x5", True, fp_shift=1),
        "duplicate": _run(tmp_path, "x6", True, dup=True),
        "resumed": _run(tmp_path, "x7", True, resumed=True),
        "incomplete": _run(tmp_path, "x8", True, final_step=STOP - 1),
        "milestone": _run(tmp_path, "x9", True, drop_milestone=True),
        "mode": _run(tmp_path, "x10", True, mode="reduce-overhead"),
        "nonfinite": _run(tmp_path, "x11", True, nonfinite=True),
        "both-eager": _run(tmp_path, "x12", False),
        # R-08: a zero loss / bpb is a malformed record (it would be a relative-difference denominator), never a
        # ZeroDivisionError or a silently None milestone.
        "zero-loss": _run(tmp_path, "x13", True, zero_loss=True),
        "zero-bpb": _run(tmp_path, "x14", True, zero_bpb=True),
        "zero-domain-bpb": _run(tmp_path, "x15", True, zero_domain=True),
    }
    for name, c in cases.items():
        rc, rec = _main(tmp_path, e, c)
        assert rc == 2 and rec["status"] == "INVALID", name
        assert rec["problems"], name
    rc, rec = _main(tmp_path, _run(tmp_path, "e0", False, zero_loss=True), _run(tmp_path, "c0", True, zero_loss=True))
    assert rc == 2 and any("non-positive loss 0.0 at step 7" in p for p in rec["problems"])
    # A loosened protocol is refused outright.
    rc, rec = _main(
        tmp_path, e, _run(tmp_path, "ok", True), proto=_proto(tmp_path, tolerances={"loss_rel": 0.01, "bpb_rel": 0.003})
    )
    assert rc == 2 and "registered" in rec["problems"][0]


def test_required_config_is_enforced(tmp_path):
    e = _run(tmp_path, "e", False)
    c = _run(tmp_path, "c", True)
    for p in (e, c):
        s = json.loads(p.read_text())
        s["config"]["warmup"] = 100
        p.write_text(json.dumps(s))
    rc, rec = _main(tmp_path, e, c)
    assert rc == 2 and any("warmup" in x for x in rec["problems"])
