"""architecture-v2 / ec-budget (protocol 18): decisions, independent recomputation, fail-closed rejection.

Includes regression tests for the independent review of 61ac20a (F4-F6, F8, F9, F11-F14)."""

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import select_rule as S  # noqa: E402

CORE, WIDE = S.CORE, S.WIDE
SEEDS = [21, 22, 23]
SRC = dict.fromkeys(S.CRITICAL, 1000000) | {"libretexts": 118_548, "python-peps": 8_465}
LR = {CORE: 4.6475e-4, WIDE: 1.6971e-3}
UF = 3071
HOST = {"gpu_uuid": "GPU-1", "pod_id": "pod1"}


def _run(d: Path, run_id, *, preset, seed, bpb, steps=S.ET_UPDATES, rate=1.0, first=60.0, lr=None, dom=None,
         code="x", device="NVIDIA A100-SXM4-80GB", data_seed=None, final=None, resumed=False, scored=S.VAL_BYTES,
         stage=None, host=None, detail_step=None, eval_kind="full_split_eval_v1", paid=None, cfg=None,
         jsonl=True):  # fmt: skip
    """rate = updates per second; train seconds follow from it."""
    lr = lr if lr is not None else LR[preset]
    stage = stage or ("fu3-confirm-ec" if run_id.startswith("f") else "fu3-confirm-et")
    by_source = {k: {"bytes": b, "bpb": (dom or {}).get(k, bpb)} for k, b in SRC.items()}
    train_s = (steps - 1) / rate
    config = {"run_id": run_id, "steps": steps, "seed": seed, "preset": preset, "seq_len": 2048, "micro_bs": 16,
              "accum": 2, "lr": lr, "warmup": 100, "min_lr_ratio": 0.1, "wd": 0.1, "wd_policy": "matrices",
              "clip": 1.0, "opt": "fused", "precision": "bf16", "compile": True, "compile_mode": "default",
              "sdpa": "default", "exact_epoch": True, "eval_set": "full-split", "preload_data": True,
              "deterministic": False, "data": "tokens:/root/data/prefix", "tokenizer": "/root/data/tok/tokenizer.json",
              "stage_id": stage} | (cfg or {})  # fmt: skip
    s = {
        "run_id": run_id,
        "config": config,
        "final_step": steps if final is None else final,
        "steps_run": steps if final is None else final,
        "last_eval": bpb * 2.9,
        "last_eval_detail": {"bpb": bpb, "by_source": by_source, "step": detail_step or steps},
        "eval_set": {"kind": eval_kind, "documents": S.VAL_DOCS, "scored_bytes": scored},
        "consumed": {"target_tokens": steps * 65536, "target_bytes": steps * 250_000},
        "data_identity": {
            "kind": "token_stream_v1",
            "tokenizer_json_sha256": "t",
            "build": "j1-base-v0.2.1",
            "seed": seed if data_seed is None else data_seed,
            "limit_tokens": steps * 65536 + 1,
            "batch_size": 16,
        },
        "device_name": device,
        "code": {"package_sha256": code},
        "tokens_per_s_median": rate * 65536,
        "train_seconds": train_s + first,
        "first_step_s": first,
        "wall_s": train_s + first + 120.0,
        "n_params": 143_000_000,
    }
    p = d / f"{run_id}.json"
    p.write_text(json.dumps(s))
    paid_s = paid if paid is not None else train_s + first + 200.0
    end = f"2026-10-10T{18 + int(paid_s // 3600):02d}:{int(paid_s % 3600 // 60):02d}:{int(paid_s % 60):02d}Z"
    if paid_s < 0:
        end = "2026-10-10T17:59:00Z"
    p.with_suffix(".status").write_text(
        json.dumps({"exit_code": 0, "args": "x", "start_utc": "2026-10-10T18:00:00Z", "end_utc": end})
    )
    p.with_suffix(".host.json").write_text(json.dumps(host or HOST))
    ev = [{"event": "eval", "step": 100, "train_seconds": 50.0, "eval_bpb": bpb + 0.3}]
    if resumed:
        ev.append({"event": "restored", "step": 50})
    if jsonl:
        p.with_suffix(".jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    else:
        p.with_suffix(".jsonl").unlink(missing_ok=True)
    return p


def _amend(d: Path, *, faster=WIDE, uf=UF, lr=None, host=None) -> Path:
    a = d / "amendment_18_1.json"
    slower = CORE if faster == WIDE else WIDE
    a.write_text(
        json.dumps(
            {
                "amendment": "18.1",
                "selected_lr": lr or LR,
                "ec_budget": {"faster": faster, "slower": slower, "decision": uf, "host": host or HOST},
            }
        )
    )
    return a


def _set(d, core, wide, ec, *, core_rate=1.0, wide_rate=1.1, ec_time=None, core_dom=None, wide_dom=None,
         core_paid=None, wide_paid=None):  # fmt: skip
    et, ecr = [], []
    t_core = (S.ET_UPDATES - 1) / core_rate
    for i, s in enumerate(SEEDS):
        et.append(_run(d, f"c{s}", preset=CORE, seed=s, bpb=core[i], rate=core_rate,
                       dom=(core_dom or [None] * 3)[i], paid=core_paid))  # fmt: skip
        et.append(_run(d, f"w{s}", preset=WIDE, seed=s, bpb=wide[i], rate=wide_rate,
                       dom=(wide_dom or [None] * 3)[i], paid=wide_paid))  # fmt: skip
        target = t_core * (1 + (ec_time or [0.0] * 3)[i])
        ecr.append(_run(d, f"f{s}", preset=WIDE, seed=s, bpb=ec[i], steps=UF, rate=(UF - 1) / target))
    return et, ecr, _amend(d)


def _sel(et, ecr, am, seeds=SEEDS):
    return S.select_architecture_v2(et, ecr, seeds, am)


def _indep(x, y):
    v = [(b - a) / b for a, b in zip(x, y, strict=True)]
    m = sum(v) / 3
    sd = math.sqrt(sum((t - m) ** 2 for t in v) / 2)
    h = 1.8856 * sd / math.sqrt(3)
    return m, m - h, m + h


WIDE_WINS = ([1.1800, 1.1790, 1.1810], [1.1790, 1.1795, 1.1800], [1.1700, 1.1690, 1.1705])


def test_wide_selected_on_noninferior_tokens_and_better_compute(tmp_path):
    et, ecr, am = _set(tmp_path, *WIDE_WINS)
    r = _sel(et, ecr, am)
    assert r["status"] == "SELECTED" and r["decision"] == WIDE, r
    m, lo, hi = _indep(WIDE_WINS[1], WIDE_WINS[0])
    st = r["candidates"][WIDE]["equal_token"]
    assert math.isclose(st["mean"], m) and math.isclose(st["lower90"], lo) and math.isclose(st["upper90"], hi)
    m2, lo2, _ = _indep(WIDE_WINS[2], WIDE_WINS[0])
    assert math.isclose(r["candidates"][WIDE]["equal_compute"]["mean"], m2) and lo2 > 0
    assert r["faster"] == WIDE and math.isclose(r["equal_compute_seconds"], S.ET_UPDATES - 1)


def test_core_selected_when_superior_on_equal_tokens(tmp_path):
    et, ecr, am = _set(tmp_path, [1.170, 1.171, 1.169], [1.182, 1.183, 1.181], [1.172, 1.173, 1.171])
    r = _sel(et, ecr, am)
    assert r["status"] == "SELECTED" and r["decision"] == CORE and r["basis"].startswith("equal-token superiority")


def test_speed_alone_never_wins_when_wide_is_worse(tmp_path):
    et, ecr, am = _set(tmp_path, [1.170, 1.171, 1.169], [1.1795, 1.1805, 1.1785], [1.158, 1.159, 1.157])
    r = _sel(et, ecr, am)
    assert r["decision"] != WIDE and not r["candidates"][WIDE]["NI_ET"]


EQUIV = ([1.1800, 1.1790, 1.1810], [1.1802, 1.1788, 1.1809], [1.1780, 1.1772, 1.1790])


def test_equivalence_needs_material_paid_efficiency(tmp_path):
    et, ecr, am = _set(tmp_path, *EQUIV, core_paid=2000.0, wide_paid=1900.0)  # wide 5.3 % more targets per paid s
    r = _sel(et, ecr, am)
    assert r["status"] == "ENGINEERING_CHOICE_UNDER_UNCERTAINTY" and r["decision"] == WIDE
    assert "80 %" in r["basis"]
    d2 = tmp_path / "b"
    d2.mkdir()
    et, ecr, am = _set(d2, *EQUIV, core_paid=2000.0, wide_paid=1980.0)  # 1 % only
    r = _sel(et, ecr, am)
    assert r["status"] == "INCONCLUSIVE" and r["decision"] is None


def test_noisy_seeds_are_inconclusive(tmp_path):
    et, ecr, am = _set(tmp_path, [1.180, 1.170, 1.190], [1.170, 1.190, 1.172], [1.175, 1.185, 1.170])
    r = _sel(et, ecr, am)
    assert r["status"] == "INCONCLUSIVE" and r["decision"] is None


def test_critical_domain_regression_blocks_selection(tmp_path):
    bad = [{"stackv2-edu": 1.25}] * 3
    et, ecr, am = _set(tmp_path, *WIDE_WINS, wide_dom=bad)
    r = _sel(et, ecr, am)
    assert not r["candidates"][WIDE]["DOM"] and r["decision"] != WIDE
    assert set(r["critical_domains"]) == set(S.CRITICAL)


def test_ec_time_mismatch_over_two_percent_disables_ec(tmp_path):
    et, ecr, am = _set(tmp_path, *WIDE_WINS, ec_time=[0.0, 0.03, 0.0])
    r = _sel(et, ecr, am)
    assert r["equal_compute_status"] == "UNMATCHED" and r["candidates"][WIDE]["equal_compute"] is None
    assert r["status"] != "SELECTED" or r["basis"].startswith("equal-token superiority")


def _mut_ok(tmp_path):
    return _set(tmp_path, [1.18] * 3, [1.179, 1.1795, 1.178], [1.17, 1.171, 1.169])


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, final=2000), "incomplete"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, code="y"), "code identity"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, device="H100"), "host/GPU"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, host={"gpu_uuid": "G2", "pod_id": "p"}), "host/GPU"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, resumed=True), "resumed"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, jsonl=False), "metrics JSONL required"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, scored=1), "registered full validation"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, eval_kind="first_windows_v1"), "registered full"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, detail_step=2784), "final evaluation at step"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, data_seed=99), "data order seed"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, lr=5e-3), "frozen"),
        (lambda d: _run(d, "w22", preset=WIDE, seed=22, bpb=1.18, cfg={"warmup": 50}), "fixed settings"),
        (lambda d: _run(d, "w22", preset=WIDE, seed=22, bpb=1.18, cfg={"min_lr_ratio": 0.0}), "fixed settings"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=float("nan")), "non-finite"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, dom={"wikimedia": float("nan")}), "non-finite"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, steps=2000), "equal-token"),
        (lambda d: _run(d, "c22", preset=CORE, seed=24, bpb=1.18), "seeds"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, stage="fu3-lr-screen"), "stage id"),
        (lambda d: _run(d, "f22", preset=WIDE, seed=22, bpb=1.17, steps=3190), "amendment 18.1 says"),
        (lambda d: _run(d, "c22", preset=CORE, seed=22, bpb=1.18, paid=-5), "paid run time"),
    ],
)
def test_fail_closed(tmp_path, mutate, needle):
    et, ecr, am = _mut_ok(tmp_path)
    assert _sel(et, ecr, am)["status"] != "INVALID"
    mutate(tmp_path)
    r = _sel(et, ecr, am)
    assert r["status"] == "INVALID" and r["decision"] is None and needle in r["reason"], r["reason"]


def test_rule_level_rejections(tmp_path):
    et, ecr, am = _mut_ok(tmp_path)
    assert _sel(et, ecr, am, seeds=[1, 2, 3])["status"] == "INVALID"  # unregistered seed set
    assert _sel(et, ecr, _amend(tmp_path, uf=3000))["status"] == "INVALID"  # EC steps vs amendment
    assert _sel(et, ecr, _amend(tmp_path, faster=CORE))["status"] == "INVALID"  # faster shape vs amendment
    assert _sel(et, ecr, _amend(tmp_path, lr={CORE: 1e-3, WIDE: LR[WIDE]}))["status"] == "INVALID"
    am2 = _amend(tmp_path)
    (tmp_path / "w21.status").unlink()
    assert _sel(et, ecr, am2)["status"] == "INVALID"


def test_missing_or_wrong_ec_runs_rejected(tmp_path):
    et, ecr, am = _mut_ok(tmp_path)
    assert _sel(et, ecr[:2], am)["status"] == "INVALID"
    slow_ec = _run(tmp_path, "fx", preset=CORE, seed=21, bpb=1.17, steps=UF, stage="fu3-confirm-ec")
    assert _sel(et, [*ecr[:2], slow_ec], am)["status"] == "INVALID"
    (tmp_path / "f21.host.json").unlink()
    assert "host record" in _sel(et, ecr, am)["reason"]


def _scr(d, name, preset, rate, **kw):
    seed, steps = kw.pop("seed", 11), kw.pop("steps", 1396)
    return _run(d, name, preset=preset, seed=seed, bpb=1.2, steps=steps, rate=rate, stage="fu3-lr-screen", **kw)


def test_ec_budget(tmp_path):
    c = [_scr(tmp_path, f"sc{i}", CORE, t) for i, t in enumerate((1.00, 1.02, 1.01, 0.99))]
    w = [_scr(tmp_path, f"sw{i}", WIDE, t) for i, t in enumerate((1.10, 1.11, 1.09, 1.12))]
    r = S.ec_budget(c, w)
    ratio = statistics.median([1.10, 1.11, 1.09, 1.12]) / statistics.median([1.00, 1.02, 1.01, 0.99])
    assert r["status"] == "DECIDED" and r["faster"] == WIDE and r["decision"] == round(2792 * ratio)
    far = [_scr(tmp_path, f"sx{i}", WIDE, 2.0) for i in range(3)]
    assert S.ec_budget(c, far)["status"] == "INVALID"
    assert S.ec_budget(c[:2], w)["status"] == "INVALID"


@pytest.mark.parametrize(
    "kw",
    [{"device": "H100"}, {"steps": 1400}, {"seed": 12}, {"code": "y"}, {"host": {"gpu_uuid": "G2", "pod_id": "p"}}],
)
def test_ec_budget_rejects_mixed_screening(tmp_path, kw):
    c = [_scr(tmp_path, f"sc{i}", CORE, 1.0) for i in range(3)]
    w = [_scr(tmp_path, f"sw{i}", WIDE, 1.1) for i in range(2)] + [_scr(tmp_path, "sw2", WIDE, 1.1, **kw)]
    assert S.ec_budget(c, w)["status"] == "INVALID"


def test_rereview_amendment_host_and_missing_amendment(tmp_path):
    et, ecr, _ = _mut_ok(tmp_path)
    other = _amend(tmp_path, host={"gpu_uuid": "GPU-OTHER", "pod_id": "podX"})
    assert "screening host" in _sel(et, ecr, other)["reason"]
    assert "amendment 18.1 record required" in S.select_architecture_v2(et, ecr, SEEDS, None)["reason"]


def test_rereview_fixed_config_covers_gqa_and_verify(tmp_path):
    et, ecr, am = _mut_ok(tmp_path)
    _run(tmp_path, "w22", preset=WIDE, seed=22, bpb=1.18, cfg={"gqa_kernel": "expand"})
    assert "fixed settings" in _sel(et, ecr, am)["reason"]


@pytest.mark.parametrize("kw", [{"cfg": {"micro_bs": 8, "accum": 4}}, {"resumed": True}, {"jsonl": False}])
def test_rereview_ec_budget_rejects_settings_and_resume(tmp_path, kw):
    c = [_scr(tmp_path, f"sc{i}", CORE, 1.0) for i in range(3)]
    w = [_scr(tmp_path, f"sw{i}", WIDE, 1.1) for i in range(2)] + [_scr(tmp_path, "sw2", WIDE, 1.1, **kw)]
    assert S.ec_budget(c, w)["status"] == "INVALID"
