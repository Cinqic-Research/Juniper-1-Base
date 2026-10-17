"""Pre-registered selection rules fail closed and reproduce by independent recomputation."""

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import select_rule as S  # noqa: E402

TOK = {"24k": ("a" * 64, 24576), "28k": ("b" * 64, 28672), "32k": ("c" * 64, 32768)}


def _summary(d: Path, run_id, *, preset="j1-small-12x384", seed=1, bpb=1.0, steps=10, final=None, tok="b" * 64,
             target_bytes=1_000_000, prefix_bytes=1_000_100, scored=50_000, bps=2e6, lr=1e-3, curve=None, first=1.0,
             code="x"):  # fmt: skip
    s = {
        "run_id": run_id,
        "config": {
            "run_id": run_id,
            "steps": steps,
            "seed": seed,
            "preset": preset,
            "seq_len": 2048,
            "micro_bs": 16,
            "accum": 2,
            "lr": lr,
            "warmup": 2,
            "data": "tokens:/d",
        },
        "final_step": steps if final is None else final,
        "last_eval": bpb * 3,
        "last_eval_detail": {"bpb": bpb},
        "eval_set": {"kind": "full_split_eval_v1", "documents": 2583, "scored_bytes": scored},
        "consumed": {"target_bytes": target_bytes, "prefix_bytes": prefix_bytes},
        "data_identity": {"tokenizer_json_sha256": tok, "seed": seed, "limit_tokens": 99},
        "device_name": "NVIDIA A100-SXM4-80GB",
        "code": {"package_sha256": code},
        "n_params": 100,
        "tokens_per_s_median": 1000.0,
        "train_bytes_per_s_median": bps,
        "first_step_s": first,
        "wall_s": 100.0,
        "peak_vram_gib": 1.0,
    }
    p = d / f"{run_id}.json"
    p.write_text(json.dumps(s))
    p.with_suffix(".status").write_text(json.dumps({"name": run_id, "exit_code": 0}))
    pts = curve or [(first + 10, bpb + 0.2), (first + 100, bpb)]
    p.with_suffix(".jsonl").write_text(
        "\n".join(
            [json.dumps({"step": 1, "loss": 5.0})]
            + [
                json.dumps({"event": "eval", "step": i + 1, "train_seconds": t, "eval_bpb": y})
                for i, (t, y) in enumerate(pts)
            ]
        )
    )
    return p


def _eval(d: Path, cand, fallback=0.002, stem_fb=0):
    tok, vocab = TOK[cand]
    e = {
        "tokenizer_json_sha256": tok,
        "total_vocab": vocab,
        "byte_fallback_rate": fallback,
        "domains": {"x": {"roundtrip_failures": 0, "control_ids_from_text": 0}},
        "probes": {k: {"roundtrip_failures": 0, "byte_fallback_tokens": stem_fb} for k in S.STEM_PROBES},
    }
    p = d / f"eval-{cand}.json"
    p.write_text(json.dumps(e))
    return p


def _tok_set(d, bpb, **kw):
    runs, evals = [], {}
    for cand, (b1, b2) in bpb.items():
        tok = TOK[cand][0]
        bytes_kw = kw.get(cand, {})
        runs.append(_summary(d, f"tok-j1tok-{cand}-s1", seed=1, bpb=b1, tok=tok, **bytes_kw))
        runs.append(_summary(d, f"tok-j1tok-{cand}-s2", seed=2, bpb=b2, tok=tok, **bytes_kw))
        evals[f"tok-j1tok-{cand}"] = _eval(d, cand)
    return runs, evals


def test_tokenizer_lowest_bpb_wins_and_recomputes(tmp_path):
    runs, evals = _tok_set(tmp_path, {"24k": (1.10, 1.12), "28k": (1.00, 1.02), "32k": (1.05, 1.03)})
    r = S.select_tokenizer(runs, evals, [1, 2])
    assert r["status"] == "DECIDED" and r["decision"] == "tok-j1tok-28k"
    # Independent recomputation of the pooled relative seed SD.
    groups = [(1.10, 1.12), (1.00, 1.02), (1.05, 1.03)]
    ss = sum(((x - statistics.fmean(g)) / statistics.fmean(g)) ** 2 for g in groups for x in g)
    assert math.isclose(r["pooled_rel_seed_sd"], math.sqrt(ss / 3))
    assert r["pooled_dof"] == 3


def test_tokenizer_tie_goes_to_bytes_per_second_then_smaller_vocab(tmp_path):
    runs, evals = _tok_set(
        tmp_path,
        {"24k": (1.003, 1.003), "28k": (1.0, 1.0), "32k": (1.002, 1.002)},
        **{"24k": {"bps": 3e6}, "32k": {"bps": 3e6}},
    )
    r = S.select_tokenizer(runs, evals, [1, 2])
    assert r["decision"] == "tok-j1tok-24k" and set(r["tied"]) == {"tok-j1tok-24k", "tok-j1tok-28k", "tok-j1tok-32k"}


@pytest.mark.parametrize(
    "mutate,needle",
    [
        ({"28k": {"target_bytes": 1_010_000}}, "consumed training bytes"),
        ({"28k": {"prefix_bytes": 5}}, "prefix source bytes"),
        ({"28k": {"scored": 49_999}}, "validation documents"),
        ({"28k": {"final": 9}}, "incomplete"),
        ({"28k": {"code": "y"}}, "code identity"),
        ({"28k": {"lr": 2e-3}}, "model/batch/lr"),
    ],
)
def test_tokenizer_fails_closed(tmp_path, mutate, needle):
    runs, evals = _tok_set(tmp_path, {"24k": (1.1, 1.1), "28k": (1.0, 1.0), "32k": (1.05, 1.05)}, **mutate)
    r = S.select_tokenizer(runs, evals, [1, 2])
    assert r["status"] == "INVALID" and r["decision"] is None and needle in r["reason"]


def test_tokenizer_missing_seed_and_nonfinite_are_invalid(tmp_path):
    runs, evals = _tok_set(tmp_path, {"24k": (1.1, 1.1), "28k": (1.0, 1.0), "32k": (1.05, 1.05)})
    r = S.select_tokenizer([p for p in runs if not p.name.endswith("-s2.json")], evals, [1, 2])
    assert r["status"] == "INVALID"
    d = json.loads(runs[0].read_text())
    d["last_eval_detail"]["bpb"] = float("nan")
    runs[0].write_text(json.dumps(d))
    assert S.select_tokenizer(runs, evals, [1, 2])["status"] == "INVALID"


def test_tokenizer_hard_requirement_excludes(tmp_path):
    runs, evals = _tok_set(tmp_path, {"24k": (1.1, 1.1), "28k": (1.0, 1.0), "32k": (1.05, 1.05)})
    evals["tok-j1tok-28k"] = _eval(tmp_path, "28k", stem_fb=1)
    assert S.select_tokenizer(runs, evals, [1, 2])["decision"] == "tok-j1tok-32k"
    evals["tok-j1tok-28k"] = _eval(tmp_path, "28k", fallback=0.006)
    assert S.select_tokenizer(runs, evals, [1, 2])["decision"] == "tok-j1tok-32k"


def test_tokenizer_eval_identity_must_match_trained_shards(tmp_path):
    runs, evals = _tok_set(tmp_path, {"24k": (1.1, 1.1), "28k": (1.0, 1.0), "32k": (1.05, 1.05)})
    evals["tok-j1tok-28k"] = _eval(tmp_path, "24k")
    assert S.select_tokenizer(runs, evals, [1, 2])["status"] == "INVALID"


def _noise(d):
    runs, _ = _tok_set(d, {"24k": (1.100, 1.101), "28k": (1.000, 1.001), "32k": (1.050, 1.051)})
    return runs  # tiny seed SD -> band = 0.5 %


def _arch(d, core, deep, wide, seeds=(1,)):
    out = []
    for seed in seeds:
        for preset, (bpb, curve) in (("j1-core-28x640", core), ("j1-deep-30x576", deep), ("j1-wide-20x768", wide)):
            out.append(_summary(d, f"arch-{preset}-s{seed}", preset=preset, seed=seed, bpb=bpb, curve=curve))
    return out


CURVE = lambda b, t: [(11.0, b + 0.3), (t / 2, b + 0.1), (t, b)]  # noqa: E731


def test_architecture_keeps_core_when_no_alternative_clears_band(tmp_path):
    runs = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (0.998, CURVE(0.998, 110)), (1.01, CURVE(1.01, 100)))
    r = S.select_architecture(runs, _noise(tmp_path))
    assert r["status"] == "DECIDED" and r["decision"] == "j1-core-28x640" and r["provisional"]
    assert math.isclose(r["noise_band_rel"], 0.005)


def test_architecture_one_sided_win_is_inconclusive(tmp_path):
    # deep is 2 % better at equal tokens but much slower, so worse at equal compute.
    runs = _arch(
        tmp_path, (1.0, CURVE(1.0, 100)), (0.98, [(11.0, 1.6), (60, 1.4), (300, 0.98)]), (1.01, CURVE(1.01, 90))
    )
    r = S.select_architecture(runs, _noise(tmp_path))
    assert r["status"] == "INCONCLUSIVE" and r["decision"] == "j1-core-28x640"


def test_architecture_two_sided_win_requires_seed_repeats_then_switches(tmp_path):
    good = (0.97, CURVE(0.97, 100))
    runs = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)), good)
    r = S.select_architecture(runs, _noise(tmp_path))
    assert r["status"] == "NEEDS_SEED_REPEATS" and r["decision"] is None
    runs2 = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)), good, seeds=(1, 2))
    r2 = S.select_architecture(runs2, _noise(tmp_path))
    assert r2["status"] == "DECIDED" and r2["decision"] == "j1-wide-20x768"


def test_architecture_refuses_extrapolation_and_bad_inputs(tmp_path):
    # The shortest run (t=100) lies before the other run's first evaluation (t=150): no common support.
    runs = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (1.0, [(150.0, 1.2), (300.0, 1.0)]), (1.0, CURVE(1.0, 100)))
    r = S.select_architecture(runs, _noise(tmp_path))
    assert r["status"] == "INVALID" and "no extrapolation" in r["reason"]
    runs = _arch(
        tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path,
        (1.0, CURVE(1.0, 100)),
        (1.0, CURVE(1.0, 100)),
        (1.0, CURVE(1.0, 100)),
    )
    assert S.select_architecture(runs, [])["status"] == "INVALID"  # no noise evidence


def test_cli_nonzero_exit_on_invalid(tmp_path):
    p = subprocess.run(
        [sys.executable, str(ROOT / "tools/select_rule.py"), "lr", str(tmp_path / "missing.json")], capture_output=True
    )
    assert p.returncode == 2 and json.loads(p.stdout)["status"] == "INVALID"


def test_lr_excludes_diverged_run_but_refuses_other_missing_summaries(tmp_path):
    ok1 = _summary(tmp_path, "lr-a", lr=6e-4, bpb=1.0)
    ok2 = _summary(tmp_path, "lr-b", lr=1.2e-3, bpb=0.9)
    div = tmp_path / "lr-c.json"
    div.with_suffix(".status").write_text(json.dumps({"exit_code": 1}))
    div.with_suffix(".jsonl").write_text(json.dumps({"event": "abort", "reason": "non_finite_grad_norm", "step": 7}))
    r = S.select_lr([ok1, ok2, div])
    assert r["status"] == "DECIDED" and r["decision"] == 1.2e-3
    assert any(x.get("diverged") for x in r["rows"])
    crashed = tmp_path / "lr-d.json"
    crashed.with_suffix(".status").write_text(json.dumps({"exit_code": 137}))
    assert S.select_lr([ok1, ok2, crashed])["status"] == "INVALID"


def test_nonzero_status_mismatched_id_and_mixed_attempts_are_invalid(tmp_path):
    p = _summary(tmp_path, "lr-a")
    q = _summary(tmp_path, "lr-b", lr=2e-3)
    p.with_suffix(".status").write_text(json.dumps({"exit_code": 1}))
    assert S.select_lr([p, q])["status"] == "INVALID"
    p = _summary(tmp_path, "lr-a")
    d = json.loads(p.read_text())
    d["run_id"] = "other"
    p.write_text(json.dumps(d))
    assert S.select_lr([p, q])["status"] == "INVALID"
    p = _summary(tmp_path, "lr-a")
    with p.with_suffix(".jsonl").open("a") as f:
        f.write("\n" + json.dumps({"event": "eval", "step": 1, "train_seconds": 1.0, "eval_bpb": 2.0}))
    assert "strictly increasing" in S.select_lr([p, q])["reason"]


def test_architecture_refuses_resumed_and_missing_required_candidates(tmp_path):
    runs = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)))
    req = ["j1-core-28x640", "j1-deep-30x576", "j1-wide-20x768"]
    assert S.select_architecture(runs, _noise(tmp_path), req)["status"] == "DECIDED"
    r = S.select_architecture(runs[:2], _noise(tmp_path), req)
    assert r["status"] == "INVALID" and "required" in r["reason"]
    with runs[1].with_suffix(".jsonl").open("a") as f:
        f.write("\n" + json.dumps({"event": "restored", "step": 5}))
    assert "resumed" in S.select_architecture(runs, _noise(tmp_path), req)["reason"]


def test_noise_band_requires_three_degrees_of_freedom(tmp_path):
    runs = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)))
    noise = _noise(tmp_path)
    assert S.select_architecture(runs, noise[:4])["status"] == "INVALID"  # two candidates x two seeds = 2 dof


def _set_device(paths, name):
    for p in paths:
        d = json.loads(p.read_text())
        d["device_name"] = name
        p.write_text(json.dumps(d))


def test_tokenizer_refuses_mixed_devices_and_uses_reference_throughput(tmp_path):
    runs, evals = _tok_set(tmp_path, {"24k": (1.003, 1.003), "28k": (1.0, 1.0), "32k": (1.002, 1.002)})
    _set_device(runs, "NVIDIA GeForce RTX 5090")
    _set_device(runs[:1], "NVIDIA A100-SXM4-80GB")
    assert "device" in S.select_tokenizer(runs, evals, [1, 2])["reason"]
    _set_device(runs, "NVIDIA GeForce RTX 5090")
    tp = {}
    for cand, tps in (("24k", 5e5), ("28k", 4e5), ("32k", 4.5e5)):
        p = _summary(tmp_path, f"tp-j1tok-{cand}", steps=1000, final=300)
        d = json.loads(p.read_text())
        d["tokens_per_s_median"], d["device_name"] = tps, "NVIDIA A100-SXM4-80GB"
        p.write_text(json.dumps(d))
        tp[f"tok-j1tok-{cand}"] = p
    for p in runs:  # consumed bytes per target token = 2.0 for every candidate in this fixture
        d = json.loads(p.read_text())
        d["consumed"]["target_tokens"] = d["consumed"]["target_bytes"] // 2
        p.write_text(json.dumps(d))
    r = S.select_tokenizer(runs, evals, [1, 2], tp)
    assert r["status"] == "DECIDED" and r["decision"] == "tok-j1tok-24k"  # all tied; highest A100 bytes/s
    assert all(x["throughput_device"] == "NVIDIA A100-SXM4-80GB" for x in r["rows"])
    del tp["tok-j1tok-32k"]
    assert S.select_tokenizer(runs, evals, [1, 2], tp)["status"] == "INVALID"
    assert "required" in S.select_tokenizer(runs, evals, [1, 2])["reason"]  # 5090 pilots need A100 --tp


def test_architecture_refuses_mixed_devices(tmp_path):
    runs = _arch(tmp_path, (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)), (1.0, CURVE(1.0, 100)))
    _set_device(runs[:1], "NVIDIA GeForce RTX 5090")
    assert "device" in S.select_architecture(runs, _noise(tmp_path))["reason"]
