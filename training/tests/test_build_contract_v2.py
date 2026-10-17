"""build_contract_v2: the contract status is derived from the readiness matrix, the equivalence record and the
manifest's profile, never chosen (review T-18); build identities must agree; limitations are a committed input."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_contract_v2 as B  # noqa: E402

D10 = ROOT / "docs/autonomous_2026-10-10"
MAN = json.loads((D10 / "launch_manifest.json").read_text())


def _data_root(tmp: Path, **over) -> Path:
    """The two Data-side records the builder reads, consistent with the committed manifest."""
    d = tmp / "data" / "reports" / "j1-base-v0.2.2"
    d.mkdir(parents=True)
    rc = {"pass": True, "inputs": {"accepted_manifest_sha256": MAN["data"]["accepted_manifest_sha256"],
                                   "build_identity": "b" * 64, "shards_json_sha256": MAN["data"]["shards_json_sha256"],
                                   "tokenizer_json_sha256": MAN["tokenizer"]["sha256"]},
          "splits": {}, "phase_a": {"bpe_sample_checked": 1, "literal_control_docs": 22}}  # fmt: skip
    lin = {"build": MAN["data"]["build"], "build_identity": "b" * 64,
           "manifest_sha256": MAN["data"]["accepted_manifest_sha256"]} | over  # fmt: skip
    (d / "full_recount_cand5-j1tok-24k.json").write_text(json.dumps(rc))
    (d / "successor_lineage.json").write_text(json.dumps(lin))
    return tmp / "data"


def _args(tmp: Path, *, status=None, matrix=D10 / "readiness_matrix.json", data=None):
    lim = ROOT / "docs/fable_2026-10-10/contract_limitations.json"
    args = ["--train-root", str(ROOT), "--data-root", str(data or _data_root(tmp)), "--manifest",
            str(D10 / "launch_manifest.json"), "--readiness-matrix", str(matrix), "--limitations", str(lim),
            "--out", str(tmp / "contract.json")]  # fmt: skip
    return args + (["--status", status] if status else [])


def test_t18_status_frozen_is_refused_over_the_committed_inputs(tmp_path):
    """The stale attempt matrix (as committed at 8f91e88, kept under launch_attempt/) names no gate-file source and
    no manifest binding, so the contract cannot be rebuilt as FROZEN from it whatever the operator types."""
    with pytest.raises(SystemExit, match="--status FROZEN refused"):
        B.main(_args(tmp_path, status="FROZEN", matrix=D10 / "launch_attempt/readiness_matrix.json"))


def test_t18_derive_status():
    man = {"execution_profile": {"compile": False}}
    fresh = {"valid": True, "source_sha256": "g", "bindings": {"manifest_sha256": "m"},
             "stages": {"BASE_PRETRAINING_LAUNCH": {"ready": True, "blocking": []}}}  # fmt: skip
    ok = {"manifest_sha256": "m", "gates_sha256": "g"}
    assert B.derive_status(fresh, man, {"status": "FAIL"}, **ok)[0] == "FROZEN"  # eager profile: FAIL is admissible
    assert B.derive_status(fresh, man, {"status": "PASS"}, **ok)[0] == "FROZEN"
    compiled = {"execution_profile": {"compile": True}}
    assert B.derive_status(fresh, compiled, {"status": "PASS"}, **ok)[0] == "FROZEN"
    assert B.derive_status(fresh, compiled, {"status": "FAIL"}, **ok)[0] == "FROZEN_LAUNCH_BLOCKED"
    stale = {**fresh, "source_sha256": "other"}
    assert B.derive_status(stale, man, {"status": "PASS"}, **ok)[0] == "FROZEN_LAUNCH_BLOCKED"
    foreign = {**fresh, "bindings": {"manifest_sha256": "other"}}
    assert B.derive_status(foreign, man, {"status": "PASS"}, **ok)[0] == "FROZEN_LAUNCH_BLOCKED"
    blocked = {**fresh, "stages": {"BASE_PRETRAINING_LAUNCH": {"ready": False, "blocking": ["B17"]}}}
    st, basis = B.derive_status(blocked, man, {"status": "PASS"}, **ok)
    assert st == "FROZEN_LAUNCH_BLOCKED" and basis["base_launch_blocking"] == ["B17"]
    assert B.derive_status({}, man, {}, **ok)[0] == "FROZEN_LAUNCH_BLOCKED"  # the committed 16-gate matrix shape


def test_t18_build_identities_must_agree(tmp_path):
    data = _data_root(tmp_path, build="j1-base-v0.2.1")
    with pytest.raises(SystemExit, match="build identities disagree"):
        B.main(_args(tmp_path, data=data))


def test_t18_limitations_input_is_validated(tmp_path):
    bad = tmp_path / "lim.json"
    bad.write_text(json.dumps({"limitations": "prose"}))
    args = _args(tmp_path)
    args[args.index("--limitations") + 1] = str(bad)
    with pytest.raises(SystemExit, match="limitations"):
        B.main(args)
