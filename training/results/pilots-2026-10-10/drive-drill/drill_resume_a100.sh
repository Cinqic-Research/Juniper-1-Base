#!/bin/bash
# Drive-only drill, resumed computation on the A100 (same contract as the reference run; the code is the Drive-restored bundle checkout).
set -u
P=/root/drive-drill; O=/root/drive-drill/verify; PY=/root/venv/bin/python
until grep -q "arch runs done" /root/runs/prep2/arch.log; do sleep 15; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
mkdir -p $O; cd $P/juniper-1-train
echo "start $(date -u +%FT%TZ) code $(git rev-parse HEAD 2>/dev/null)" > $O/summary.txt
(cd $P && sha256sum -c --quiet MANIFEST.sha256 2>/dev/null | tail -3; echo "manifest check rc=${PIPESTATUS[0]} (entries not uploaded are reported missing)") >> $O/summary.txt 2>&1
$PY -m j1bench.train --steps 40 --warmup 5 --ckpt-every-steps 10 --ckpt-every-seconds 900 --stage-dir $O/stage \
  --preset j1-core-28x640 --data tokens:$P/cand-j1tok-28k-prefix-700MB --tokenizer $P/j1tok-28k.tokenizer.json \
  --seq-len 2048 --micro-bs 16 --compile --deterministic --lr 9.2376e-04 --run-id t1 --resume $P/checkpoint/ckpt-000000025-b2150637 \
  --stop-at 27 --device cuda --ckpt-root $O/ckpt --metrics $O/resume.jsonl --summary $O/resume.json > $O/resume.log 2>&1
echo "resume rc=$?" >> $O/summary.txt
$PY - "$P" "$O" >> $O/summary.txt <<'PY'
import json, sys
P, O = sys.argv[1], sys.argv[2]
ref = {r["step"]: r for r in map(json.loads, open(f"{P}/checkpoint/ref.jsonl")) if "batch_fp" in r}
got = [r for r in map(json.loads, open(f"{O}/resume.jsonl")) if "batch_fp" in r]
for r in got:
    a = ref[r["step"]]
    print(json.dumps({"step": r["step"], "batch_fp_equal": r["batch_fp"] == a["batch_fp"], "lr_equal": r["lr"] == a["lr"],
                      "loss_bitwise_equal": r.get("loss_hex") == a.get("loss_hex"), "loss_resumed": r["loss"], "loss_reference": a["loss"],
                      "rel_diff": abs(r["loss"] - a["loss"]) / a["loss"]}))
ok = bool(got) and all(r["batch_fp"] == ref[r["step"]]["batch_fp"] and abs(r["loss"] - ref[r["step"]]["loss"]) / ref[r["step"]]["loss"] <= 0.01 for r in got)
print("DRIVE-ONLY DRILL RESUME:", "PASS" if ok else "FAIL", "(pre-registered: fingerprints exact, losses within 1 %)")
PY
echo "end $(date -u +%FT%TZ)" >> $O/summary.txt; touch $O/DONE
