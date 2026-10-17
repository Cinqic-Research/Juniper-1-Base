"""Score SFT evaluation generations and compare a candidate with the frozen base.

``score``: decode generated token IDs with the canonical tokenizer, parse chat outputs with
``protocol.parse_generation`` and grade the content with the hidden answer specs
(``j1data.sftscore``). Every item is kept (ID, family, outputs, strict and lenient scores).

* strict: the generation is a well-formed ``<|final|> ... <|eot|>`` turn and its text passes.
* lenient: control tokens are dropped and the remaining text is graded (diagnostic).
* few-shot (plain-text base baseline): text up to the first blank line is graded.

``compare``: paired by item ID: per category the mean difference, a paired bootstrap 95% CI
(fixed seed), and an exact two-sided McNemar test, then the pre-registered decision rule.
A sandbox failure stops scoring (an executable item is never scored as failed silently).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from j1data import codecheck, jtok, sandbox
from j1data.hashing import sha256_file, sha256_json
from j1data.protocol import parse_generation
from j1data.sftscore import ABSTAIN_RE, normalize, score

VERSION = "j1.sfteval.v3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def _content(tok: Any, ids: list[int], variant: str) -> tuple[bool, str, str]:
    """(protocol_valid, strict content, lenient content)."""
    plain = tok.decode([t for t in ids if t >= jtok.N_CONTROL], skip_special_tokens=False)
    if variant == "base_fewshot":
        text = plain.split("\n\n")[0].split("\nQuestion:")[0].strip()
        return True, text, text
    parsed = parse_generation(tok, ids)
    finals = [m for m in parsed.messages if m.channel == "final"]
    valid = parsed.stop == "eot" and len(finals) == 1 and len(parsed.messages) == 1
    return valid, (finals[0].content.strip() if valid else ""), plain.strip()


def score_run(
    bundle: Path, generations: Path, tokenizer: Path, expected_tokenizer_sha256: str
) -> dict[str, Any]:
    if sha256_file(tokenizer) != expected_tokenizer_sha256:
        raise SystemExit("tokenizer differs from the bundle's frozen tokenizer")
    if not sandbox.available():
        raise SystemExit("sandbox unavailable: executable items cannot be scored")
    tok = jtok.load(tokenizer.read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    if sha256_json({k: v for k, v in manifest.items() if k != "bundle_sha256"}) != manifest["bundle_sha256"]:
        raise SystemExit("bundle manifest self-hash mismatch")
    if manifest["tokenizer_sha256"] != expected_tokenizer_sha256:
        raise SystemExit("bundle tokenizer identity mismatch")
    if sha256_file(bundle / "answers.jsonl") != manifest["files"]["answers.jsonl"]["sha256"]:
        raise SystemExit("answers.jsonl differs from the bundle manifest")
    if sha256_file(bundle / "prompts.jsonl") != manifest["files"]["prompts.jsonl"]["sha256"]:
        raise SystemExit("prompts.jsonl differs from the bundle manifest")
    answer_rows = read_jsonl(bundle / "answers.jsonl")
    prompt_rows = read_jsonl(bundle / "prompts.jsonl")
    answers = {a["id"]: a for a in answer_rows}
    prompts = {p["id"]: p for p in prompt_rows}
    if (
        len(answers) != len(answer_rows)
        or len(prompts) != len(prompt_rows)
        or answers.keys() != prompts.keys()
    ):
        raise SystemExit("bundle prompts/answers are missing, duplicated or unpaired")
    run = json.loads((generations / "run.json").read_text())
    if run["bundle_sha256"] != manifest["bundle_sha256"]:
        raise SystemExit("generations were made from a different bundle")
    if sha256_file(generations / "generations.jsonl") != run["generations_sha256"]:
        raise SystemExit("generations.jsonl differs from its run record")
    gens = read_jsonl(generations / "generations.jsonl")
    sel = run["selection"]
    expected = {
        i
        for i, p in prompts.items()
        if sel["split"] in ("all", p["split"])
        and sel["variant"] in ("all", p["variant"])
        and (sel["families"] == "all" or p["family"].startswith(tuple(sel["families"].split(","))))
    }
    got = [g["id"] for g in gens]
    if len(got) != len(set(got)) or set(got) != expected:
        raise SystemExit("generations are missing, duplicated or outside the recorded selection")
    items = []
    for g in gens:
        p, a = prompts[g["id"]], answers[g["id"]]
        if g["variant"] != p["variant"]:
            raise SystemExit("generation variant differs from its pinned prompt")
        for field in ("split", "family", "category"):
            if field in g and g[field] != p[field]:
                raise SystemExit(f"generation {field} differs from its pinned prompt")
        valid, strict_text, lenient_text = _content(tok, g["generated_ids"], p["variant"])
        if a["spec"].get("type") == "open_ended":  # rubric review only; never counted as pass or fail
            lenient_ok, detail, strict_ok, unscored = False, "unscored: open-ended", False, True
        else:
            lenient_ok, detail = score(a["spec"], lenient_text)
            strict_ok = valid and score(a["spec"], strict_text)[0]
            unscored = False
        items.append(
            {
                "id": g["id"],
                "split": p["split"],
                "variant": g["variant"],
                "family": p["family"],
                "category": p["category"],
                "protocol_valid": valid,
                "unscored": unscored,
                "strict": bool(strict_ok),
                "lenient": bool(lenient_ok),
                "abstained": bool(ABSTAIN_RE.search(normalize(lenient_text))),
                "stop": g["stop"],
                "new_tokens": g["new_tokens"],
                "output": lenient_text[:600],
                "detail": str(detail)[:200],
            }
        )
    return {
        "schema": "j1.sft-scores.v1",
        "scorer": VERSION,
        "code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("sfteval.py", "sftscore.py", "codecheck.py", "sandbox.py", "protocol.py", "jtok.py")
        },
        "executable_scoring": {
            "policy": codecheck.EXECUTION_POLICY,
            "purpose": "functional_diagnostic",
            "limitation": codecheck.LIMITATION,
        },
        "model": run["model"],
        "bundle_sha256": manifest["bundle_sha256"],
        "generations_sha256": run["generations_sha256"],
        "items": sorted(items, key=lambda r: r["id"]),
        "summary": summarize(items),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in items:
        if r.get("unscored"):
            groups[f"{r['split']}/{r['variant']}/unscored/{r['category']}"].append(r)
            continue
        groups[f"{r['split']}/{r['variant']}/category/{r['category']}"].append(r)
        groups[f"{r['split']}/{r['variant']}/family/{r['family']}"].append(r)
        groups[f"{r['split']}/{r['variant']}/all"].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        n = len(rs)
        ks = sum(r["strict"] for r in rs)
        out[k] = {
            "n": n,
            "strict": ks / n,
            "strict_ci95": wilson(ks, n),
            "lenient": sum(r["lenient"] for r in rs) / n,
            "protocol_valid": sum(r["protocol_valid"] for r in rs) / n,
            "abstained": sum(r["abstained"] for r in rs) / n,
        }
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b (base only) and c (candidate only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def paired(
    base: list[bool], cand: list[bool], *, resamples: int = 10000, seed: int = 20261012
) -> dict[str, Any]:
    n = len(base)
    diffs = [int(c) - int(b) for b, c in zip(base, cand, strict=True)]
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    b_only = sum(1 for b, c in zip(base, cand, strict=True) if b and not c)
    c_only = sum(1 for b, c in zip(base, cand, strict=True) if c and not b)
    return {
        "n": n,
        "base": sum(base) / n,
        "candidate": sum(cand) / n,
        "difference": sum(diffs) / n,
        "bootstrap_ci95": (boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]),
        "base_only": b_only,
        "candidate_only": c_only,
        "mcnemar_p": mcnemar_p(b_only, c_only),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[k]))
        out[k] = running
    return out


def _groups(rows: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for i, r in rows.items():
        by[r["category"]].append(i)
        by["all"].append(i)
        if r["family"].startswith("heldout."):
            by["heldout_" + r["category"]].append(i)
        elif r["category"] == "instruction_following":
            by["instruction_following_in_distribution"].append(i)
    return by


def compare(base: dict[str, Any], cand: dict[str, Any], split: str = "test") -> dict[str, Any]:
    """Two paired views of the candidate's strict chat score:

    * ``versus_base_chat``: the base on the identical chat prompts (it never saw the protocol, so this
      mostly measures format learning);
    * ``versus_base_fewshot``: the base on plain-text 3-shot prompts of the *same items* (its
      capability without the protocol); regression p-values are Holm-adjusted across categories.
    """
    if base["bundle_sha256"] != cand["bundle_sha256"]:
        raise SystemExit("reports scored different bundles")
    bc = {
        r["id"]: r
        for r in base["items"]
        if r["split"] == split and r["variant"] == "chat" and not r.get("unscored")
    }
    bf = {
        r["id"].removesuffix("#fewshot"): r
        for r in base["items"]
        if r["split"] == split and r["variant"] == "base_fewshot"
    }
    cc = {
        r["id"]: r
        for r in cand["items"]
        if r["split"] == split and r["variant"] == "chat" and not r.get("unscored")
    }
    if bc.keys() != cc.keys():
        raise SystemExit("reports do not cover the same chat items")
    out: dict[str, Any] = {"versus_base_chat": {}, "versus_base_fewshot": {}}
    for k, ids in sorted(_groups(cc).items()):
        ids.sort()
        out["versus_base_chat"][k] = paired([bc[i]["strict"] for i in ids], [cc[i]["strict"] for i in ids])
        few = [i for i in ids if i in bf]
        if few:
            out["versus_base_fewshot"][k] = paired(
                [bf[i]["strict"] for i in few], [cc[i]["strict"] for i in few]
            )
    # Holm family: the categories themselves (no overall or overlapping subsets counted twice).
    family = {
        k: v for k, v in out["versus_base_fewshot"].items() if k in {r["category"] for r in cc.values()}
    }
    decreases = {k: v["mcnemar_p"] for k, v in family.items() if v["difference"] < 0}
    adjusted = holm({k: v["mcnemar_p"] for k, v in family.items()})
    for k, v in family.items():
        v["mcnemar_p_holm"] = adjusted[k]
    out["significant_regressions_vs_fewshot"] = sorted(k for k in decreases if adjusted[k] < 0.05)
    fa_ids = sorted(i for i, r in cc.items() if r["category"] == "factuality")
    unc_ids = sorted(i for i, r in cc.items() if r["category"] == "uncertainty")
    out["abstention"] = {
        "uncertainty_strict": sum(cc[i]["strict"] for i in unc_ids) / max(1, len(unc_ids)),
        "false_abstention_on_facts": sum(cc[i]["abstained"] for i in fa_ids) / max(1, len(fa_ids)),
        "base_false_abstention_on_facts": sum(bc[i]["abstained"] for i in fa_ids) / max(1, len(fa_ids)),
        "n_uncertainty": len(unc_ids),
        "n_facts": len(fa_ids),
    }
    out["protocol_valid"] = {
        "base": sum(r["protocol_valid"] for r in bc.values()) / len(bc),
        "candidate": sum(r["protocol_valid"] for r in cc.values()) / len(cc),
        "n": len(cc),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sfteval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--generations", type=Path, required=True)
    s.add_argument("--tokenizer", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        m = json.loads((a.bundle / "manifest.json").read_text())
        rep = score_run(a.bundle, a.generations, a.tokenizer, m["tokenizer_sha256"])
        rep["report_sha256"] = sha256_json(rep)
        a.out.write_text(json.dumps(rep, indent=1, sort_keys=True))
        top = {
            k: round(v["strict"], 4)
            for k, v in rep["summary"].items()
            if "/category/" in k or k.endswith("/all")
        }
        print(json.dumps(top, indent=1))
        return 0
    base, cand = json.loads(a.base.read_text()), json.loads(a.candidate.read_text())
    res = {"base_model": base["model"], "candidate_model": cand["model"], "test": compare(base, cand)}
    res["comparison_sha256"] = sha256_json(res)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    view = res["test"]
    print(
        json.dumps(
            {
                v: {k: round(x["difference"], 4) for k, x in view[v].items()}
                for v in ("versus_base_chat", "versus_base_fewshot")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
