"""Build, verify and recount the Juniper 1 instruction-SFT dataset and its evaluation bundle.

``python -m j1data.sftbuild build --config configs/sft_v1.toml --root $J1_DATA_ROOT --out DIR``

The build is deterministic: the same config, pinned source bytes, Wikidata snapshot,
tokenizer and code produce byte-identical artifacts (no timestamps inside identities).
Stages:

1. Load pinned sources (hash-verified downloads at fixed revisions) and the Wikidata snapshot.
2. Generate templated families per split with split-specific seeds; split external sources
   by their upstream split or a keyed hash.
3. Cross-split isolation: a task key or normalized conversation that occurs in a held-out
   split is removed from every earlier split (priority test > dev > train).
4. Execute program-output tasks in the sandbox; verify every reference target against the
   scorer that later grades the model (``j1data.sftscore``); run source-specific checks.
5. Decontaminate train/dev against all 16 registered evaluation benchmarks
   (``j1.contamination-layers.v1``) and remove near-duplicates of natural-text items (MinHash).
6. Encode admitted train/dev episodes with ``j1data.sft.encode_text_episode`` and shift them with
   ``j1data.sft.causal_rows`` (one episode per row, never truncated).
7. Write the training artifacts, the separate evaluation bundle (prompt token IDs and hidden
   answer specs in different files), every rejection, statistics and a self-hashed manifest.

``verify`` re-hashes every file, recounts tokens and episodes, re-validates every episode and
re-checks the causal alignment and split isolation independently of the build code path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import sys
import time
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from j1data import codecheck, episodes, jtok, sandbox, sftgen, sftoasst, sftops, sftscore, sftsources
from j1data.acquire import HF_ENDPOINT, download_file, list_hf_files
from j1data.hashing import canonical_json, doc_id, sha256_bytes, sha256_file, sha256_json, unit_interval
from j1data.minhash import LSHConfig, MinHashConfig, MinHasher, band_keys
from j1data.ngrams import word_hashes
from j1data.normalize import signature_words
from j1data.protocol import PROTOCOL, Message, serialize
from j1data.registry import REPO_ROOT
from j1data.sft import IGNORE, SFTAdmissionError, causal_rows, encode_text_episode
from j1data.sftgen import Item

VERSION = "j1.sft-build.v1"
SPLITS = ("train", "dev", "test")
TOKENIZER_REL = "tokenizer/j1-base-v0.2.2/frozen/cand5-j1tok-24k/tokenizer.json"
LAYERS_REL = "exclusions/layers-v1"
LAYERS_MIN_COVERAGE = 0.6  # pre-registered for j1.contamination-layers.v1
# MBPP non-test splits are SFT sources by design; MBPP+ re-hosts MBPP problems from every split.
SELF_MATCH_ALLOWED = {"google-research-datasets/mbpp": {"mbpp", "mbppplus"}}
CODE_FILES = (
    "sftbuild.py",
    "sftgen.py",
    "sftops.py",
    "sftoasst.py",
    "codecheck.py",
    "sftsources.py",
    "sftscore.py",
    "sandbox.py",
    "sft.py",
    "protocol.py",
    "episodes.py",
    "jtok.py",
    "minhash.py",
    "ngrams.py",
    "normalize.py",
    "hashing.py",
    "acquire.py",
    "dedup.py",
)
TOOL_FILES = ("tools/contamination_layers.py",)

CATEGORY = {
    "arith": "math",
    "gsm8k": "math",
    "ext.gsm8k": "math",
    "distill": "math",
    "if": "instruction_following",
    "heldout.if": "instruction_following",
    "heldout.ops": "instruction_following",
    "if.paraphrase": "if_paraphrase",
    "if.compose_heldout": "if_composition_heldout",
    "fmt": "formatting",
    "extract": "formatting",
    "conv": "multi_turn",
    "qa.context": "grounded_qa",
    "chat": "open_ended",
    "json": "json",
    "heldout.json": "json",
    "facts": "factuality",
    "uncertainty": "uncertainty",
    "hierarchy": "hierarchy",
    "injection": "injection",
    "python": "python",
    "ext.humaneval": "python",
    "ext.mbpp": "python",
}
FAMILY_EVIDENCE: dict[str, tuple[str, dict[str, Any]]] = {
    "arith": (
        "VERIFIED_KNOWN_ANSWER",
        {
            "name": "answer_recomputed",
            "oracle": "known_answer",
            "passed": True,
            "detail": "Python integer arithmetic",
        },
    ),
    "if": ("NOT_APPLICABLE", {}),
    "fmt": ("NOT_APPLICABLE", {}),
    "extract": ("NOT_APPLICABLE", {}),
    "conv": ("NOT_APPLICABLE", {}),
    "qa": ("NOT_APPLICABLE", {}),
    "json": ("NOT_APPLICABLE", {}),
    "uncertainty": ("NOT_APPLICABLE", {}),
    "injection": ("NOT_APPLICABLE", {}),
}
FEWSHOT_FAMILIES = ("arith.basic", "if", "json", "facts", "gsm8k")
# Generation budgets by family prefix (fixed; independent of each reference's length).
MAX_NEW = {
    "arith": 64,
    "gsm8k": 320,
    "ext.gsm8k": 320,
    "if": 128,
    "heldout.if": 128,
    "heldout.ops": 128,
    "fmt": 192,
    "extract": 96,
    "conv": 128,
    "qa": 32,
    "chat": 512,
    "json": 128,
    "heldout.json": 128,
    "facts": 32,
    "uncertainty": 32,
    "hierarchy": 48,
    "injection": 192,
    "python.output": 96,
    "python.mbpp": 384,
    "python.mbpp_fix": 384,
    "ext.humaneval": 384,
    "ext.mbpp": 384,
}


def budget_for(family: str) -> int:
    for prefix in sorted(MAX_NEW, key=len, reverse=True):
        if family == prefix or family.startswith(prefix + "."):
            return MAX_NEW[prefix]
    raise KeyError(family)


_T0 = time.monotonic()


def _progress(msg: str) -> None:
    print(f"[sftbuild {time.monotonic() - _T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def category(family: str) -> str:
    for prefix in sorted(CATEGORY, key=len, reverse=True):
        if family == prefix or family.startswith(prefix + "."):
            return CATEGORY[prefix]
    raise KeyError(family)


def _layers_index(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "j1_contamination_layers", REPO_ROOT / "tools/contamination_layers.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Index(root / LAYERS_REL)


def _seed(base: int, *parts: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{':'.join(parts)}".encode()).digest()[:8], "big")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL rows split on b"\\n" only (str.splitlines would also split on U+2028/U+0085 inside strings)."""
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line]


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def conversation_text(item: Item) -> str:
    return "\n".join(t.text for t in item.turns)


# --- sources ----------------------------------------------------------------------------------


def fetch_sources(cfg: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    """Download (or re-verify) every pinned upstream file; returns path and hashes per file."""
    out: dict[str, dict[str, Any]] = {}
    lock: dict[str, str] = cfg["sources"].get("lock", {})
    for name in ("gsm8k", "mbpp", "humaneval"):
        src = cfg["sources"][name]
        wanted = [v for k, v in src.items() if k.endswith("_file")] + list(src.get("train_files", []))
        local = {rel: root / "sft/raw" / name / src["revision"] / rel for rel in wanted}
        if all(
            f"{name}/{rel}" in lock and p.is_file() and sha256_file(p) == lock[f"{name}/{rel}"]
            for rel, p in local.items()
        ):
            for rel, p in local.items():
                out[f"{name}/{rel}"] = {
                    "path": str(p),
                    "sha256": lock[f"{name}/{rel}"],
                    "size": p.stat().st_size,
                }
            continue  # offline: every pinned file present and matching the lock
        listing = {e["path"]: e for e in list_hf_files(src["repo"], src["revision"])}
        for rel in wanted:
            info = listing[rel]
            target = root / "sft/raw" / name / src["revision"] / rel
            upstream = (info.get("lfs") or {}).get("oid")
            if upstream is None:
                raise RuntimeError(f"{name}/{rel}: no LFS SHA-256 to verify against")
            url = f"{HF_ENDPOINT}/datasets/{src['repo']}/resolve/{src['revision']}/{rel}"
            digest = download_file(url, target, int(info["size"]), upstream)
            if f"{name}/{rel}" in lock and digest != lock[f"{name}/{rel}"]:
                raise RuntimeError(f"{name}/{rel}: bytes differ from the configured lock")
            out[f"{name}/{rel}"] = {"path": str(target), "sha256": digest, "size": int(info["size"])}
    return out


def _local(files: dict[str, dict[str, Any]], key: str) -> Path:
    path = Path(files[key]["path"])
    if sha256_file(path) != files[key]["sha256"]:
        raise RuntimeError(f"{key}: bytes changed since download")
    return path


# --- generation and isolation -----------------------------------------------------------------


def _generate(
    cfg: dict[str, Any],
    facts_by_split: dict[str, list[dict[str, Any]]],
    real_labels: frozenset[str],
    relation_labels: dict[str, list[str]],
) -> tuple[dict[str, list[Item]], list[dict[str, Any]]]:
    seed = int(cfg["build"]["seed"])
    out: dict[str, list[Item]] = {s: [] for s in SPLITS}
    notes: list[dict[str, Any]] = []
    for split in SPLITS:
        prepared = [
            {
                "question": f["question"],
                "answer": f["answer"],
                "spec": f["spec"],
                "key": f["key"],
                "record": f["record"],
            }
            for f in facts_by_split[split]
            if not f["case_sensitive"]  # case rules would change a case-sensitive answer (element symbols)
        ]
        fams = (
            sftgen.families(real_labels, prepared, relation_labels)
            | sftops.TRAIN_FAMILIES
            | sftgen.HELDOUT_FAMILIES
            | sftops.TEST_FAMILIES
        )
        for fam, want in cfg["families"][split].items():
            stream = sftgen.stream(fams[fam], _seed(seed, fam, split))
            keys: set[str] = set()
            tries = 0
            while len(keys) < want and tries < 60 * want:
                tries += 1
                item = next(stream)
                if item.key in keys:
                    continue
                keys.add(item.key)
                item.split = split
                out[split].append(item)
            if len(keys) < want:
                notes.append({"family": fam, "split": split, "wanted": want, "generated": len(keys)})
    return out, notes


def _keys(it: Item) -> list[str]:
    return [it.key, *it.extra_keys]


def _prompt(it: Item) -> str:
    """User-visible request text (system/developer configuration excluded), normalized."""
    return _norm("\n".join(t.text for t in it.turns if t.role == "user"))


def _isolate(
    items: dict[str, list[Item]], protected: frozenset[str] = frozenset()
) -> tuple[dict[str, list[Item]], list[dict[str, Any]]]:
    """Remove from earlier splits every item sharing a task key (including keys of embedded tasks,
    e.g. the arithmetic inside a hierarchy item) or a normalized user request with any candidate of a
    later-priority split (test > dev > train), kept or rejected. ``protected`` holds SHA-256 task keys
    of earlier test sets (historical diagnostics): train and dev never contain them."""
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_prompts: set[str] = set()
    kept: dict[str, list[Item]] = {}
    for split in ("test", "dev", "train"):
        kept[split] = []
        local: set[str] = set()
        for it in items[split]:
            prompt = _prompt(it)
            if any(k in seen_keys for k in _keys(it)):
                reason = "task key also in a held-out split"
            elif split != "test" and any(sha256_bytes(k.encode()) in protected for k in _keys(it)):
                reason = "task key in a historical test set"
            elif prompt in seen_prompts:
                reason = "request also in a held-out split"
            elif prompt in local:
                reason = "exact duplicate request within split"
            else:
                reason = None
            if reason:
                rejected.append(_rej(it, split, reason))
                continue
            local.add(prompt)
            kept[split].append(it)
        for it in items[split]:
            seen_keys.update(_keys(it))
            seen_prompts.add(_prompt(it))
    return kept, rejected


def _rej(item: Item, split: str | None, reason: str) -> dict[str, Any]:
    return {
        "family": item.family,
        "key_sha256": sha256_bytes(item.key.encode()),
        "source_id": item.source_id,
        "source_record": item.source_record or None,
        "split": split,
        "reason": reason,
    }


def _execute_outputs(items: list[Item]) -> list[dict[str, Any]]:
    """Fill python.output targets from sandboxed execution."""
    todo = [it for it in items if "needs_execution" in it.meta]

    def run(it: Item) -> sandbox.Result:
        return sandbox.run_python(it.meta["needs_execution"] + "\n")

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, todo))
    rejected: list[dict[str, Any]] = []
    for it, res in zip(todo, results, strict=True):
        if not res.ok or not res.stdout.strip():
            it.meta["rejected"] = "program failed or printed nothing in the sandbox"
            continue
        out = res.stdout.rstrip("\n")
        last = it.turns[-1]
        last.text = f"```\n{out}\n```"
        last.spec = {"type": "stdout", "value": out}
        it.extra_checks.append(
            {
                "name": "output_from_sandboxed_execution",
                "oracle": "execution",
                "passed": True,
                "detail": sandbox.VERSION,
            }
        )
        del it.meta["needs_execution"]
    return rejected


def _verify_targets(items: list[Item]) -> None:
    """Every assistant reference must score 1 under its own spec (executable specs run in the sandbox)."""

    def check(it: Item) -> str | None:
        if "rejected" in it.meta:
            return str(it.meta["rejected"])
        for t in it.turns:
            if t.role != "assistant" or not t.text:
                continue
            if t.spec is None:
                return "assistant turn without an answer specification"
            if t.spec.get("type") == "open_ended":
                continue  # human-written target: admitted by its source filters and rubric review
            try:
                ok, detail = sftscore.score(t.spec, t.text)
            except sftscore.SpecError as e:
                return f"malformed spec: {e}"
            if not ok:
                return f"reference fails its own scorer ({detail})"
        return None

    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(check, items))
    for it, verdict in zip(items, verdicts, strict=True):
        if verdict:
            it.meta["rejected"] = verdict


def _verify_eval_references(items: list[Item]) -> None:
    """External eval specs: canonical solutions must pass in the sandbox (harness check)."""

    def check(it: Item) -> str | None:
        spec = it.turns[-1].spec
        assert spec is not None
        if it.family == "ext.humaneval":
            ok, detail = sftscore.score(spec, f"```python\n{it.meta['canonical_solution']}\n```")
            return None if ok else f"canonical solution fails harness: {detail}"
        return None

    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(check, items))
    for it, verdict in zip(items, verdicts, strict=True):
        if verdict:
            it.meta["rejected"] = verdict


# --- decontamination and near-duplicates ------------------------------------------------------


def _decontaminate(items: list[Item], index: Any) -> None:
    for it in items:
        if "rejected" in it.meta:
            continue
        hit = index.match(conversation_text(it), LAYERS_MIN_COVERAGE)
        it.meta["decon_best"] = 0.0
        if hit is None:
            continue
        benches = (
            set(hit["layer_a_benchmarks"])
            | set(hit["layer_c_benchmarks"])
            | {index.bench[int(i)] for i in hit["layer_a_common_only"]}
        )
        cov = max([*hit["layer_a"].values(), *hit["layer_a_common_only"].values(), 0.0])
        it.meta["decon_best"] = float(cov)
        allowed = SELF_MATCH_ALLOWED.get(it.source_id, set())
        if benches - allowed:
            it.meta["rejected"] = f"contamination: matches {sorted(benches - allowed)}"
        elif benches:
            it.meta["decon_note"] = f"expected self-match with {sorted(benches)} (MBPP non-test source)"


def _task_text(it: Item) -> str:
    return str(it.meta.get("task_text") or it.meta.get("question") or _prompt(it))


def _task_root(it: Item) -> str:
    return f"{it.source_id}:{(it.source_record or it.key).split('#')[0]}"


_STOP = frozenset(
    [
        "write",
        "a",
        "an",
        "the",
        "python",
        "function",
        "to",
        "program",
        "given",
        "in",
        "of",
        "and",
        "for",
        "that",
        "which",
        "is",
        "from",
    ]
)


def canonical_code(code: str) -> str:
    """Code with every locally bound name (functions, arguments, assigned and loop names) renamed in
    order of first binding, so solutions differing only in identifiers compare equal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    bound: dict[str, str] = {}

    def bind(name: str) -> None:
        bound.setdefault(name, f"v{len(bound)}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name)
        elif isinstance(node, ast.arg):
            bind(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bind(node.id)

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            node.id = bound.get(node.id, node.id)
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = bound.get(node.arg, node.arg)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = bound.get(node.name, node.name)
            self.generic_visit(node)
            return node

    return ast.unparse(Rename().visit(tree))


def _statement_grams(it: Item) -> set[tuple[str, ...]]:
    words = [w for w in signature_words(_task_text(it)) if w not in _STOP]
    return {tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2))} or {tuple(words)}


def _mbpp_overlaps(natural: list[Item], protected: list[Item], threshold: float = 0.6) -> None:
    """Exhaustive MBPP check (the sets are small): a train/dev MBPP item is rejected when its problem
    statement (stopwords removed, word 3-grams) has Jaccard >= ``threshold`` with a held-out MBPP item,
    or when its reference code equals a held-out reference after identifier renaming."""
    held = [it for it in protected if it.source_id == "google-research-datasets/mbpp"]
    held_grams = [(_statement_grams(h), h) for h in held]
    held_code = {canonical_code(h.meta["reference_code"]): h for h in held if h.meta.get("reference_code")}
    for it in natural:
        if it.source_id != "google-research-datasets/mbpp" or "rejected" in it.meta:
            continue
        code = it.meta.get("normalized_code")
        if code and canonical_code(code) in held_code:
            it.meta["rejected"] = (
                f"reference code equals held-out {held_code[canonical_code(code)].source_record} up to renaming"
            )
            continue
        grams = _statement_grams(it)
        for hg, h in held_grams:
            union = len(grams | hg)
            if union and len(grams & hg) / union >= threshold:
                it.meta["rejected"] = (
                    f"task statement near-duplicate (Jaccard {len(grams & hg) / union:.2f}) of held-out {h.source_record}"
                )
                break


def _near_duplicates(natural: list[Item], protected: list[Item]) -> None:
    """MinHash LSH (14x8, ~0.72) over the *task statements* of natural-text items (GSM8K question,
    MBPP problem text; word 3-grams); reject a train/dev item that near-duplicates a held-out item or an
    earlier admitted item (confirmed exact Jaccard >= 0.7), or whose MBPP reference code equals a
    held-out item's normalized reference code."""
    hasher = MinHasher(MinHashConfig(num_perm=112, shingle_words=3, seed=1))
    lsh = LSHConfig(bands=14, rows=8)
    _mbpp_overlaps(natural, protected)
    pool = protected + natural
    held_code = {it.meta["reference_code"]: it for it in protected if it.meta.get("reference_code")}
    for it in natural:
        code = it.meta.get("normalized_code")
        if code and code in held_code and "rejected" not in it.meta:
            it.meta["rejected"] = f"reference code identical to held-out {held_code[code].source_record}"
    shingles = [np.unique(hasher.shingles(word_hashes(signature_words(_task_text(it))))) for it in pool]
    sigs = np.stack([hasher.signature(s) for s in shingles], axis=1)
    buckets: dict[int, list[int]] = defaultdict(list)
    for b in range(lsh.bands):
        for i, k in enumerate(band_keys(sigs, b, lsh.rows).tolist()):
            buckets[(b << 60) ^ k].append(i)
    alive_before: set[int] = set(range(len(protected)))
    for i in range(len(protected), len(pool)):
        it = pool[i]
        if "rejected" in it.meta:
            continue
        cands = {
            j
            for b in range(lsh.bands)
            for j in buckets[(b << 60) ^ int(band_keys(sigs[:, i : i + 1], b, lsh.rows)[0])]
        }
        for j in sorted(cands):
            if j == i or j not in alive_before or shingles[i].size == 0:
                continue
            if pool[j].split == it.split and _task_root(pool[j]) == _task_root(it):
                continue  # a derived variant (MBPP bug-fix) of the same upstream task, same split
            inter = np.intersect1d(shingles[i], shingles[j]).size
            union = np.union1d(shingles[i], shingles[j]).size
            if union and inter / union >= 0.7:
                it.meta["rejected"] = (
                    f"near-duplicate (Jaccard {inter / union:.2f}) of "
                    f"{pool[j].family}:{pool[j].source_record}"
                )
                break
        if "rejected" not in it.meta:
            alive_before.add(i)


# --- episodes and encoding ----------------------------------------------------------------------


def _quarantine_twins(natural: list[Item], ccfg: dict[str, Any]) -> None:
    """Train/dev MBPP items whose task is a functional twin or ambiguous near-twin of a protected task
    (committed j1.codecheck.v1 report; finite cross-execution is evidence of overlap, not proof)."""
    path = REPO_ROOT / ccfg["twin_report"]
    if sha256_file(path) != ccfg["twin_report_sha256"]:
        raise SystemExit("codecheck twin report differs from the configured digest")
    q = codecheck.quarantined_tasks(codecheck.load_twin_report(path))
    for it in natural:
        tid = codecheck.task_id_for_record(it.source_record or "")
        if tid in q and "rejected" not in it.meta:
            it.meta["rejected"] = f"functional twin/ambiguous with held-out {q[tid][:3]} (j1.codecheck.v1)"


def _strengthen_code_eval(external: list[Item], root: Path, ccfg: dict[str, Any]) -> None:
    """MBPP/HumanEval test specs become: upstream asserts AND the graded out-of-process I/O suite."""
    path = root / ccfg["strong_suites"]
    if sha256_file(path) != ccfg["strong_suites_sha256"]:
        raise SystemExit("strong suites differ from the configured digest")
    suites = {r["task"]: r for r in read_jsonl(path)}
    for it in external:
        spec = it.turns[-1].spec or {}
        tid = codecheck.task_id_for_record(it.source_record or "")
        if spec.get("type") != "python_tests" or tid not in suites:
            continue
        suite = {"spec": suites[tid]["spec"]}
        graded = {"type": "python_suite", "entry_point": spec["entry_point"], "suite": suite,
                  "tiers": list(ccfg["graded_tiers"])}  # fmt: skip
        if spec.get("prefix"):
            graded["prefix"] = spec["prefix"]
        it.turns[-1].spec = {"type": "all", "parts": [spec, graded]}
        it.meta["code_eval"] = f"asserts + j1.codecheck.v1 suite ({'+'.join(ccfg['graded_tiers'])})"


def to_episode(it: Item, decon_identity: str) -> dict[str, Any]:
    prefix = it.family.split(".")[0]
    factuality, family_check = FAMILY_EVIDENCE.get(prefix, (it.factuality, {}))
    if it.factuality != "NOT_APPLICABLE":
        factuality = it.factuality
    checks = list(it.extra_checks)
    if family_check:
        checks.append(family_check)
    last_spec = it.turns[-1].spec or {}
    executable = last_spec.get("type") in ("python_tests", "stdout")
    if last_spec.get("type") == "open_ended":
        checks += [
            {
                "name": "human_source_filters",
                "oracle": "deterministic_rule",
                "passed": True,
                "detail": "origin/language/label/model-origin/PII/claim/length filters (docs/sft/oasst_v1.md)",
            },
            {
                "name": "source_human_ranking",
                "oracle": "human_review",
                "passed": True,
                "detail": "every assistant turn is the rank-0 reply of the source's volunteer ranking; review_result true",
            },
        ]
    else:
        checks.append(
            {
                "name": "reference_scores_1_under_spec",
                "oracle": "execution" if executable else "deterministic_rule",
                "passed": True,
                "detail": f"{sftscore.VERSION}: {last_spec.get('type')}",
            }
        )
    if it.family == "json" or it.family.startswith("json."):
        checks.append(
            {"name": "json_parses_and_matches", "oracle": "schema", "passed": True, "detail": "json.loads"}
        )
    messages = []
    for t in it.turns:
        m: dict[str, Any] = {
            "role": t.role,
            "content": [
                {
                    "type": "text",
                    "text": t.text,
                    "trust": "trusted" if t.role in ("system", "developer", "assistant") else "untrusted",
                }
            ],
            "train": t.role == "assistant",
        }
        if t.role == "assistant":
            m["channel"] = "final"
        messages.append(m)
    contaminated = "rejected" in it.meta and str(it.meta["rejected"]).startswith("contamination")
    return {
        "schema": "j1.episode.v1",
        "episode_id": doc_id(it.source_id, it.key),
        "stage": "agent-instruction-sft",
        "task_family": it.family,
        "difficulty": it.difficulty,
        "behaviors": it.behaviors,
        "messages": messages,
        "provenance": {
            "origin": it.origin,
            "teacher_record_id": None,
            "source_prompt_id": it.source_record or None,
            "source_license": it.license,
            "created": VERSION,
        },
        "validation": {
            "validator_version": f"{VERSION}+{sftscore.VERSION}",
            "checks": checks,
            "factuality": factuality,
            "accepted": True,
            "acceptance_reason": "all references verified; isolated; decontaminated",
        },
        "decontamination": {
            "index_identity": decon_identity,
            "best_coverage": round(float(it.meta.get("decon_best", 0.0)), 4),
            "passed": not contaminated,
        },
    }


def encode(
    tok: Any, episode: dict[str, Any], cfg: dict[str, Any], decon_identity: str
) -> tuple[list[int], list[int]]:
    enc = encode_text_episode(
        tok,
        episode,
        approved_source_licenses=frozenset(cfg["build"]["approved_licenses"]),
        expected_decontamination_index=decon_identity,
    )
    if len(enc.ids) > int(cfg["build"]["max_tokens"]):
        raise SFTAdmissionError(f"{len(enc.ids)} tokens exceed max_tokens {cfg['build']['max_tokens']}")
    xs, ys = causal_rows([enc], len(enc.ids) - 1, 0, vocab_size=tok.get_vocab_size())
    return xs[0], ys[0]


def _messages(turns: list[sftgen.Turn]) -> list[Message]:
    msgs, turn = [], 0
    for t in turns:
        if t.role == "user":
            turn += 1
        msgs.append(Message(t.role, t.text, "final" if t.role == "assistant" else None, turn))
    return msgs


def _record(
    it: Item, split: str, ep: dict[str, Any], x: list[int] | None, y: list[int] | None
) -> dict[str, Any]:
    content = canonical_json(ep["messages"])
    rec = {
        "episode_id": ep["episode_id"],
        "split": split,
        "family": it.family,
        "category": category(it.family),
        "source_id": it.source_id,
        "source_revision": it.source_revision,
        "source_record": it.source_record or None,
        "license": it.license,
        "rights_decision": RIGHTS_DECISION[it.license],
        "origin": it.origin,
        "task_key_sha256": sha256_bytes(it.key.encode()),
        "task_keys_sha256": sorted(sha256_bytes(k.encode()) for k in _keys(it)),
        "content_sha256": sha256_bytes(content),
        "verification": [c["name"] + ":" + c["oracle"] for c in ep["validation"]["checks"]],
        "factuality": ep["validation"]["factuality"],
        "decontamination_best_coverage": ep["decontamination"]["best_coverage"],
        "notes": [n for n in (it.meta.get("decon_note"), it.meta.get("modification")) if n],
    }
    if x is not None and y is not None:
        rec["tokens"] = len(x) + 1
        rec["target_tokens"] = sum(1 for v in y if v != IGNORE)
    rec["record_sha256"] = sha256_json({**rec, "episode": ep})
    return rec


RIGHTS_DECISION = {
    "Cinqic-generated": "R00 Cinqic generator output "
    "(deterministic computation; no hosted-assistant text as targets)",
    "MIT": "R01/R03 MIT (GSM8K main / HumanEval eval-only)",
    "CC-BY-4.0": "R02 CC-BY-4.0 MBPP (attribution required; reformatted)",
    "CC0-1.0": "R04 Wikidata CC0",
    "Apache-2.0": "R07 OASST1 / R14 EvalPlus (Apache-2.0; attribution and license text in NOTICE)",
    "LicenseRef-gpt-oss-20b-output": "R05 output of a locally run openai/gpt-oss-20b @6cee5e81 (Apache-2.0 weights); "
    "problem text GSM8K (MIT); distillation arms only",
}


def _save_rows(path: Path, rows: list[tuple[list[int], list[int]]], ids: list[str]) -> str:
    lens = np.array([len(x) for x, _ in rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    inputs = np.fromiter((t for x, _ in rows for t in x), dtype=np.int32, count=int(lens.sum()))
    targets = np.fromiter((t for _, y in rows for t in y), dtype=np.int32, count=int(lens.sum()))
    # uint8[n, 16]: a NumPy "S16" array would strip trailing NUL bytes from IDs ending in 00
    episode_ids = np.frombuffer(b"".join(bytes.fromhex(e) for e in ids), dtype=np.uint8).reshape(len(ids), 16)
    from j1data.dedup import save_npz_deterministic

    return save_npz_deterministic(
        path, inputs=inputs, targets=targets, offsets=offsets, episode_ids=episode_ids
    )


def _fewshot_family(family: str) -> bool:
    return any(family == f or family.startswith(f + ".") for f in FEWSHOT_FAMILIES)


def _single_turn(it: Item) -> bool:
    return [t.role for t in it.turns if t.role != "system"] == ["user", "assistant"]


def _fewshot_prompt(tok: Any, shots: list[Item], it: Item) -> list[int]:
    def qa(x: Item) -> tuple[str, str]:
        user = next(t.text for t in x.turns if t.role == "user")
        return user, x.turns[-1].text

    text = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in map(qa, shots))
    text += f"Question: {qa(it)[0]}\nAnswer:"
    return list(tok.encode(text, add_special_tokens=False).ids)


def build(config_path: Path, root: Path, out: Path) -> dict[str, Any]:
    cfg = tomllib.loads(config_path.read_text())
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; a build never overwrites")
    if not sandbox.available():
        raise SystemExit("sandbox unavailable: refusing to build (references must be executed in isolation)")
    tok_path = root / TOKENIZER_REL
    if sha256_file(tok_path) != cfg["build"]["tokenizer_sha256"]:
        raise SystemExit("tokenizer bytes differ from the configured frozen tokenizer")
    tok = jtok.load(tok_path.read_text())
    seed = int(cfg["build"]["seed"])
    files = fetch_sources(cfg, root)
    _progress("sources verified")
    rejected: list[dict[str, Any]] = []

    snap_path = root / cfg["sources"]["wikidata"]["snapshot"]
    snap_sha = sha256_file(snap_path)
    snapshot = json.loads(snap_path.read_text())
    facts, fact_rej = sftsources.wikidata_facts(snapshot)
    rejected += [r | {"split": None} for r in fact_rej]
    real_labels = frozenset(
        lab.casefold()
        for q in snapshot["queries"].values()
        for b in q["bindings"]
        for lab in (b.get("sLabel", {}).get("value", ""), b.get("oLabel", {}).get("value", ""))
        if lab
    )
    wcfg = cfg["sources"]["wikidata"]
    items: dict[str, list[Item]] = {s: [] for s in SPLITS}
    facts_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for f in facts:
        u = unit_interval("j1.sft.v1.facts", f["subject_qid"])
        split = "train" if u < wcfg["split_train"] else "dev" if u < wcfg["split_dev"] else "test"
        it = sftsources.fact_item(f, random.Random(_seed(seed, f["record"])), split, snap_sha)
        if sftsources.fact_leaks(it):
            rejected.append(_rej(it, None, "answer recoverable from the question"))
            continue
        items[split].append(it)
        facts_by_split[split].append(
            {
                "question": it.meta["question"],
                "answer": it.turns[-1].text,
                "spec": it.turns[-1].spec,
                "key": it.key,
                "record": f["record"],
                "case_sensitive": f["case_sensitive"],
            }
        )

    gsrc = cfg["sources"]["gsm8k"]
    g_items, g_rej = sftsources.gsm8k_items(
        _local(files, f"gsm8k/{gsrc['train_file']}"), split="train", revision=gsrc["revision"]
    )
    rejected += [r | {"split": "train"} for r in g_rej]
    g_train: list[Item] = []
    for it in g_items:
        it.split = (
            "dev" if unit_interval("j1.sft.v1.gsm8k", it.source_record) < gsrc["dev_fraction"] else "train"
        )
        (items["dev"] if it.split == "dev" else g_train).append(it)
    # Mix decision (not a quality rejection): a deterministic subset caps GSM8K's share of target tokens.
    g_train.sort(key=lambda i: unit_interval("j1.sft.v1.gsm8k.cap", i.source_record))
    for it in g_train[gsrc["train_cap"] :]:
        rejected.append(_rej(it, "train", "mix cap: GSM8K limited to train_cap episodes"))
    items["train"] += sorted(g_train[: gsrc["train_cap"]], key=lambda i: i.source_record)
    g_eval = sftsources.gsm8k_eval_items(
        _local(files, f"gsm8k/{gsrc['test_file']}"), revision=gsrc["revision"]
    )
    g_eval = sorted(
        random.Random(_seed(seed, "gsm8k-eval")).sample(g_eval, gsrc["test_eval_items"]), key=lambda i: i.key
    )

    msrc = cfg["sources"]["mbpp"]
    for rel in msrc["train_files"]:
        m_items, m_rej = sftsources.mbpp_items(
            _local(files, f"mbpp/{rel}"),
            split="train",
            revision=msrc["revision"],
            upstream_split=Path(rel).stem.split("-")[0],
            with_fix=True,
            seed=seed,
        )
        items["train"] += m_items
        rejected += [r | {"split": "train"} for r in m_rej]
    m_items, m_rej = sftsources.mbpp_items(
        _local(files, f"mbpp/{msrc['dev_file']}"),
        split="dev",
        revision=msrc["revision"],
        upstream_split="validation",
        with_fix=True,
        seed=seed,
    )
    items["dev"] += m_items
    rejected += [r | {"split": "dev"} for r in m_rej]
    m_eval = sftsources.mbpp_eval_items(_local(files, f"mbpp/{msrc['test_file']}"), revision=msrc["revision"])
    hsrc = cfg["sources"]["humaneval"]
    h_eval = sftsources.humaneval_items(
        _local(files, f"humaneval/{hsrc['test_file']}"), revision=hsrc["revision"]
    )

    if "oasst1" in cfg["sources"]:
        o_items, o_ledger = sftoasst.oasst_items(REPO_ROOT / cfg["sources"]["oasst1"]["config"], root)
        keep = ("family", "key_sha256", "source_id", "source_record", "split")
        rejected += [
            {k: r[k] for k in keep} | {"reason": f"oasst1 {r['kind']}: {r['reason']}"} for r in o_ledger
        ]
        train_ok = bool(cfg["sources"]["oasst1"].get("admit_train", True))
        for split, its in o_items.items():
            for it in its:
                if split != "test" and not train_ok:
                    rejected.append(_rej(it, split, cfg["sources"]["oasst1"]["train_rejection"]))
                    continue
                if it.meta.get("factuality_tag") != "CLAIM_LIGHT":
                    rejected.append(
                        _rej(it, split, "oasst1: unverified factual claims (v1.3 admits claim-light only)")
                    )
                    continue
                it.meta["behavior_class"] = it.behaviors[1:]
                it.behaviors = ["direct_answer", "no_tool"]
                items[split].append(it)

    relation_labels: dict[str, list[str]] = {}
    for f in facts:
        relation_labels.setdefault(f["relation"], []).append(f["object_label"])
    relation_labels = {k: sorted(set(v)) for k, v in relation_labels.items()}
    generated, shortfalls = _generate(cfg, facts_by_split, real_labels, relation_labels)
    _progress("families generated")
    for s in SPLITS:
        items[s] += generated[s]
    protected: frozenset[str] = frozenset()
    if "protected_test_records" in cfg["build"]:
        prot = cfg["build"]["protected_test_records"]
        ppath = root / prot["path"]
        if sha256_file(ppath) != prot["sha256"]:
            raise SystemExit("protected test records differ from the configured digest")
        protected = frozenset(k for r in read_jsonl(ppath) for k in r["task_keys_sha256"])
    items, iso_rej = _isolate(items, protected)
    rejected += iso_rej
    _progress("splits isolated")

    every = [it for s in SPLITS for it in items[s]]
    _execute_outputs(every)
    _progress("program outputs executed")
    _verify_targets(every)
    external = g_eval + m_eval + h_eval
    _verify_eval_references(external)
    _progress("references verified")

    index = _layers_index(root)
    decon_identity = "layers-v1:" + sha256_file(root / LAYERS_REL / "index.json") + "+keys:" + sftgen.VERSION
    trainable = items["train"] + items["dev"]
    _decontaminate(trainable, index)
    natural_protected = [it for it in items["test"] + external if it.origin == "human"]
    natural = [it for it in items["dev"] + items["train"] if it.origin == "human"]
    _near_duplicates(natural, natural_protected)
    if "codecheck" in cfg:
        _quarantine_twins(natural, cfg["codecheck"])
        _strengthen_code_eval(external, root, cfg["codecheck"])
        _verify_eval_references(external)  # canonical HumanEval solutions must pass the strengthened specs
    _progress("decontaminated and deduplicated")

    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    rows: dict[str, list[tuple[list[int], list[int]]]] = {"train": [], "dev": []}
    ids: dict[str, list[str]] = {"train": [], "dev": []}
    eval_prompts: list[dict[str, Any]] = []
    eval_answers: list[dict[str, Any]] = []
    shots_pool: dict[str, list[Item]] = defaultdict(list)

    for split in ("train", "dev"):
        for it in items[split]:
            if "rejected" in it.meta:
                rejected.append(_rej(it, split, str(it.meta["rejected"])))
                continue
            ep = to_episode(it, decon_identity)
            problems = episodes.validate(ep)
            if problems:
                rejected.append(_rej(it, split, "episode invalid: " + "; ".join(problems)))
                continue
            try:
                x, y = encode(tok, ep, cfg, decon_identity)
            except SFTAdmissionError as e:
                rejected.append(_rej(it, split, f"admission: {e}"))
                continue
            rec = _record(it, split, ep, x, y)
            records[split].append({**rec, "episode": ep})
            rows[split].append((x, y))
            ids[split].append(ep["episode_id"])
            if split == "train" and _single_turn(it) and _fewshot_family(it.family):
                shots_pool[it.family].append(it)

    for it in items["test"] + external:
        if "rejected" in it.meta:
            rejected.append(_rej(it, "test", str(it.meta["rejected"])))
            continue
        _add_eval(tok, it, "test", eval_prompts, eval_answers, shots_pool, seed)
        records["test"].append(
            {
                "episode_id": doc_id(it.source_id, it.key),
                "split": "test",
                "family": it.family,
                "category": category(it.family),
                "source_id": it.source_id,
                "source_revision": it.source_revision,
                "source_record": it.source_record or None,
                "license": it.license,
                "task_key_sha256": sha256_bytes(it.key.encode()),
                "task_keys_sha256": sorted(sha256_bytes(k.encode()) for k in _keys(it)),
            }
        )
    for it in items["dev"]:
        if "rejected" not in it.meta:
            _add_eval(tok, it, "dev", eval_prompts, eval_answers, shots_pool, seed)

    _progress("episodes encoded")
    artifacts: dict[str, dict[str, Any]] = {}

    def write_jsonl(name: str, rows_: list[dict[str, Any]]) -> None:
        p = out / name
        p.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows_))
        artifacts[name] = {"sha256": sha256_file(p), "size": p.stat().st_size, "rows": len(rows_)}

    notice = (REPO_ROOT / "manifests/sft/NOTICE").read_bytes()
    (out / "NOTICE").write_bytes(notice)
    artifacts["NOTICE"] = {"sha256": sha256_file(out / "NOTICE"), "size": len(notice)}
    for split in ("train", "dev"):
        write_jsonl(f"{split}.jsonl", records[split])
        p = out / f"{split}_rows.npz"
        _save_rows(p, rows[split], ids[split])
        artifacts[p.name] = {"sha256": sha256_file(p), "size": p.stat().st_size, "rows": len(rows[split])}
    rejected.sort(
        key=lambda r: (
            r.get("split") or "",
            r["family"],
            r.get("source_record") or "",
            r.get("key_sha256", ""),
            r["reason"],
        )
    )
    write_jsonl("rejections.jsonl", rejected)

    eval_dir = out / "eval"
    eval_dir.mkdir()
    eval_files: dict[str, dict[str, Any]] = {}
    (eval_dir / "NOTICE").write_bytes(notice)
    eval_files["NOTICE"] = {"sha256": sha256_file(eval_dir / "NOTICE"), "size": len(notice)}
    for name, rows_ in (
        ("prompts.jsonl", eval_prompts),
        ("answers.jsonl", eval_answers),
        ("test_records.jsonl", records["test"]),
    ):
        p = eval_dir / name
        p.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows_))
        eval_files[name] = {"sha256": sha256_file(p), "size": p.stat().st_size, "rows": len(rows_)}
    eval_manifest = {
        "schema": "j1.sft-eval-bundle.v1",
        "version": cfg["build"]["version"].replace("j1-sft", "j1-sft-eval"),
        "protocol": PROTOCOL,
        "tokenizer_sha256": cfg["build"]["tokenizer_sha256"],
        "stop_ids": [jtok.control_id("<|eot|>"), jtok.control_id("<|call|>")],
        "forbidden_ids": sorted(
            {
                jtok.control_id(t)
                for t in ("<|system|>", "<|developer|>", "<|user|>", "<|tool|>", "<|endoftext|>")
            }
            | set(range(12, 64))
        ),
        "files": eval_files,
        "note": "answers.jsonl is read only by the scorer; the generation harness reads prompts.jsonl only",
    }
    eval_manifest["bundle_sha256"] = sha256_json(eval_manifest)
    (eval_dir / "manifest.json").write_text(json.dumps(eval_manifest, indent=1, sort_keys=True))

    stats = statistics(records, rejected, eval_prompts, shortfalls)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True))
    artifacts["stats.json"] = {
        "sha256": sha256_file(out / "stats.json"),
        "size": (out / "stats.json").stat().st_size,
    }
    manifest = {
        "schema": "j1.sft-dataset.v1",
        "version": cfg["build"]["version"],
        "builder": VERSION,
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT))
            if config_path.is_relative_to(REPO_ROOT)
            else str(config_path),
            "sha256": sha256_file(config_path),
        },
        "code_sha256": {
            **{f: sha256_file(REPO_ROOT / "src/j1data" / f) for f in CODE_FILES},
            **{f: sha256_file(REPO_ROOT / f) for f in TOOL_FILES},
        },
        "tokenizer_sha256": cfg["build"]["tokenizer_sha256"],
        "protocol": PROTOCOL,
        "stage": "agent-instruction-sft",
        "row_format": "per-episode causal rows from j1data.sft.causal_rows: inputs = ids[:-1], "
        "targets = labels[1:] (-100 masked); one episode per row; no packing",
        "max_tokens": cfg["build"]["max_tokens"],
        "sources": {k: {"sha256": v["sha256"], "size": v["size"]} for k, v in sorted(files.items())},
        "wikidata_snapshot_sha256": snap_sha,
        "decontamination_index_identity": decon_identity,
        "approved_licenses": cfg["build"]["approved_licenses"],
        "files": artifacts,
        "eval_bundle_sha256": eval_manifest["bundle_sha256"],
        "counts": stats["summary"],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out / "card.md").write_text(card(manifest, stats))
    return manifest


def _add_eval(
    tok: Any,
    it: Item,
    split: str,
    prompts: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    shots_pool: dict[str, list[Item]],
    seed: int,
) -> None:
    eid = doc_id(it.source_id, it.key)
    context = it.turns[:-1]
    ids_ = serialize(tok, _messages(context), add_generation_prompt=True).ids
    ref = it.turns[-1].text
    budget = budget_for(it.family)
    base = {"split": split, "family": it.family, "category": category(it.family), "max_new_tokens": budget}
    prompts.append({"id": eid, "variant": "chat", **base, "prompt_ids": list(ids_)})
    answers.append(
        {
            "id": eid,
            "spec": it.turns[-1].spec,
            "reference": ref or None,
            "turns": len([t for t in it.turns if t.role == "user"]),
        }
    )
    fam = it.family.removeprefix("ext.")  # GSM8K test items take exemplars from GSM8K train
    if split == "test" and _fewshot_family(fam) and shots_pool.get(fam) and _single_turn(it):
        rng = random.Random(_seed(seed, "fewshot", eid))
        shots = rng.sample(shots_pool[fam], 3)
        prompts.append(
            {
                "id": eid + "#fewshot",
                "variant": "base_fewshot",
                **base,
                "prompt_ids": _fewshot_prompt(tok, shots, it),
            }
        )
        answers.append(
            {"id": eid + "#fewshot", "spec": it.turns[-1].spec, "reference": ref or None, "turns": 1}
        )


def statistics(
    records: dict[str, list[dict[str, Any]]],
    rejected: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    shortfalls: list[dict[str, Any]],
) -> dict[str, Any]:
    fam: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        for r in records[split]:
            f = fam.setdefault(
                f"{split}/{r['family']}", {"episodes": 0, "tokens": 0, "target_tokens": 0, "max_tokens": 0}
            )
            f["episodes"] += 1
            f["tokens"] += r["tokens"]
            f["target_tokens"] += r["target_tokens"]
            f["max_tokens"] = max(f["max_tokens"], r["tokens"])
    by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("train", "dev"):
        for r in records[split]:
            by_cat[split][r["category"]] += r["target_tokens"]
    rej_reasons = Counter(f"{r.get('split')}/{r['family']}: {r['reason'].split(':')[0]}" for r in rejected)
    eval_counts = Counter(f"{p['split']}/{p['variant']}/{p['category']}" for p in prompts)
    lic = Counter(f"{s}/{r['license']}" for s in ("train", "dev") for r in records[s])
    summary: dict[str, Any] = {
        split: {
            "episodes": len(records[split]),
            "tokens": sum(r.get("tokens", 0) for r in records[split]),
            "target_tokens": sum(r.get("target_tokens", 0) for r in records[split]),
        }
        for split in ("train", "dev")
    }
    summary["test_eval_items"] = len(records["test"])
    summary["rejected"] = len(rejected)
    return {
        "summary": summary,
        "families": dict(sorted(fam.items())),
        "target_tokens_by_category": {k: dict(sorted(v.items())) for k, v in by_cat.items()},
        "licenses": dict(sorted(lic.items())),
        "rejections_by_reason": dict(sorted(rej_reasons.items())),
        "eval_prompts": dict(sorted(eval_counts.items())),
        "generation_shortfalls": shortfalls,
    }


def card(manifest: dict[str, Any], stats: dict[str, Any]) -> str:
    s = stats["summary"]
    lines = [
        f"# Dataset card: {manifest['version']}",
        "",
        f"Manifest SHA-256 `{manifest['manifest_sha256']}`; evaluation bundle `{manifest['eval_bundle_sha256']}`.",
        f"Tokenizer j1tok-24k `{manifest['tokenizer_sha256'][:16]}…`, protocol `{manifest['protocol']}`, stage `{manifest['stage']}`.",
        "",
        "## Purpose and scope",
        "",
        "Instruction-SFT set for the frozen Juniper 1 base (`juniper1-base-v1`): American-English instruction",
        "following with checkable outputs (including compositional operations, formatting/extraction, multi-turn",
        "conversations and grounded answerable/unanswerable questions where the build includes `j1data.sftops`),",
        "arithmetic and grade-school math, Python (program output, MBPP solutions and bug fixes), Wikidata facts, JSON",
        "output, abstention on unknowable or fictional questions, system/developer priority, and data that contains",
        "forged instructions or role delimiters. Every training target is verified by execution, recomputation, a pinned",
        "source record or a deterministic rule; no target is model-generated. Open-ended writing, tool calls,",
        "analysis-channel reasoning and teacher outputs are out of scope. Version design record:",
        f"`docs/sft/dataset_{manifest['version'].removeprefix('j1-sft-')}.md` (v1.0–v1.2: `docs/sft/dataset_v1.md`).",
        "",
        "## Size",
        "",
        "| split | episodes | tokens | loss-bearing target tokens |",
        "|---|---:|---:|---:|",
        *[
            f"| {k} | {v['episodes']:,} | {v['tokens']:,} | {v['target_tokens']:,} |"
            for k, v in s.items()
            if isinstance(v, dict)
        ],
        f"| test (evaluation bundle only) | {s['test_eval_items']:,} | — | — |",
        "",
        f"Rejected candidates: {s['rejected']:,} (every one recorded in `rejections.jsonl` with its reason).",
        "",
        "## Families (train)",
        "",
        "| family | episodes | target tokens |",
        "|---|---:|---:|",
        *[
            f"| {k.split('/', 1)[1]} | {v['episodes']:,} | {v['target_tokens']:,} |"
            for k, v in stats["families"].items()
            if k.startswith("train/")
        ],
        "",
        "## Sources and rights",
        "",
        "| source | license | use |",
        "|---|---|---|",
        "| Cinqic generators (`j1data.sftgen`) | Cinqic-generated | targets computed by program logic or sandboxed execution; short fixed frames listed below |",
        "| openai/gsm8k `main` @740312ad | MIT, © 2021 OpenAI (Cobbe et al. 2021) | train (calculator annotations re-executed); test subsample eval-only |",
        "| google-research-datasets/mbpp `full` @4bb6404f | CC BY 4.0, Google Research (Austin et al. 2021) | train/prompt/validation (code normalized with ast.unparse; bug-fix variants derived); test eval-only |",
        "| openai/openai_humaneval @7dce6050 | MIT, © OpenAI (Chen et al. 2021) | eval only |",
        "| Wikidata (snapshot, CC0) | CC0 1.0 | facts; subjects split train/dev/test by QID |",
        "| OpenAssistant/oasst1 @fdf72ae0 (when configured) | Apache-2.0 (Kopf et al. 2023) | v1.3: evaluation prompts only (the independent rubric gate rejected it as training data) |",
        "| EvalPlus HumanEval+/MBPP+ (when configured) | Apache-2.0 (Liu et al. 2023) | evaluation-only test inputs of the coding suites |",
        "",
        "Third-party notices and license texts: `NOTICE` (shipped with the dataset and the evaluation bundle).",
        "",
        "Facts derived from Wikidata, CC0 1.0. MBPP examples are adapted (reformatted, mutated for bug-fix tasks) under CC BY 4.0.",
        "Excluded: Anthropic/hh-rlhf, Dolly-15k, Alpaca and any hosted-assistant output (see the rights review).",
        "",
        "Target text is computed (arithmetic notation, transformed inputs, JSON, program output), copied from a source",
        "(GSM8K solutions with their `#### n` line, MBPP code, Wikidata labels), or the single fixed phrase `I don't know.`",
        "(abstention). No other natural-language target text was written in generator code. User prompts are templated",
        "in generator code (loss-masked).",
        "",
        "Train items overlapping MBPP test problems (task-text Jaccard >= 0.7 or identical reference code) are removed;",
        "MBPP non-test items still match MBPP+ (which re-hosts MBPP problems from every split), so MBPP+ is not a clean",
        "benchmark for checkpoints trained on this set.",
        "",
        "## Verification, isolation and contamination",
        "",
        "* Each reference target scores 1 under `j1data.sftscore` (the evaluation scorer); program outputs, MBPP code and",
        "  bug-fix failures are executed in the bubblewrap sandbox (no network, no home, no host secrets).",
        "* Splits: templated tasks by generation stream with task-key and exact-conversation isolation (test > dev > train);",
        "  GSM8K/MBPP by upstream split (GSM8K train 4% keyed dev); facts by subject QID.",
        "* Train/dev are scanned against all registered evaluation benchmarks (`j1.contamination-layers.v1`, all-token",
        "  10-grams, coverage 0.6, distinctive identifiers); only MBPP non-test items may match MBPP/MBPP+ (expected).",
        "* Natural-text items (GSM8K, MBPP) are deduplicated on their task statements (MinHash 14×8 over word 3-grams,",
        "  confirmed Jaccard ≥ 0.7) against held-out items and each other, and MBPP reference code identical to a held-out",
        "  item's is removed. Templated families are isolated by task key, including keys of embedded tasks.",
        "",
        "## Known limitations",
        "",
        "* Templated families teach formats with limited lexical diversity; held-out instruction families measure transfer.",
        "* GSM8K verification covers arithmetic and the final label, not the human-written reasoning logic.",
        "* Fact coverage is narrow (countries, US states, elements); answers are bare labels. Abstention is one phrase.",
        "* No open-ended writing, multi-paragraph answers, tool use or reasoning traces.",
        "",
    ]
    return "\n".join(lines)


# --- independent verification ------------------------------------------------------------------


def _render_independently(tok: Any, episode: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Minimal re-implementation of the juniper-chat-v1 final-only rendering and loss mask, written
    without protocol.serialize or sft.causal_rows so verify does not share their code path."""
    cid = {name: i for i, name in enumerate(jtok.CONTROL_TOKENS)}
    ids: list[int] = []
    labels: list[int] = []
    for m in episode["messages"]:
        text = "".join(part["text"] for part in m["content"])
        ids.append(cid[f"<|{m['role']}|>"])
        labels.append(IGNORE)  # every header in a final-only episode is host-supplied
        if m["role"] == "assistant":
            body = [cid["<|final|>"], *tok.encode(text, add_special_tokens=False).ids, cid["<|eot|>"]]
            ids += body
            labels += body if m["train"] else [IGNORE] * len(body)
        else:
            body = [*tok.encode(text, add_special_tokens=False).ids, cid["<|end|>"]]
            ids += body
            labels += [IGNORE] * len(body)
    return ids[:-1], labels[1:]


def verify(out: Path, tokenizer: Path) -> dict[str, Any]:
    """Recount and re-check a finished build without the build's code path: file hashes, recounts,
    next-token alignment, an independent re-rendering of every row and loss mask from the stored
    episode text, episode validation, split isolation and hidden-answer separation."""
    tok = jtok.load(tokenizer.read_text())
    m = json.loads((out / "manifest.json").read_text())
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    problems: list[str] = []
    if sha256_json(body) != m["manifest_sha256"]:
        problems.append("manifest self-hash mismatch")
    for name, rec in m["files"].items():
        if sha256_file(out / name) != rec["sha256"]:
            problems.append(f"{name} hash mismatch")
    em = json.loads((out / "eval/manifest.json").read_text())
    if (
        em["bundle_sha256"] != m["eval_bundle_sha256"]
        or sha256_json({k: v for k, v in em.items() if k != "bundle_sha256"}) != em["bundle_sha256"]
    ):
        problems.append("eval bundle identity mismatch")
    for name, rec in em["files"].items():
        if sha256_file(out / "eval" / name) != rec["sha256"]:
            problems.append(f"eval/{name} hash mismatch")
    counts: dict[str, Any] = {}
    keys: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        recs = read_jsonl(out / f"{split}.jsonl")
        with np.load(out / f"{split}_rows.npz") as z:
            x, y, off, eids = z["inputs"], z["targets"], z["offsets"], z["episode_ids"]
        if len(recs) != len(off) - 1 or [bytes(e).hex() for e in eids] != [r["episode_id"] for r in recs]:
            problems.append(f"{split}: rows and records disagree")
        target_tokens = int((y != IGNORE).sum())
        for i in range(len(off) - 1):
            xi, yi = x[off[i] : off[i + 1]], y[off[i] : off[i + 1]]
            # causal alignment: every target except the last equals the next input token
            inner = yi[:-1]
            mask = inner != IGNORE
            if not np.array_equal(inner[mask], xi[1:][mask]) or yi[-1] == IGNORE:
                problems.append(f"{split}: episode {i} is not next-token aligned")
                break
            ep = recs[i]["episode"]
            rx, ry = _render_independently(tok, ep)
            if rx != xi.tolist() or ry != yi.tolist():
                problems.append(
                    f"{split}: episode {i} rows/mask differ from an independent rendering of its text"
                )
                break
            if episodes.validate(ep):
                problems.append(f"{split}: episode {i} fails validation")
                break
            if recs[i]["target_tokens"] != int(mask.sum()) + 1:
                problems.append(f"{split}: episode {i} target count differs")
                break
        counts[split] = {
            "episodes": len(recs),
            "tokens": int(len(x) + len(recs)),
            "target_tokens": target_tokens,
        }
        if counts[split] != {k: m["counts"][split][k] for k in ("episodes", "tokens", "target_tokens")}:
            problems.append(f"{split}: recount {counts[split]} differs from manifest {m['counts'][split]}")
        keys[split] = {k for r in recs for k in r["task_keys_sha256"]}  # includes embedded-task keys
    test_keys = {
        k for r in read_jsonl(out / "eval/test_records.jsonl") for k in r.get("task_keys_sha256", [])
    }
    if keys["train"] & keys["dev"] or keys["train"] & test_keys or keys["dev"] & test_keys:
        problems.append("task keys shared between splits")
    prompts = read_jsonl(out / "eval/prompts.jsonl")
    if any("spec" in p or "reference" in p for p in prompts):
        problems.append("hidden answers present in prompts.jsonl")
    return {
        "ok": not problems,
        "problems": problems,
        "recount": counts,
        "manifest_sha256": m["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sftbuild")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", type=Path, default=REPO_ROOT / "configs/sft_v1.toml")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--tokenizer", type=Path, required=True)
    w = sub.add_parser("fetch-wikidata")
    w.add_argument("--snapshot", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build(a.config.resolve(), a.root, a.out)
        print(json.dumps({"manifest_sha256": m["manifest_sha256"], "counts": m["counts"]}, indent=1))
        return 0
    if a.cmd == "verify":
        r = verify(a.out, a.tokenizer)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    if a.snapshot.exists():
        raise SystemExit(f"{a.snapshot} exists; snapshots are immutable")
    snap = sftsources.fetch_wikidata(a.snapshot)
    print(json.dumps({k: len(q["bindings"]) for k, q in snap["queries"].items()}), sha256_file(a.snapshot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
