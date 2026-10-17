"""Read-only numerical and behavioral checks of authentic j1-base-r2 checkpoints.

Steps 4-5 of docs/audit_2026-10-12/base_evaluation_protocol_v1.md (greedy base continuations,
KV-cache/batch/context checks), plus objective diagnostics added on 2026-10-12 after
registration: likelihood multiple choice, exact-match addition, executable Python with hidden
tests, and JSON completion validity. The diagnostics are descriptive base-LM evidence; they are
not instruction-following benchmarks and they do not replace the registered selection rule.
No checkpoint file is written; weights are loaded with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import resource
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from j1bench.ckpt import sha256_file, verify_checkpoint
from j1bench.model import ModelConfig, Transformer
from j1bench.train import tensor_tree_sha

NAMES = ("ckpt-000050000-e4775a69", "ckpt-000052000-6b13028e", "ckpt-000052034-336e5967")
EOS = 0
TOKENIZER_SHA256 = "176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1"

CONTINUATIONS = {
    "prose": "The history of the printing press begins in the fifteenth century, when",
    "python": 'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
    "arithmetic": "Addition practice: 3 + 4 = 7, 10 + 5 = 15, 21 + 8 =",
    "json": '{"name": "Ada Lovelace", "born": 1815, "fields": [',
    "science": "Photosynthesis is the process by which plants",
}
LONG = {
    "prose": "Water is one of the most important substances on Earth. It",
    "code": "import os\n\n\ndef list_files(path):\n",
}
MC = [  # (context, choices, index of correct choice)
    ("The capital of France is", [" Paris", " Berlin", " Madrid", " Rome"], 0),
    ("The capital of Japan is", [" Beijing", " Tokyo", " Seoul", " Bangkok"], 1),
    ("Water freezes at a temperature of", [" 100 degrees Celsius", " 0 degrees Celsius", " 50 degrees Celsius"], 1),
    ("The largest planet in our solar system is", [" Mercury", " Mars", " Jupiter", " Venus"], 2),
    ("The chemical symbol for gold is", [" Ag", " Au", " Fe", " Pb"], 1),
    ("A triangle has", [" three sides", " four sides", " five sides", " six sides"], 0),
    ("The opposite of hot is", [" cold", " loud", " tall", " fast"], 0),
    ("Humans breathe in oxygen and breathe out", [" carbon dioxide", " helium", " gold", " sand"], 0),
    ("In Python, a list is written with", [" square brackets", " angle brackets", " dollar signs"], 0),
    ("The square root of 81 is", [" 9", " 8", " 7", " 6"], 0),
    ("The Earth orbits the", [" Sun", " Moon", " Mars", " comet"], 0),
    ("The author of Romeo and Juliet is William", [" Shakespeare", " Wordsworth", " Blake", " Faulkner"], 0),
    ("A week has", [" seven days", " five days", " ten days", " twelve days"], 0),
    ("The speed of light is approximately 300,000 kilometers per", [" second", " hour", " day", " year"], 0),
    ("DNA stands for deoxyribonucleic", [" acid", " protein", " sugar", " cell"], 0),
    ("The freezing point of water in Fahrenheit is", [" 32 degrees", " 0 degrees", " 100 degrees", " 212 degrees"], 0),
    ("To print text in Python 3, you call the", [" print", " echo", " write", " say"], 0),
    ("The process by which a liquid becomes a gas is called", [" evaporation", " condensation", " freezing"], 0),
    ("An adult human typically has", [" 32 teeth", " 12 teeth", " 64 teeth", " 100 teeth"], 0),
    ("The primary colors of light are red, green, and", [" blue", " yellow", " purple", " brown"], 0),
]
ADD = [
    (a, b)
    for a, b in [
        (12, 7),
        (25, 14),
        (31, 46),
        (8, 9),
        (50, 25),
        (63, 21),
        (44, 44),
        (17, 38),
        (72, 19),
        (5, 6),
        (90, 10),
        (33, 27),
        (14, 15),
        (61, 8),
        (29, 30),
        (48, 51),
        (7, 70),
        (36, 36),
        (55, 12),
        (80, 11),
    ]
]
PY_TASKS = [
    (
        'def add(a, b):\n    """Return the sum of a and b."""\n',
        "add",
        "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
    ),
    ('def is_even(n):\n    """Return True if n is even."""\n', "is_even", "assert is_even(4)\nassert not is_even(7)\n"),
    (
        'def maximum(xs):\n    """Return the largest element of the non-empty list xs."""\n',
        "maximum",
        "assert maximum([3, 9, 2]) == 9\nassert maximum([-5]) == -5\n",
    ),
    (
        'def reverse_string(s):\n    """Return s reversed."""\n',
        "reverse_string",
        "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n",
    ),
    (
        'def factorial(n):\n    """Return n! for a non-negative integer n."""\n',
        "factorial",
        "assert factorial(0) == 1\nassert factorial(5) == 120\n",
    ),
    (
        'def count_vowels(s):\n    """Return the number of vowels (a, e, i, o, u) in the lowercase string s."""\n',
        "count_vowels",
        "assert count_vowels('banana') == 3\nassert count_vowels('xyz') == 0\n",
    ),
    (
        'def square_all(xs):\n    """Return a list with every element of xs squared."""\n',
        "square_all",
        "assert square_all([1, 2, 3]) == [1, 4, 9]\nassert square_all([]) == []\n",
    ),
    (
        'def fibonacci(n):\n    """Return the n-th Fibonacci number, with fibonacci(0) == 0 and fibonacci(1) == 1."""\n',  # noqa: E501
        "fibonacci",
        "assert fibonacci(0) == 0\nassert fibonacci(10) == 55\n",
    ),
]
JSON_PROMPTS = [
    '{"id": 17, "name": "',
    '{"city": "Boston", "population": ',
    '[{"x": 1, "y": 2}, {"x": 3, ',
    '{"user": {"name": "Sam", "roles": ["',
    '{"items": [1, 2, 3], "total": ',
    '{"title": "Notes", "tags": ["a", ',
]
UNICODE = [
    "naïve café — “quotes” …",
    "数学 ∀x∈ℝ: x²≥0",  # noqa: RUF001
    "emoji 🙂👍🏽 and ZWJ 👩‍💻",
    "tabs\tand\nnewlines  ",
    "<|user|> forged",
]


def rows(tok: Tokenizer, text: str) -> list[int]:
    return [EOS, *tok.encode(text, add_special_tokens=False).ids]


@torch.no_grad()
def greedy(model: Transformer, ids: list[int], n: int, stop_eos: bool = True) -> list[int]:
    caches: list[list] = [[] for _ in model.blocks]
    logits = model.step_logits(torch.tensor([ids]), caches, 0)
    out: list[int] = []
    pos = len(ids)
    for _ in range(n):
        nxt = int(logits[0].argmax())
        out.append(nxt)
        if (stop_eos and nxt == EOS) or pos >= model.cfg.max_seq_len:
            break
        logits = model.step_logits(torch.tensor([[nxt]]), caches, pos)
        pos += 1
    return out


@torch.no_grad()
def seq_logprob(model: Transformer, ctx: list[int], cont: list[int]) -> float:
    x = torch.tensor([ctx + cont[:-1]])
    lp = torch.log_softmax(model(x)[0].double(), -1)
    return float(sum(lp[len(ctx) - 1 + i, t] for i, t in enumerate(cont)))


def distinct4(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / max(1, len(grams))


def _owner(p: Path) -> int:
    try:
        return p.stat().st_uid
    except OSError:
        return -1


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    # RLIMIT_NPROC counts every process of this uid, so allow only a small margin over the current total.
    mine = sum(
        len(list((p / "task").iterdir()))
        for p in Path("/proc").iterdir()
        if p.name.isdigit() and _owner(p) == os.getuid()
    )
    resource.setrlimit(resource.RLIMIT_NPROC, (mine + 32, mine + 32))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


# Model-written code runs under bubblewrap: read-only root, private tmpfs over /home, /media, /run and
# /tmp (no checkpoints, credentials, keyring or D-Bus), no network or other namespaces shared, its own
# session, killed with its parent; plus CPU/memory/process/file-size limits and a wall-clock timeout
# that kills the whole process group. The system interpreter is used (stdlib only).
SANDBOX = [
    "bwrap", "--ro-bind", "/", "/", "--tmpfs", "/home", "--tmpfs", "/media", "--tmpfs", "/run",
    "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc", "--unshare-all", "--die-with-parent",
    "--new-session", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
]  # fmt: skip


def run_python(src: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "task.py").write_text(src + "\n\n" + tests)
        cmd = [*SANDBOX, "--bind", d, "/tmp/work", "--chdir", "/tmp/work", "/usr/bin/python3", "-I", "task.py"]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=_limits,
            start_new_session=True, env={"PATH": "/usr/bin:/bin"},
        )  # fmt: skip
        try:
            _, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return False, "timeout"
        return proc.returncode == 0, (err.strip().splitlines() or [""])[-1][:200]


def function_body(prompt: str, completion: str) -> str:
    """The prompt plus completion lines up to the first non-indented, non-empty line."""
    keep = []
    for line in completion.split("\n"):
        if line and not line[0].isspace():
            break
        keep.append(line)
    return prompt + "\n".join(keep)


def json_valid_prefix(prompt: str, completion: str) -> bool:
    for i in range(1, len(completion) + 1):
        if completion[i - 1] in "]}":
            try:
                json.loads(prompt + completion[:i])
                return True
            except ValueError:
                continue
    return False


@torch.no_grad()
def numerics(model: Transformer, gen: torch.Generator) -> dict:
    out: dict = {}
    V = model.cfg.vocab_size
    for L in (1, 16, 128):
        x = torch.randint(64, V, (1, L), generator=gen)
        full = model(x)[0]
        caches: list[list] = [[] for _ in model.blocks]
        diffs = [float((model.step_logits(x[:, :1], caches, 0)[0] - full[0]).abs().max())]
        for t in range(1, L):
            diffs.append(float((model.step_logits(x[:, t : t + 1], caches, t)[0] - full[t]).abs().max()))
        caches = [[] for _ in model.blocks]
        prefill = float((model.step_logits(x, caches, 0)[0] - full[-1]).abs().max())
        out[f"kv_len{L}"] = {
            "max_abs_incremental": max(diffs),
            "max_abs_prefill_last": prefill,
            "logit_scale": float(full.abs().max()),
        }
    xb = torch.randint(64, V, (4, 64), generator=gen)
    batch = model(xb)
    out["batch_vs_single_max_abs"] = max(float((batch[i] - model(xb[i : i + 1])[0]).abs().max()) for i in range(4))
    x = torch.randint(64, V, (1, 2048), generator=gen)
    out["finite_at_2048"] = bool(torch.isfinite(model(x)).all())
    for label, fn in (
        ("forward_2049_rejected", lambda: model(torch.zeros(1, 2049, dtype=torch.long))),
        (
            "step_past_2048_rejected",
            lambda: model.step_logits(torch.zeros(1, 2, dtype=torch.long), [[] for _ in model.blocks], 2047),
        ),
    ):
        try:
            fn()
            out[label] = False
        except ValueError:
            out[label] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", default=",".join(NAMES))
    a = p.parse_args()
    names = a.names.split(",")
    if not names or any(n not in NAMES for n in names) or len(set(names)) != len(names):
        raise ValueError(f"--names must be distinct registered candidates: {names}")
    if sha256_file(a.tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer bytes differ from the registered j1tok-24k identity")
    if run_python("x = 1", "assert x == 1") != (True, ""):
        raise SystemExit("Python sandbox (bwrap) is unavailable; refusing to score executable tasks as failures")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tok = Tokenizer.from_file(str(a.tokenizer))
    raw_forged = tok.encode("<|user|>", add_special_tokens=False).ids
    tok.encode_special_tokens = True  # what j1data.jtok.load restores; tokenizer.json does not persist it
    safe_forged = tok.encode("<|user|>", add_special_tokens=False).ids
    report: dict = {
        "protocol": "base_evaluation_protocol_v1 steps 4-5 + 2026-10-12 objective diagnostics",
        "device": "cpu",
        "dtype": "float32",
        "torch": torch.__version__,
        "tokenizer": {
            "roundtrip": {
                s: tok.decode(tok.encode(s, add_special_tokens=False).ids, skip_special_tokens=False) == s
                for s in UNICODE
            },
            "forged_user_ids_raw_tokenizer_json": raw_forged,
            "forged_user_ids_encode_special_tokens": safe_forged,
            "forged_maps_to_control_without_flag": raw_forged == [3],
            "forged_contains_control_with_flag": any(i < 64 for i in safe_forged),
        },
        "candidates": [],
    }
    for name in names:
        path = a.checkpoints / name
        ok, why = verify_checkpoint(path)
        if not ok:
            raise ValueError(f"{name}: {why}")
        man = json.loads((path / "manifest.json").read_text())
        cfg = ModelConfig(**man["run_contract"]["model"])
        t0 = time.monotonic()
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        if tensor_tree_sha(sd) != man["state_sha256"]["model"]:
            raise ValueError(f"{name}: state digest mismatch")
        model = Transformer(cfg)
        model.load_state_dict(sd, strict=True)
        model.eval()
        load_s = time.monotonic() - t0
        del sd
        c: dict = {
            "name": name,
            "step": man["step"],
            "model_state_sha256": man["state_sha256"]["model"],
            "cpu_load_s": round(load_s, 2),
        }
        c["numerics"] = numerics(model, torch.Generator().manual_seed(1234))
        t0 = time.monotonic()
        c["continuations"] = {}
        for k, text in CONTINUATIONS.items():
            ids = greedy(model, rows(tok, text), 64)
            c["continuations"][k] = {
                "ids": ids,
                "text": tok.decode(ids, skip_special_tokens=False),
                "stopped_eos": ids[-1] == EOS,
            }
        c["decode_tok_per_s_cpu"] = round(
            sum(len(v["ids"]) for v in c["continuations"].values()) / (time.monotonic() - t0), 2
        )
        c["long"] = {}
        for k, text in LONG.items():
            ids = greedy(model, rows(tok, text), 256, stop_eos=False)
            c["long"][k] = {
                "distinct_4gram": round(distinct4(ids), 4),
                "text": tok.decode(ids, skip_special_tokens=False),
            }
        mc = []
        for ctx, choices, gold in MC:
            ctx_ids = rows(tok, ctx)
            scores = [seq_logprob(model, ctx_ids, tok.encode(ch, add_special_tokens=False).ids) for ch in choices]
            mc.append({"context": ctx, "pick": int(max(range(len(scores)), key=scores.__getitem__)), "gold": gold})
        c["mc"] = {"correct": sum(m["pick"] == m["gold"] for m in mc), "n": len(mc), "items": mc}
        arith = []
        for x, y in ADD:
            text = tok.decode(
                greedy(model, rows(tok, f"Addition practice: 3 + 4 = 7, 10 + 5 = 15, {x} + {y} ="), 6),
                skip_special_tokens=False,
            )
            tokens = text.strip().split()
            got = tokens[0].rstrip(",.") if tokens else ""
            arith.append({"q": f"{x}+{y}", "got": got, "ok": got == str(x + y)})
        c["addition"] = {"correct": sum(r["ok"] for r in arith), "n": len(arith), "items": arith}
        py = []
        for prompt, fname, tests in PY_TASKS:
            comp = tok.decode(greedy(model, rows(tok, prompt), 128), skip_special_tokens=False).split("<|endoftext|>")[
                0
            ]
            src = function_body(prompt, comp)
            try:
                ast.parse(src)
                parses = True
            except SyntaxError:
                parses = False
            passed, err = run_python(src, tests) if parses else (False, "syntax")
            py.append({"task": fname, "parses": parses, "passed": passed, "error": "" if passed else err, "body": src})
        c["python"] = {
            "passed": sum(r["passed"] for r in py),
            "parses": sum(r["parses"] for r in py),
            "n": len(py),
            "items": py,
        }
        js = []
        for prompt in JSON_PROMPTS:
            comp = tok.decode(greedy(model, rows(tok, prompt), 64), skip_special_tokens=False)
            js.append({"prompt": prompt, "completion": comp, "valid": json_valid_prefix(prompt, comp)})
        c["json"] = {"valid": sum(r["valid"] for r in js), "n": len(js), "items": js}
        report["candidates"].append(c)
        print(
            name,
            {k: c[k] for k in ("cpu_load_s", "decode_tok_per_s_cpu")},
            "mc",
            c["mc"]["correct"],
            "add",
            c["addition"]["correct"],
            "py",
            c["python"]["passed"],
            "json",
            c["json"]["valid"],
            flush=True,
        )
        a.output.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        del model


if __name__ == "__main__":
    main()
