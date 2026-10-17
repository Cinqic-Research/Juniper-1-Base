"""Admitted SFT dataset loading, verification and deterministic batching.

The dataset is produced by Juniper-1-Data (``j1data.sftbuild``, schema ``j1.sft-dataset.v1``).
Each row holds one whole episode as next-token pairs made by ``j1data.sft.causal_rows``:
``inputs = ids[:-1]`` and ``targets = labels[1:]`` with ``-100`` on every host-authored or
otherwise untrained position. This loader never re-derives labels; it verifies them:

* the manifest self-hash and the caller's pinned manifest digest, every payload file hash,
  the tokenizer identity and the stage;
* for every episode, each supervised target equals the *next* input token (and the final
  target, the stop token, is supervised). A dataset carrying unshifted, token-aligned labels
  (which would teach the model to copy its input) fails this check;
* token IDs lie inside the vocabulary.

Batching is a pure function of ``(dataset, seed, epoch, update index, episodes per update,
token budget)``. Episodes are never packed: each micro-batch row holds one episode, right-padded
with token 0 and target -100, so no episode can attend to another (the model is causal and pads
come after real tokens).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IGNORE = -100
STOP_IDS = (10, 11)  # <|call|>, <|eot|> in juniper-chat-v1
DATASET_SCHEMA = "j1.sft-dataset.v1"
EVAL_SCHEMA = "j1.sft-eval-bundle.v1"


class DatasetError(RuntimeError):
    """The dataset is not the pinned, admitted, correctly shifted SFT dataset."""


def canonical_json(value) -> bytes:
    """Identical to ``j1data.hashing.canonical_json`` (sorted keys, compact, UTF-8)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def check_alignment(inputs: np.ndarray, targets: np.ndarray, offsets: np.ndarray) -> None:
    """Every supervised target must be the next input token; the last target must be supervised."""
    if len(inputs) != len(targets) or offsets[0] != 0 or offsets[-1] != len(inputs):
        raise DatasetError("row arrays and offsets disagree")
    lens = np.diff(offsets)
    if (lens < 1).any():
        raise DatasetError("empty episode row")
    nxt = np.empty_like(inputs)
    nxt[:-1] = inputs[1:]
    last = offsets[1:] - 1
    nxt[last] = -1  # no next input inside the episode
    sup = targets != IGNORE
    inner = sup.copy()
    inner[last] = False
    if not np.array_equal(targets[inner], nxt[inner]):
        bad = int(np.nonzero(inner & (targets != nxt))[0][0])
        ep = int(np.searchsorted(offsets, bad, side="right") - 1)
        raise DatasetError(
            f"episode {ep}: target at position {bad - offsets[ep]} is not the next input token "
            "(labels look token-aligned/unshifted or misaligned)"
        )
    if not sup[last].all():
        raise DatasetError("an episode's final target (its stop token) is not supervised")
    if not np.isin(targets[last], STOP_IDS).all():
        raise DatasetError("an episode's final target is not a stop token (<|eot|> or <|call|>)")
    per_ep = np.add.reduceat(sup.astype(np.int64), offsets[:-1])
    if (per_ep < 1).any():
        raise DatasetError("an episode has no supervised target")


@dataclass
class Split:
    inputs: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray
    episode_ids: np.ndarray

    @property
    def n(self) -> int:
        return len(self.offsets) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def episode(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.inputs[a:b], self.targets[a:b]


class SFTDataset:
    def __init__(self, directory: Path, *, expected_manifest_sha256: str, tokenizer_sha256: str, vocab_size: int):
        self.dir = Path(directory)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        if manifest.get("schema") == EVAL_SCHEMA or "bundle_sha256" in manifest:
            raise DatasetError("this is an evaluation bundle; it can never be used as training data")
        if manifest.get("schema") != DATASET_SCHEMA:
            raise DatasetError(f"unknown dataset schema {manifest.get('schema')!r}")
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if hashlib.sha256(canonical_json(body)).hexdigest() != manifest.get("manifest_sha256"):
            raise DatasetError("dataset manifest self-hash mismatch")
        if manifest["manifest_sha256"] != expected_manifest_sha256:
            raise DatasetError("dataset manifest is not the one the stage contract pins")
        if manifest.get("tokenizer_sha256") != tokenizer_sha256:
            raise DatasetError("dataset tokenizer differs from the frozen base tokenizer")
        if manifest.get("stage") != "agent-instruction-sft":
            raise DatasetError(f"dataset stage {manifest.get('stage')!r} is not agent-instruction-sft")
        for name, rec in manifest["files"].items():
            if _sha256_file(self.dir / name) != rec["sha256"]:
                raise DatasetError(f"{name}: bytes differ from the manifest")
        self.manifest = manifest
        self.splits: dict[str, Split] = {}
        for split in ("train", "dev"):
            with np.load(self.dir / f"{split}_rows.npz") as z:
                s = Split(
                    z["inputs"].astype(np.int64),
                    z["targets"].astype(np.int64),
                    z["offsets"].astype(np.int64),
                    z["episode_ids"],
                )
            check_alignment(s.inputs, s.targets, s.offsets)
            if s.inputs.min() < 0 or s.inputs.max() >= vocab_size:
                raise DatasetError(f"{split}: input token outside the vocabulary")
            sup = s.targets[s.targets != IGNORE]
            if sup.min() < 0 or sup.max() >= vocab_size:
                raise DatasetError(f"{split}: target token outside the vocabulary")
            self.splits[split] = s
        ids = {split: {bytes(e) for e in s.episode_ids} for split, s in self.splits.items()}
        if any(len(ids[k]) != self.splits[k].n for k in ids) or ids["train"] & ids["dev"]:
            raise DatasetError("episode IDs are duplicated or shared between train and dev")
        listed = set(manifest["files"])
        for name in ("train_rows.npz", "dev_rows.npz"):
            if name not in listed:
                raise DatasetError(f"{name} is not bound by the manifest")

    def identity(self) -> dict:
        tr = self.splits["train"]
        return {
            "schema": DATASET_SCHEMA,
            "version": self.manifest["version"],
            "manifest_sha256": self.manifest["manifest_sha256"],
            "train_episodes": tr.n,
            "train_target_tokens": int((tr.targets != IGNORE).sum()),
            "eval_bundle_sha256": self.manifest["eval_bundle_sha256"],
        }


def update_plan(n: int, seed: int, epoch: int, episodes_per_update: int) -> np.ndarray:
    """Episode indices of every update in ``epoch`` (shape ``(n // G, G)``); the remainder is skipped."""
    perm = np.random.default_rng([seed, epoch]).permutation(n)
    k = n // episodes_per_update
    return perm[: k * episodes_per_update].reshape(k, episodes_per_update)


def micro_batches(split: Split, episodes: np.ndarray, token_budget: int, pad_id: int = 0) -> list[tuple]:
    """Split one update's episodes into right-padded micro-batches under a token budget.

    Episodes are ordered by (length desc, index) so similar lengths share a micro-batch. Returns
    ``(x, y, episode_indices, target_count)`` tuples as int64 CPU tensors."""
    lens = split.lengths()[episodes]
    order = sorted(range(len(episodes)), key=lambda i: (-int(lens[i]), int(episodes[i])))
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        width = int(lens[order[0]]) if not cur else int(lens[cur[0]])
        if cur and (len(cur) + 1) * width > token_budget:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    batches = []
    for group in out:
        idx = [int(episodes[i]) for i in group]
        width = max(int(lens[i]) for i in group)
        if width > token_budget:
            raise DatasetError(f"episode of {width} tokens exceeds the micro-batch token budget {token_budget}")
        x = np.full((len(idx), width), pad_id, dtype=np.int64)
        y = np.full((len(idx), width), IGNORE, dtype=np.int64)
        for r, e in enumerate(idx):
            xi, yi = split.episode(e)
            x[r, : len(xi)] = xi
            y[r, : len(yi)] = yi
        batches.append((torch.from_numpy(x), torch.from_numpy(y), idx, int((y != IGNORE).sum())))
    return batches


def plan_fingerprint(batches: list[tuple]) -> str:
    h = hashlib.sha256()
    for x, y, idx, _ in batches:
        h.update(np.asarray(idx, dtype=np.int64).tobytes())
        h.update(x.numpy().tobytes())
        h.update(y.numpy().tobytes())
    return h.hexdigest()
