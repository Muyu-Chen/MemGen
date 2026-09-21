"""Shared, score-free utilities for HardBench generation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HARD_BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HARD_BENCH_ROOT.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
BASE_MODEL_PATH = WORKSPACE_ROOT / "models" / "Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = (
    WORKSPACE_ROOT
    / "models"
    / "memgen-checkpoints"
    / "Qwen2.5-1.5B-Instruct"
    / "gsm8k"
    / "weaver-sft"
    / "pn=1_pl=8_in=3_il=8"
    / "model"
)
DEFAULT_MANIFEST = (
    HARD_BENCH_ROOT / "manifests" / "gsm8k_harder_min6_seed20260920.jsonl"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def portable_path(path: Path) -> str:
    """Serialize a path relative to the repository, including sibling model assets."""
    return Path(os.path.relpath(path.resolve(), REPO_ROOT)).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Durably append one raw result before any scoring is attempted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta_path_for(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".meta.json")


def load_manifest(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise FileNotFoundError(f"Manifest is empty or missing: {path}")
    return records[:max_samples] if max_samples is not None else records


def build_messages(
    sample: dict[str, Any], condition: str, manifest_meta: dict[str, Any]
) -> list[dict[str, str]]:
    target = [dict(message) for message in sample["official_prompt_messages"]]
    if condition in {
        "A_base_zero_shot",
        "D_memgen_official",
        "R_memgen_random50_inference",
        "N_memgen_no_inference",
        "H_memgen_human_single",
    }:
        return target
    if condition == "B_base_3shot":
        prefix = [dict(message) for message in manifest_meta["few_shot_messages"]]
        return prefix + target
    raise ValueError(f"Unknown condition: {condition}")


def completed_sample_ids(path: Path, condition: str) -> set[str]:
    return {
        record["sample_id"]
        for record in read_jsonl(path)
        if record.get("condition") == condition
    }


def resolve_torch_dtype(name: str):
    import torch

    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(mapping)}") from exc


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer
