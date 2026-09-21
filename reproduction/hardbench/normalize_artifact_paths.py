"""Rewrite absolute path metadata in selected JSON/JSONL artifacts as repo-relative paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import portable_path, write_json, write_jsonl


PATH_KEYS = {
    "base_model",
    "checkpoint",
    "checkpoint_path",
    "manifest",
    "model_path",
    "path",
    "plan_file",
    "policy_file",
    "raw_path",
    "scored_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def normalize(value: Any, key: str | None = None) -> tuple[Any, int]:
    if isinstance(value, dict):
        result = {}
        changed = 0
        for child_key, child_value in value.items():
            result[child_key], child_changed = normalize(child_value, child_key)
            changed += child_changed
        return result, changed
    if isinstance(value, list):
        result = []
        changed = 0
        for child_value in value:
            normalized, child_changed = normalize(child_value)
            result.append(normalized)
            changed += child_changed
        return result, changed
    if key in PATH_KEYS and isinstance(value, str) and Path(value).is_absolute():
        return portable_path(Path(value)), 1
    return value, 0


def candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix in {".json", ".jsonl"}
    )


def main() -> None:
    args = parse_args()
    files = []
    for path in args.paths:
        files.extend(candidate_files(path.resolve()))
    changes = 0
    changed_files = 0
    for path in sorted(set(files)):
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            normalized, count = normalize(value)
            if count:
                write_json(path, normalized)
        else:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            normalized, count = normalize(records)
            if count:
                write_jsonl(path, normalized)
        if count:
            changed_files += 1
            changes += count
    print(f"files={len(set(files))} changed_files={changed_files} paths_rewritten={changes}")


if __name__ == "__main__":
    main()
