from __future__ import annotations

import json
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from datasets import Dataset, DatasetDict


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing dataset file paths and field names."""

    data_paths: Tuple[Path, ...]
    input_field: str = "input_text"
    target_field: str = "target_text"
    validation_ratio: float = 0.1
    seed: int = 42


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _validate_fields(records: Iterable[dict], required_fields: Sequence[str]) -> None:
    missing = set()
    for row in records:
        for field in required_fields:
            if field not in row or not row[field]:
                missing.add(field)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Dataset rows are missing required fields: {joined}")


def _expand_data_paths(raw_paths: Sequence[Path]) -> List[Path]:
    expanded: List[Path] = []
    for path in raw_paths:
        string_path = str(path)
        if any(ch in string_path for ch in "*?[]"):
            matches = [Path(match) for match in glob(string_path, recursive=True)]
            expanded.extend(matches)
            continue

        if path.is_dir():
            matches = sorted(path.rglob("*.jsonl"))
            expanded.extend(matches)
        elif path.exists():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"Dataset path not found: {path}")

    unique_files = []
    seen = set()
    for file_path in sorted(expanded):
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def load_local_dataset(config: DatasetConfig) -> DatasetDict:
    """Load a question generation dataset from local JSONL files."""

    data_files = _expand_data_paths(config.data_paths)
    if not data_files:
        raise FileNotFoundError("No dataset files found for the provided paths.")

    records: List[dict] = []
    for file_path in data_files:
        records.extend(_read_jsonl(file_path))
    _validate_fields(records, [config.input_field, config.target_field])

    rng = random.Random(config.seed)
    rng.shuffle(records)

    val_size = max(1, int(len(records) * config.validation_ratio)) if len(records) > 1 else 0
    train_records = records[val_size:] if val_size else records
    validation_records = records[:val_size] if val_size else records

    train_dataset = Dataset.from_list(train_records)

    if val_size:
        validation_dataset = Dataset.from_list(validation_records)
    else:
        validation_dataset = Dataset.from_list(records)

    return DatasetDict(train=train_dataset, validation=validation_dataset)
