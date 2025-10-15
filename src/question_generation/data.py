from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from datasets import Dataset, DatasetDict


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing dataset file paths and field names."""

    data_path: Path
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


def load_local_dataset(config: DatasetConfig) -> DatasetDict:
    """Load a question generation dataset from a local JSONL file."""

    records = _read_jsonl(config.data_path)
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
