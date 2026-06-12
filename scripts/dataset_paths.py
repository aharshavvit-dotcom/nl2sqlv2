from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPIDER_DIR = RAW_DATA_DIR / "spider"
WIKISQL_DIR = RAW_DATA_DIR / "wikisql"
BIRD_DIR = RAW_DATA_DIR / "bird"
BIRD_MINI_DEV_DIR = BIRD_DIR / "mini_dev"
BIRD_MINI_DEV_HF_DIR = BIRD_DIR / "mini_dev_hf"
BIRD_LEGACY_MINIDEV_DIR = BIRD_DIR / "minidev" / "MINIDEV"
BIRD_FULL_DIR = BIRD_DIR / "full"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "option_c_model"


def ensure_dataset_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPIDER_DIR,
        WIKISQL_DIR,
        BIRD_DIR,
        BIRD_MINI_DEV_DIR,
        BIRD_MINI_DEV_HF_DIR,
        BIRD_FULL_DIR,
        ARTIFACT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_dataset_dir(dataset_name: str) -> Path:
    normalized = dataset_name.strip().lower().replace("_", "-")
    mapping = {
        "spider": SPIDER_DIR,
        "wikisql": WIKISQL_DIR,
        "wiki-sql": WIKISQL_DIR,
        "bird": resolve_bird_mini_dir(),
        "bird-mini": resolve_bird_mini_dir(),
        "bird-mini-dev": resolve_bird_mini_dir(),
        "bird-full": BIRD_FULL_DIR,
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return mapping[normalized]


def expected_files_for_dataset(dataset_name: str) -> list[str]:
    normalized = dataset_name.strip().lower().replace("_", "-")
    if normalized in {"wikisql", "wiki-sql"}:
        return [
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "train.tables.jsonl",
            "dev.tables.jsonl",
            "test.tables.jsonl",
        ]
    if normalized == "spider":
        return [
            "train_spider.json",
            "dev.json",
            "tables.json",
        ]
    if normalized in {"bird", "bird-mini", "bird-mini-dev"}:
        return [
            "mini_dev_sqlite.json",
            "dev_tables.json",
            "dev_databases/",
        ]
    if normalized == "bird-full":
        return []
    raise ValueError(f"Unknown dataset: {dataset_name}")


def parse_dataset_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = value.split(",")
    return [item.strip().lower().replace("_", "-") for item in raw if item.strip()]


def resolve_bird_mini_dir() -> Path:
    if (BIRD_MINI_DEV_DIR / "mini_dev_sqlite.json").exists():
        return BIRD_MINI_DEV_DIR
    if BIRD_LEGACY_MINIDEV_DIR.exists():
        return BIRD_LEGACY_MINIDEV_DIR
    if (BIRD_MINI_DEV_HF_DIR / "dataset_dict.json").exists():
        return BIRD_MINI_DEV_HF_DIR
    return BIRD_MINI_DEV_DIR
