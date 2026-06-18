from __future__ import annotations

from datetime import datetime
from pathlib import Path


def generate_model_version(root: str | Path = "artifacts") -> str:
    date_part = datetime.now().strftime("%Y-%m-%d")
    root_path = Path(root)
    existing = sorted(root_path.glob(f"**/{date_part}_*.json")) if root_path.exists() else []
    next_index = len(existing) + 1
    return f"{date_part}_{next_index:03d}"
