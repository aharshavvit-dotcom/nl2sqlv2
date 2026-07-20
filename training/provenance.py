"""Training provenance recorder.

Records all information needed to reproduce a training run:
- Git commit hash and dirty state
- Config hash
- Dataset file hashes
- Split version
- Random seed
- Python/PyTorch/CUDA versions
- Timestamps
- Command-line arguments

Usage::

    provenance = TrainingProvenance.capture(
        config=config,
        train_path="data/processed/generic_ir_train.jsonl",
        val_path="data/processed/generic_ir_validation.jsonl",
    )
    provenance.save(output_dir / "provenance.json")
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TrainingProvenance:
    """Immutable provenance record for a training run."""

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    # Git state
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Config
    config_hash: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # Data
    train_path: str = ""
    train_hash: str = ""
    train_size_bytes: int = 0
    train_line_count: int = 0
    val_path: str = ""
    val_hash: str = ""
    val_size_bytes: int = 0
    val_line_count: int = 0
    hard_negatives_path: str = ""
    hard_negatives_hash: str = ""
    partial_supervision_path: str = ""
    partial_supervision_hash: str = ""

    # Split and seed
    split_version: str = ""
    seed: int = 42

    # Environment
    python_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""
    cuda_available: bool = False
    hostname: str = ""
    os_info: str = ""
    command_line: str = ""

    # Pipeline
    pipeline_run_id: str = ""

    @classmethod
    def capture(
        cls,
        config: dict[str, Any] | Any = None,
        train_path: str | Path = "",
        val_path: str | Path = "",
        hard_negatives_path: str | Path = "",
        partial_supervision_path: str | Path = "",
        pipeline_run_id: str = "",
    ) -> "TrainingProvenance":
        """Capture current environment and data provenance.

        Parameters
        ----------
        config :
            Training config dict or object with to_dict()/model_dump().
        train_path, val_path, hard_negatives_path, partial_supervision_path :
            Paths to training data files.
        pipeline_run_id :
            Optional pipeline run identifier.
        """
        prov = cls()
        prov.started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        prov.pipeline_run_id = pipeline_run_id

        # Config
        config_dict = _config_to_dict(config)
        prov.config_snapshot = config_dict
        prov.config_hash = _hash_dict(config_dict)
        prov.seed = int(config_dict.get("training", {}).get("seed", 42))
        prov.split_version = str(config_dict.get("data", {}).get("split_version", ""))

        # Git
        prov.git_commit = _git("rev-parse", "HEAD")
        prov.git_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        prov.git_dirty = bool(_git("status", "--porcelain"))

        # Data files
        if train_path:
            tp = Path(train_path)
            prov.train_path = str(tp)
            if tp.exists():
                prov.train_hash = _file_hash(tp)
                prov.train_size_bytes = tp.stat().st_size
                prov.train_line_count = _line_count(tp)

        if val_path:
            vp = Path(val_path)
            prov.val_path = str(vp)
            if vp.exists():
                prov.val_hash = _file_hash(vp)
                prov.val_size_bytes = vp.stat().st_size
                prov.val_line_count = _line_count(vp)

        if hard_negatives_path:
            hp = Path(hard_negatives_path)
            prov.hard_negatives_path = str(hp)
            if hp.exists():
                prov.hard_negatives_hash = _file_hash(hp)

        if partial_supervision_path:
            pp = Path(partial_supervision_path)
            prov.partial_supervision_path = str(pp)
            if pp.exists():
                prov.partial_supervision_hash = _file_hash(pp)

        # Environment
        prov.python_version = sys.version
        prov.hostname = platform.node()
        prov.os_info = f"{platform.system()} {platform.release()}"
        prov.command_line = " ".join(sys.argv)

        try:
            import torch
            prov.pytorch_version = torch.__version__
            prov.cuda_available = torch.cuda.is_available()
            prov.cuda_version = torch.version.cuda or "" if prov.cuda_available else ""
        except ImportError:
            pass

        return prov

    def mark_completed(self) -> None:
        """Set the completion timestamp."""
        self.completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def save(self, path: str | Path) -> None:
        """Save provenance to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return asdict(self)


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert config to a plain dict."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return dict(config) if config else {}


def _hash_dict(d: dict) -> str:
    """Deterministic hash of a dict."""
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hash (first 16 hex chars) of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


def _line_count(path: Path) -> int:
    """Count lines in a file efficiently."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
