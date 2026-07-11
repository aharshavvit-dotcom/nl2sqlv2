from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.dataset_artifacts import build_capability_artifacts, dump_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build additive capability and partial-supervision artifacts from frozen generic QueryIR splits."
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "generic_training")
    parser.add_argument("--dialect", default="sqlite")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_capability_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        dialect=args.dialect,
    )
    print(dump_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
