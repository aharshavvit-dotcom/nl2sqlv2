from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.pipeline_runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full generic NL-to-SQL training pipeline.")
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline_configs" / "smoke_training.yaml")
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--stop-after", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = PipelineRunner().run(str(args.config), start_at=args.start_at, stop_after=args.stop_after)
    print(json.dumps({"pipeline_name": report["pipeline_name"], "status": report["status"]}, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
