from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_runtime import main


if __name__ == "__main__":
    print("scripts/evaluate.py delegates to scripts/evaluate_runtime.py.", file=sys.stderr)
    raise SystemExit(main())
