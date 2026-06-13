from __future__ import annotations

import sys

from scripts.evaluate_runtime import main


if __name__ == "__main__":
    print("scripts/evaluate.py is a legacy compatibility entrypoint; delegating to scripts/evaluate_runtime.py.", file=sys.stderr)
    raise SystemExit(main())
