from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class IterationReporter:
    def write(self, output_dir: str | Path, report: dict[str, Any], title: str = "Self-Improvement Iteration") -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [f"# {title}", ""]
        for key, value in (report.get("summary") or report).items():
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"- **{key}**: {value}")
        (path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
