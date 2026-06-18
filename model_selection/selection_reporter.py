from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SelectionReporter:
    def write(self, path: str | Path, report: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md = target.with_suffix(".md")
        lines = ["# Model Selection Report", "", f"Selected: {report.get('selected_model')}", ""]
        if report.get("blocking_issues"):
            lines.append("## Blocking Issues")
            lines.extend(f"- {item}" for item in report["blocking_issues"])
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
