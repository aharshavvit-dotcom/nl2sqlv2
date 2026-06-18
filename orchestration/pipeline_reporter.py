from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PipelineReporter:
    def write(self, output_dir: str | Path, report: dict[str, Any]) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "pipeline_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = ["# Training Pipeline Report", "", f"Pipeline: {report.get('pipeline_name')}", f"Status: {report.get('status')}", ""]
        lines.append("## Steps")
        for step in report.get("steps", []):
            lines.append(f"- {step.get('step')}: {step.get('status')}")
        (path / "pipeline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
