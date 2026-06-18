from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import write_json


def save_report_pair(path: str | Path, report: dict[str, Any], title: str) -> None:
    target = Path(path)
    write_json(target, report)
    target.with_suffix(".md").write_text(report_to_markdown(report, title), encoding="utf-8")


def report_to_markdown(report: dict[str, Any], title: str) -> str:
    lines = [f"# {title}", ""]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else report
    lines.append("## Summary")
    for key, value in list(summary.items())[:20]:
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- **{key}**: {value}")
    for section in [
        "split_counts",
        "dataset_distribution",
        "intent_distribution",
        "by_dataset",
        "by_intent",
        "failure_categories",
        "thresholds",
    ]:
        value = report.get(section)
        if isinstance(value, dict) and value:
            lines.extend(["", f"## {section.replace('_', ' ').title()}"])
            for item_key, item_value in value.items():
                lines.append(f"- {item_key}: {item_value}")
    recommendations = report.get("recommended_next_actions") or report.get("recommendations") or []
    if recommendations:
        lines.extend(["", "## Recommended Next Actions"])
        lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines) + "\n"
