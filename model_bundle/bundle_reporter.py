"""Bundle reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BundleReporter:
    """Generates human-readable and machine-readable bundle reports."""

    def write_validation_report(self, output_dir: str | Path, validation: dict[str, Any]) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "bundle_validation_report.json").write_text(
            json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        lines = [
            "# Bundle Validation Report",
            "",
            f"**Passed:** {'✅ Yes' if validation.get('passed') else '❌ No'}",
            "",
        ]
        if validation.get("blocking_issues"):
            lines.append("## Blocking Issues")
            for issue in validation["blocking_issues"]:
                lines.append(f"- ❌ {issue}")
            lines.append("")
        if validation.get("warnings"):
            lines.append("## Warnings")
            for warning in validation["warnings"]:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")
        if validation.get("checked_files"):
            lines.append("## Checked Files")
            for checked in validation["checked_files"]:
                lines.append(f"- {checked}")
            lines.append("")
        (path / "bundle_validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    def write_promotion_report(self, output_dir: str | Path, promotion: dict[str, Any]) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "bundle_promotion_report.json").write_text(
            json.dumps(promotion, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        promoted = promotion.get("promoted", False)
        lines = [
            "# Bundle Promotion Report",
            "",
            f"**Promoted:** {'✅ Yes' if promoted else '❌ No'}",
            f"**Bundle ID:** {promotion.get('bundle_id', 'unknown')}",
        ]
        if promoted:
            lines.append(f"**Promoted at:** {promotion.get('promoted_at', 'unknown')}")
        else:
            lines.append(f"**Reason:** {promotion.get('reason', 'unknown')}")
        lines.append("")
        (path / "bundle_promotion_report.md").write_text("\n".join(lines), encoding="utf-8")
