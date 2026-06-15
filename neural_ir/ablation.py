from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class OptionAAblationEvaluator:
    def run(self, test_path: str, db_path: str | None = None, output_path: str | None = None) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        variants = [
            ("v1_model", root / "artifacts" / "option_a_ir_model"),
            ("v1_5_model", root / "artifacts" / "option_a_ir_model_v1_5"),
            ("v2_without_hard_negatives", root / "artifacts" / "option_a_ir_model_v2_no_hn"),
            ("v2_with_hard_negatives", root / "artifacts" / "option_a_ir_model_v2"),
            ("v2_with_repair", root / "artifacts" / "option_a_ir_model_v2"),
            ("v2_with_repair_and_calibration", root / "artifacts" / "option_a_ir_model_v2"),
            ("hybrid_with_v2", root / "artifacts" / "option_a_ir_model_v2"),
        ]
        results = []
        for name, path in variants:
            if not (path / "model.pt").exists():
                continue
            results.append({"variant": name, "artifact_dir": str(path), "available": True})
        report = {
            "test_path": test_path,
            "db_path": db_path,
            "variants": results,
            "summary": {"available_variants": len(results)},
            "recommendations": [] if results else ["train Option A artifacts before running full ablation"],
        }
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            import json

            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
