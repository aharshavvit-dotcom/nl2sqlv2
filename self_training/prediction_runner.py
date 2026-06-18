"""Prediction Runner — batch predictions using the neural IR model.

Runs the NeuralIRPredictor on a list of dataset examples and returns
structured prediction results suitable for comparison against gold labels.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class PredictionRunner:
    """Runs batch predictions using the neural IR model."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._predictor: Any = None

    def _load_predictor(self) -> Any:
        """Lazy-load the NeuralIRPredictor to avoid import cost at init."""
        if self._predictor is None:
            from neural_ir.predictor import NeuralIRPredictor
            self._predictor = NeuralIRPredictor(str(self.model_dir))
        return self._predictor

    def predict_batch(
        self,
        examples: list[dict[str, Any]],
        max_examples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run predictions for a list of examples.

        Parameters
        ----------
        examples:
            Dataset rows.  Each must have at least ``question`` and
            ``schema`` (or ``serialized_schema``).
        max_examples:
            If set, only predict the first *N* examples.

        Returns
        -------
        list of dicts, each containing:
          - ``example_id``, ``question``, ``dataset_name``, ``db_id``
          - ``predicted_query_ir``, ``predicted_sql``, ``confidence``
          - ``gold_query_ir``, ``gold_sql``
          - ``prediction_failed`` (bool), ``error_message`` (str | None)
          - ``prediction_time_ms`` (float)
        """

        predictor = self._load_predictor()
        subset = examples[:max_examples] if max_examples else examples
        results: list[dict[str, Any]] = []

        total = len(subset)
        for idx, example in enumerate(subset, start=1):
            if idx % max(1, total // 10) == 0 or idx == total:
                print(f"  [Predict] {idx}/{total} examples processed")

            result = self._predict_single(predictor, example)
            results.append(result)

        return results

    @staticmethod
    def _predict_single(
        predictor: Any,
        example: dict[str, Any],
    ) -> dict[str, Any]:
        """Run prediction for a single example with error handling."""

        question = example.get("question", "")
        schema = example.get("schema") or _schema_from_serialized(example.get("serialized_schema"))
        example_id = str(example.get("example_id", ""))
        gold_ir = example.get("query_ir") or example.get("gold_query_ir") or {}
        gold_sql = example.get("source_sql") or example.get("gold_sql") or ""

        start = time.perf_counter()
        try:
            prediction = predictor.predict(question, schema or {})
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            return {
                "example_id": example_id,
                "question": question,
                "dataset_name": example.get("dataset_name", ""),
                "db_id": example.get("db_id", ""),
                "split": example.get("split", ""),
                "predicted_query_ir": prediction.get("query_ir") or prediction.get("repaired_query_ir"),
                "predicted_sql": prediction.get("sql"),
                "confidence": float(prediction.get("confidence", 0.0)),
                "raw_confidence": float(prediction.get("raw_confidence", 0.0)),
                "ir_validation": prediction.get("ir_validation"),
                "sql_validation": prediction.get("sql_validation") or prediction.get("validation"),
                "gold_query_ir": gold_ir,
                "gold_sql": gold_sql,
                "prediction_failed": False,
                "error_message": None,
                "prediction_time_ms": round(elapsed_ms, 2),
                "warnings": prediction.get("warnings", []),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "example_id": example_id,
                "question": question,
                "dataset_name": example.get("dataset_name", ""),
                "db_id": example.get("db_id", ""),
                "split": example.get("split", ""),
                "predicted_query_ir": None,
                "predicted_sql": None,
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "ir_validation": {"is_valid": False, "errors": [str(exc)]},
                "sql_validation": {"is_valid": False, "issues": [str(exc)]},
                "gold_query_ir": gold_ir,
                "gold_sql": gold_sql,
                "prediction_failed": True,
                "error_message": str(exc),
                "prediction_time_ms": round(elapsed_ms, 2),
                "warnings": [str(exc)],
            }


def _schema_from_serialized(serialized: str | None) -> dict[str, Any] | None:
    """Best-effort parse of a serialized schema string into a dict."""
    if not serialized:
        return None
    # The serialized format is typically a text representation; we wrap it
    # so the predictor has something to work with.
    return {"serialized_schema": serialized, "tables": {}}
