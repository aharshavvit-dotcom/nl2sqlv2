"""Tests for sklearn metadata validation and checkpoint identity verification.

Validates Review #15 and Review #16.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pytest
import sklearn

from retrieval.artifact_compatibility import (
    build_sklearn_metadata,
    write_sklearn_metadata,
    validate_sklearn_metadata,
    validate_file_checksums,
)
from training.evaluate_generic_models import evaluate_generic_models


def test_build_and_validate_sklearn_metadata(tmp_path: Path) -> None:
    # 1. Build metadata
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")
    
    pickle_a = tmp_path / "model_a.pkl"
    pickle_a.write_text("pickle content a", encoding="utf-8")
    
    metadata = build_sklearn_metadata(
        artifact_types=["test_retriever"],
        source_path=dataset_file,
        config={"param": "value"},
        artifact_dir=tmp_path,
        pickle_filenames=["model_a.pkl"],
    )
    
    assert metadata["sklearn_version"] == sklearn.__version__
    assert metadata["artifact_type"] == "test_retriever"
    assert "model_a.pkl" in metadata["files"]
    
    # 2. Write metadata
    meta_path = write_sklearn_metadata(tmp_path, metadata)
    assert meta_path.name == "sklearn_artifact_metadata.json"
    
    # 3. Validate metadata (compatible version)
    val_result = validate_sklearn_metadata(tmp_path, mode="runtime")
    assert val_result["compatible"] is True
    
    # 4. Validate checksums (valid)
    chk_result = validate_file_checksums(tmp_path, metadata)
    assert chk_result["valid"] is True
    
    # 5. Validate checksums (mismatched)
    pickle_a.write_text("tampered pickle content", encoding="utf-8")
    chk_result_mismatch = validate_file_checksums(tmp_path, metadata)
    assert chk_result_mismatch["valid"] is False
    assert "model_a.pkl" in chk_result_mismatch["mismatched_files"]


def test_evaluate_generic_models_extracts_checkpoint_identity(tmp_path: Path) -> None:
    # Mock data files
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"example_id": "1", "question": "test?", "query_ir": {}}\n', encoding="utf-8")
    
    unseen_test_file = tmp_path / "unseen.jsonl"
    unseen_test_file.write_text('{"example_id": "2", "question": "unseen?", "query_ir": {}}\n', encoding="utf-8")
    
    neural_dir = tmp_path / "neural_ir"
    neural_dir.mkdir()
    
    # Mock best_model.pt and model.pt
    best_model = neural_dir / "best_model.pt"
    best_model.write_text("best model weights", encoding="utf-8")
    
    model = neural_dir / "model.pt"
    model.write_text("best model weights", encoding="utf-8")  # equivalent
    
    # Mock checkpoint_metadata.json
    import hashlib
    weights_hash = hashlib.sha256(b"best model weights").hexdigest()
    checkpoint_meta = {
        "best_epoch": 12,
        "best_checkpoint_sha256": weights_hash,
    }
    (neural_dir / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_meta), encoding="utf-8")
    
    # Mock retrieval directory
    retrieval_dir = tmp_path / "retrieval_ir"
    retrieval_dir.mkdir()
    (retrieval_dir / "example_index.pkl").write_text("", encoding="utf-8")
    (retrieval_dir / "schema_index.pkl").write_text("", encoding="utf-8")
    (retrieval_dir / "pattern_index.pkl").write_text("", encoding="utf-8")
    (retrieval_dir / "rag_metadata.json").write_text("{}", encoding="utf-8")
    (retrieval_dir / "manifest.json").write_text("{}", encoding="utf-8")
    
    # Write valid sklearn metadata to satisfy validation check at load time
    metadata = build_sklearn_metadata(
        artifact_types=["rag_index"],
        config={},
    )
    write_sklearn_metadata(retrieval_dir, metadata)
    
    args = argparse.Namespace(
        test=test_file,
        unseen_db_test=unseen_test_file,
        retrieval_model_dir=retrieval_dir,
        neural_model_dir=neural_dir,
        output=tmp_path / "evaluation_report.json",
        thresholds=Path("evaluation/model_quality_thresholds.yaml"),
        max_examples=1,
        allow_gold_replay_baseline=True,
    )
    from unittest.mock import patch
    
    with patch("training.evaluate_generic_models._predict_with_retrieval_model") as mock_predict:
        # Mock returns a single row prediction result
        mock_predict.return_value = (
            [{"example_id": "1", "question": "test?", "predicted_query_ir": {}, "sql_validation_passed": True, "simple_query_pass": True}],
            "neural_queryir",
            "artifact_dirs"
        )
        report = evaluate_generic_models(args)
    
    assert "checkpoint" in report
    ckpt = report["checkpoint"]
    assert ckpt["selected_checkpoint_file"] == "best_model.pt"
    assert ckpt["selected_checkpoint_epoch"] == 12
    assert ckpt["runtime_export_equivalent_to_selected_checkpoint"] is True


def test_quality_gate_fails_on_missing_metadata() -> None:
    from quality_gates.model_quality_gate import ModelQualityGate
    
    # 1. Test production gate passes when metadata is valid
    report = {
        "quality_gate_mode": "production",
        "sklearn_info": {
            "retrieval_sklearn_metadata_valid": True,
            "retrieval_checksums_valid": True,
            "calibration_metadata_valid": True,
        },
        "test_performance": {"summary": {"sql_validation_rate": 0.95, "simple_query_pass_rate": 0.96, "query_ir_validity_rate": 0.98, "unsafe_sql_count": 0, "unnecessary_join_rate": 0.01, "wrong_table_rate": 0.02}},
        "summary": {"unsafe_sql_count": 0},
    }
    thresholds = {
        "minimums": {
            "query_ir_validity_rate": 0.90,
            "sql_validation_rate": 0.90,
            "simple_query_pass_rate": 0.80,
            "simple_query_pass_rate_production": 0.95,
            "unsafe_sql_count_max": 0,
            "unnecessary_join_rate_max": 0.05,
            "wrong_table_rate_max": 0.15,
        }
    }
    
    gate = ModelQualityGate()
    result = gate.evaluate(report, thresholds)
    assert result["passed"] is True
    
    # 2. Test production gate fails when retrieval sklearn metadata is invalid
    report_bad_sklearn = dict(report)
    report_bad_sklearn["sklearn_info"] = {
        "retrieval_sklearn_metadata_valid": False,
        "retrieval_checksums_valid": True,
        "calibration_metadata_valid": True,
    }
    result = gate.evaluate(report_bad_sklearn, thresholds)
    assert result["passed"] is False
    assert any(c["metric"] == "retrieval_sklearn_metadata_valid" for c in result["failed_checks"])

    # 3. Test production gate fails when retrieval checksum is invalid
    report_bad_checksum = dict(report)
    report_bad_checksum["sklearn_info"] = {
        "retrieval_sklearn_metadata_valid": True,
        "retrieval_checksums_valid": False,
        "calibration_metadata_valid": True,
    }
    result = gate.evaluate(report_bad_checksum, thresholds)
    assert result["passed"] is False
    assert any(c["metric"] == "retrieval_checksums_valid" for c in result["failed_checks"])

    # 4. Test production gate fails when calibration metadata is invalid
    report_bad_calibration = dict(report)
    report_bad_calibration["sklearn_info"] = {
        "retrieval_sklearn_metadata_valid": True,
        "retrieval_checksums_valid": True,
        "calibration_metadata_valid": False,
    }
    result = gate.evaluate(report_bad_calibration, thresholds)
    assert result["passed"] is False
    assert any(c["metric"] == "calibration_metadata_valid" for c in result["failed_checks"])

