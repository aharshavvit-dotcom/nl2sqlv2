from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator
from inference.synonym_loader import load_metric_dimension_maps
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import SchemaGraph


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "work" / "retrieval_ir"
DEFAULT_SAMPLE_MODEL = ROOT / "models" / "tfidf_retriever.joblib"
DEFAULT_SAMPLE_EXAMPLES = ROOT / "training_data" / "examples.jsonl"
DEFAULT_TEMPLATES = ROOT / "data" / "templates.yaml"
DEFAULT_SYNONYMS = ROOT / "data" / "synonyms.yaml"
DEFAULT_NEURAL_IR_ARTIFACT_DIR = ROOT / "artifacts" / "work" / "neural_ir"
DEFAULT_NEURAL_IR_V2_ARTIFACT_DIR = ROOT / "artifacts" / "work" / "neural_ir"


@dataclass
class RetrievalNL2SQLModel:
    retriever: Any
    templates_path: Path = DEFAULT_TEMPLATES
    synonyms_path: Path = DEFAULT_SYNONYMS
    artifact_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    orchestrator: PredictionOrchestrator = field(default_factory=PredictionOrchestrator)
    metric_synonyms: dict[str, list[str]] = field(default_factory=dict)
    dimension_synonyms: dict[str, list[str]] = field(default_factory=dict)
    neural_ir_model_dir: Path = DEFAULT_NEURAL_IR_ARTIFACT_DIR
    use_neural_ir_fallback: bool = False

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        sample_model_path: str | Path = DEFAULT_SAMPLE_MODEL,
        sample_examples_path: str | Path = DEFAULT_SAMPLE_EXAMPLES,
        templates_path: str | Path = DEFAULT_TEMPLATES,
        synonyms_path: str | Path = DEFAULT_SYNONYMS,
        neural_ir_model_dir: str | Path | None = None,
        use_neural_ir_fallback: bool = False,
        allow_dev_fallback: bool | None = None,
        confidence_calibration_path: str | Path | None = None,
        # Backward-compatible aliases
        option_a_model_dir: str | Path | None = None,
        use_option_a_fallback: bool | None = None,
    ) -> "RetrievalNL2SQLModel":
        artifact_path = Path(artifact_dir)
        templates = Path(templates_path)
        synonyms = Path(synonyms_path)
        bundle_context = cls._bundle_context(artifact_path)
        bundle_manifest = bundle_context.get("manifest") or {}
        if bundle_context:
            artifact_path = Path(bundle_context["retrieval_model_dir"])
            if neural_ir_model_dir is None and bundle_context.get("neural_model_dir"):
                neural_ir_model_dir = Path(bundle_context["neural_model_dir"])
            if confidence_calibration_path is None and bundle_context.get("calibration_path"):
                confidence_calibration_path = Path(bundle_context["calibration_path"])

        # Accept old param names
        _model_dir = neural_ir_model_dir or option_a_model_dir
        _fallback = use_neural_ir_fallback if use_option_a_fallback is None else use_option_a_fallback

        neural_ir_path = Path(_model_dir) if _model_dir is not None else cls._default_neural_ir_dir()
        orchestrator = PredictionOrchestrator(
            neural_ir_model_dir=neural_ir_path if _model_dir is not None else None,
            use_neural_ir_fallback=_fallback,
            confidence_calibration_path=confidence_calibration_path,
            schema_drift_baseline=bundle_manifest.get("schema_drift_baseline") or {},
        )
        metric_synonyms, dimension_synonyms = cls._load_synonyms(synonyms)
        if cls.rag_artifact_ready(artifact_path):
            from retrieval.rag_retriever import RAGRetrieverAdapter

            metadata = cls._load_metadata(artifact_path)
            metadata["retrieval_backend"] = "local_rag"
            if bundle_context:
                metadata["model_bundle"] = bundle_manifest
                metadata["calibration_loaded"] = bool(confidence_calibration_path and Path(confidence_calibration_path).exists())
            return cls(
                retriever=RAGRetrieverAdapter.load(artifact_path),
                templates_path=templates,
                synonyms_path=synonyms,
                artifact_dir=artifact_path,
                metadata=metadata,
                orchestrator=orchestrator,
                metric_synonyms=metric_synonyms,
                dimension_synonyms=dimension_synonyms,
                neural_ir_model_dir=neural_ir_path,
                use_neural_ir_fallback=_fallback,
            )
        if cls.artifact_ready(artifact_path):
            metadata = cls._load_metadata(artifact_path)
            if bundle_context:
                metadata["model_bundle"] = bundle_manifest
                metadata["calibration_loaded"] = bool(confidence_calibration_path and Path(confidence_calibration_path).exists())
            return cls(
                retriever=TfidfRetriever.load(artifact_path),
                templates_path=templates,
                synonyms_path=synonyms,
                artifact_dir=artifact_path,
                metadata=metadata,
                orchestrator=orchestrator,
                metric_synonyms=metric_synonyms,
                dimension_synonyms=dimension_synonyms,
                neural_ir_model_dir=neural_ir_path,
                use_neural_ir_fallback=_fallback,
            )
        dev_fallback = cls._dev_fallbacks_enabled() if allow_dev_fallback is None else allow_dev_fallback
        if not dev_fallback:
            raise RuntimeError(
                "No validated model bundle found. Run python training/train_model.py --config configs/training.yaml"
            )
        return cls(
            retriever=TfidfRetriever.load_or_train(sample_model_path, sample_examples_path),
            templates_path=templates,
            synonyms_path=synonyms,
            artifact_dir=None,
            metadata={},
            orchestrator=orchestrator,
            metric_synonyms=metric_synonyms,
            dimension_synonyms=dimension_synonyms,
            neural_ir_model_dir=neural_ir_path,
            use_neural_ir_fallback=_fallback,
        )

    @staticmethod
    def artifact_ready(artifact_dir: str | Path) -> bool:
        path = Path(artifact_dir)
        return (
            (path / "training_examples.jsonl").exists()
            and (path / "tfidf_vectorizer.pkl").exists()
            and (path / "tfidf_matrix.pkl").exists()
        )

    @staticmethod
    def rag_artifact_ready(artifact_dir: str | Path) -> bool:
        path = Path(artifact_dir)
        return (
            (path / "example_index.pkl").exists()
            and (path / "schema_index.pkl").exists()
            and (path / "pattern_index.pkl").exists()
            and (path / "rag_metadata.json").exists()
            and (path / "manifest.json").exists()
        )

    @staticmethod
    def _default_neural_ir_dir() -> Path:
        return DEFAULT_NEURAL_IR_V2_ARTIFACT_DIR if (DEFAULT_NEURAL_IR_V2_ARTIFACT_DIR / "model.pt").exists() else DEFAULT_NEURAL_IR_ARTIFACT_DIR

    @staticmethod
    def _bundle_context(path: Path) -> dict[str, Any]:
        bundle_dir: Path | None = None
        if (path / "bundle_manifest.json").exists():
            bundle_dir = path
        elif (path.parent / "bundle_manifest.json").exists():
            bundle_dir = path.parent
        if bundle_dir is None:
            return {}
        manifest_path = bundle_dir / "bundle_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        paths = manifest.get("paths") or {}
        evaluation_dir = bundle_dir / paths.get("evaluation", "evaluation/")
        calibration_path = evaluation_dir / "calibration_report.json"
        return {
            "bundle_dir": str(bundle_dir),
            "manifest": manifest,
            "retrieval_model_dir": str(bundle_dir / paths.get("retrieval_ir", "retrieval_ir/")),
            "neural_model_dir": str(bundle_dir / paths.get("neural_ir", "neural_ir/")),
            "evaluation_dir": str(evaluation_dir),
            "calibration_path": str(calibration_path) if calibration_path.exists() else None,
        }

    @staticmethod
    def _dev_fallbacks_enabled() -> bool:
        return (
            os.getenv("APP_MODE", "").strip().lower() == "demo"
            or os.getenv("ENABLE_DEV_FALLBACKS", "").strip().lower() in {"1", "true", "yes"}
        )

    def predict(self, question: str, schema: SchemaGraph, use_neural_ir_fallback: bool | None = None, use_option_a_fallback: bool | None = None) -> PredictionResult:
        _fallback = use_neural_ir_fallback if use_neural_ir_fallback is not None else use_option_a_fallback
        result = self.orchestrator.predict(
            question=question,
            schema=schema,
            retriever=self.retriever,
            templates=None,
            metric_synonyms=self.metric_synonyms,
            dimension_synonyms=self.dimension_synonyms,
            validator=None,
            use_neural_ir_fallback=self.use_neural_ir_fallback if _fallback is None else _fallback,
        )
        result.debug["dev_fallback_used"] = self.artifact_dir is None
        # Precise runtime_source: distinguish validated bundles from raw artifact dirs
        bundle_meta = self.metadata.get("model_bundle") or {}
        if self.artifact_dir is None:
            result.debug["runtime_source"] = "dev_fallback"
        elif bundle_meta:
            bundle_status = bundle_meta.get("status", "candidate")
            result.debug["runtime_source"] = f"model_bundle_{bundle_status}"
        else:
            result.debug["runtime_source"] = "artifact_dirs"
        # Additional runtime provenance for the Streamlit UI
        result.debug["calibration_loaded"] = bool(self.orchestrator.confidence_calibration)
        result.debug["schema_drift_baseline_loaded"] = bool(self.orchestrator.schema_drift_baseline)
        result.debug["bundle_id"] = bundle_meta.get("bundle_id", "")
        result.debug["bundle_dir"] = str(self.artifact_dir) if self.artifact_dir else ""
        result.debug["bundle_status"] = bundle_meta.get("status", "")
        result.debug["artifact_dir"] = str(self.artifact_dir) if self.artifact_dir else ""
        return result

    @staticmethod
    def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for name in [
            "rag_metadata.json",
            "supported_patterns.json",
            "dataset_stats.json",
            "training_report.json",
            "evaluation_report.json",
        ]:
            path = artifact_dir / name
            if path.exists():
                metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        return metadata

    @staticmethod
    def _load_synonyms(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        return load_metric_dimension_maps(path)
