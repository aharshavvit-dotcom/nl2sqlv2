from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator
from inference.synonym_loader import load_metric_dimension_maps
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import SchemaGraph


ROOT = Path(__file__).resolve().parents[1]

# New canonical artifact dirs (with fallback to old names)
def _resolve_dir(new_name: str, old_name: str) -> Path:
    new_path = ROOT / "artifacts" / new_name
    return new_path if new_path.exists() else ROOT / "artifacts" / old_name

DEFAULT_ARTIFACT_DIR = _resolve_dir("retrieval_ir_model", "option_c_model")
DEFAULT_SAMPLE_MODEL = ROOT / "models" / "tfidf_retriever.joblib"
DEFAULT_SAMPLE_EXAMPLES = ROOT / "training_data" / "examples.jsonl"
DEFAULT_TEMPLATES = ROOT / "data" / "templates.yaml"
DEFAULT_SYNONYMS = ROOT / "data" / "synonyms.yaml"
DEFAULT_NEURAL_IR_ARTIFACT_DIR = _resolve_dir("neural_ir_model", "option_a_ir_model")
DEFAULT_NEURAL_IR_V2_ARTIFACT_DIR = _resolve_dir("neural_ir_model", "option_a_ir_model_v2")


@dataclass
class RetrievalNL2SQLModel:
    retriever: TfidfRetriever
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
        # Backward-compatible aliases
        option_a_model_dir: str | Path | None = None,
        use_option_a_fallback: bool | None = None,
    ) -> "RetrievalNL2SQLModel":
        artifact_path = Path(artifact_dir)
        templates = Path(templates_path)
        synonyms = Path(synonyms_path)

        # Accept old param names
        _model_dir = neural_ir_model_dir or option_a_model_dir
        _fallback = use_neural_ir_fallback if use_option_a_fallback is None else use_option_a_fallback

        neural_ir_path = Path(_model_dir) if _model_dir is not None else cls._default_neural_ir_dir()
        orchestrator = PredictionOrchestrator(
            neural_ir_model_dir=neural_ir_path if _model_dir is not None else None,
            use_neural_ir_fallback=_fallback,
        )
        metric_synonyms, dimension_synonyms = cls._load_synonyms(synonyms)
        if cls.artifact_ready(artifact_path):
            return cls(
                retriever=TfidfRetriever.load(artifact_path),
                templates_path=templates,
                synonyms_path=synonyms,
                artifact_dir=artifact_path,
                metadata=cls._load_metadata(artifact_path),
                orchestrator=orchestrator,
                metric_synonyms=metric_synonyms,
                dimension_synonyms=dimension_synonyms,
                neural_ir_model_dir=neural_ir_path,
                use_neural_ir_fallback=_fallback,
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
    def _default_neural_ir_dir() -> Path:
        return DEFAULT_NEURAL_IR_V2_ARTIFACT_DIR if (DEFAULT_NEURAL_IR_V2_ARTIFACT_DIR / "model.pt").exists() else DEFAULT_NEURAL_IR_ARTIFACT_DIR

    def predict(self, question: str, schema: SchemaGraph, use_neural_ir_fallback: bool | None = None, use_option_a_fallback: bool | None = None) -> PredictionResult:
        _fallback = use_neural_ir_fallback if use_neural_ir_fallback is not None else use_option_a_fallback
        return self.orchestrator.predict(
            question=question,
            schema=schema,
            retriever=self.retriever,
            templates=None,
            metric_synonyms=self.metric_synonyms,
            dimension_synonyms=self.dimension_synonyms,
            validator=None,
            use_neural_ir_fallback=self.use_neural_ir_fallback if _fallback is None else _fallback,
        )

    @staticmethod
    def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for name in [
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
