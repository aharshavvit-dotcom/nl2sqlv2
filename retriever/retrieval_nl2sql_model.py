from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import SchemaGraph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "option_c_model"
DEFAULT_SAMPLE_MODEL = ROOT / "models" / "tfidf_retriever.joblib"
DEFAULT_SAMPLE_EXAMPLES = ROOT / "training_data" / "examples.jsonl"
DEFAULT_TEMPLATES = ROOT / "data" / "templates.yaml"
DEFAULT_SYNONYMS = ROOT / "data" / "synonyms.yaml"


@dataclass
class RetrievalNL2SQLModel:
    retriever: TfidfRetriever
    templates_path: Path = DEFAULT_TEMPLATES
    synonyms_path: Path = DEFAULT_SYNONYMS
    artifact_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    orchestrator: PredictionOrchestrator = field(default_factory=PredictionOrchestrator)

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        sample_model_path: str | Path = DEFAULT_SAMPLE_MODEL,
        sample_examples_path: str | Path = DEFAULT_SAMPLE_EXAMPLES,
        templates_path: str | Path = DEFAULT_TEMPLATES,
        synonyms_path: str | Path = DEFAULT_SYNONYMS,
    ) -> "RetrievalNL2SQLModel":
        artifact_path = Path(artifact_dir)
        templates = Path(templates_path)
        synonyms = Path(synonyms_path)
        if cls.artifact_ready(artifact_path):
            return cls(
                retriever=TfidfRetriever.load(artifact_path),
                templates_path=templates,
                synonyms_path=synonyms,
                artifact_dir=artifact_path,
                metadata=cls._load_metadata(artifact_path),
            )
        return cls(
            retriever=TfidfRetriever.load_or_train(sample_model_path, sample_examples_path),
            templates_path=templates,
            synonyms_path=synonyms,
            artifact_dir=None,
            metadata={},
        )

    @staticmethod
    def artifact_ready(artifact_dir: str | Path) -> bool:
        path = Path(artifact_dir)
        return (
            (path / "training_examples.jsonl").exists()
            and (path / "tfidf_vectorizer.pkl").exists()
            and (path / "tfidf_matrix.pkl").exists()
        )

    def predict(self, question: str, schema: SchemaGraph) -> PredictionResult:
        return self.orchestrator.predict(
            question=question,
            schema=schema,
            retriever=self.retriever,
            templates=None,
            metric_synonyms=None,
            dimension_synonyms=None,
            validator=None,
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
