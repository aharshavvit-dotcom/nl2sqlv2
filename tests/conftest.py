"""Shared pytest configuration for rationalized execution lanes.

Purpose: Applies default lane markers from test paths so every collected item
has an execution lane even when an older test module lacks explicit marks.
"""

from __future__ import annotations

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        name = item.name.lower()
        marker_names = {marker.name for marker in item.iter_markers()}

        def add(marker: str) -> None:
            if marker not in marker_names:
                item.add_marker(marker)
                marker_names.add(marker)

        if "/legacy/" in path:
            add("legacy")
        elif "/integration/" in path or "integration" in path or "connected_db" in path or "postgres" in path:
            add("integration")
        elif "/e2e/" in path or "smoke" in path or "end_to_end" in path:
            add("e2e")
        elif "/regression/" in path or "regression" in path or "golden" in path:
            add("regression")
        elif "safety" in path or "validation_policy" in path or "telemetry_privacy" in path:
            add("safety")
        elif "contract" in path or "bundle" in path or "policy" in path:
            add("contract")
        else:
            add("unit")

        if "training" in path or "train_" in path:
            add("training")
        if any(token in path or token in name for token in ("database", "postgres", "sqlite", "connected_db")):
            add("database")
        if any(token in path or token in name for token in ("slow", "full_training")):
            add("slow")
