"""Contract validator for pipeline step inputs and outputs.

Enforces fail-fast: if a required input is missing before a step,
or a required output is missing after a step, the pipeline stops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .step_contract import StepContract


class ContractValidator:
    """Validates pipeline step contracts against the filesystem."""

    def validate_inputs(self, contract: StepContract, base_dir: str | Path = ".") -> dict[str, Any]:
        """Check that all declared inputs exist before running a step.

        Returns:
            dict with keys: valid (bool), missing (list[str]), checked (list[str])
        """
        return self._check_paths(contract.inputs, base_dir)

    def validate_outputs(self, contract: StepContract, base_dir: str | Path = ".") -> dict[str, Any]:
        """Check that all declared outputs exist after running a step.

        Returns:
            dict with keys: valid (bool), missing (list[str]), checked (list[str])
        """
        return self._check_paths(contract.outputs, base_dir)

    @staticmethod
    def _check_paths(paths: list[str], base_dir: str | Path) -> dict[str, Any]:
        base = Path(base_dir)
        missing: list[str] = []
        checked: list[str] = []
        for path_str in paths:
            resolved = Path(path_str) if Path(path_str).is_absolute() else base / path_str
            checked.append(str(resolved))
            if not resolved.exists():
                missing.append(str(resolved))
        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "checked": checked,
        }
