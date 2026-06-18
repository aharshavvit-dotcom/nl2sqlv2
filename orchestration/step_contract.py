"""Pipeline step contracts for fail-fast validation.

Each pipeline step declares its required inputs and expected outputs.
The pipeline runner validates these before and after execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepContract:
    """Declares the contract for a single pipeline step.

    Attributes:
        name: Unique step identifier.
        required: Whether this step must succeed for the pipeline to continue.
        inputs: List of file/directory paths that must exist before the step runs.
        outputs: List of file/directory paths that must exist after the step completes.
        can_skip: Whether the step may be skipped (e.g. disabled in config).
        skip_reason: If skipped, the reason must be recorded here.
    """
    name: str
    required: bool = True
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    can_skip: bool = False
    skip_reason: str | None = None

    def with_skip(self, reason: str) -> "StepContract":
        """Return a copy of this contract marked as skippable."""
        return StepContract(
            name=self.name,
            required=False,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            can_skip=True,
            skip_reason=reason,
        )
