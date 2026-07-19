# Testing Guide

Generated: 2026-07-15T17:11:11+00:00

## Purpose

The test suite is organized around production behaviour rather than one file per
bug, phase, helper class or implementation detail. Every retained test module is
mapped to at least one requirement in `tests/test_catalog.yaml`.

## Execution Lanes

| Lane | Command | Blocks |
| --- | --- | --- |
| Fast pull request | `pytest -m "unit or contract or safety" --tb=short` | Merge |
| Integration | `pytest -m "integration or regression" --tb=short` | Merge and release |
| Training smoke | `pytest -m "training and not slow" --tb=short` | Model promotion |
| Full pre-promotion | `pytest tests/ --tb=short` | Model promotion |
| Slow/GPU/performance | `pytest -m "slow or gpu or performance" --tb=short` | Release review when relevant |

## Markers

Registered pytest markers: `unit`, `integration`, `regression`, `e2e`,
`safety`, `contract`, `property`, `slow`, `gpu`, `database`, `performance`,
`training`, and `legacy`.

`tests/conftest.py` assigns a default lane marker during collection for older
tests that do not yet have explicit module-level marks.

## Requirement Catalog

The active requirement catalog currently maps 9 requirements.
Every retained active test file appears in `artifacts/repository_cleanup/test_inventory.json`.

## Legacy Policy

`tests/legacy` remains excluded from default pytest collection. Legacy tests are
not considered blocking until migrated into an active regression or compatibility
module. Their status is documented in `tests/legacy/README.md`.

## Cleanup Gates

T1 inventory: PASS
T2 requirement mapping: PASS
T3 consolidation: PASS
T4 legacy resolution: PASS
T5 coverage preservation: PASS
T6 final execution: REVIEW

## Latest Validation

The rationalized suite was validated with:

- `pytest --collect-only -q tests --tb=short`: 958 tests collected.
- `pytest -m "unit or contract or safety" --tb=short`: 847 passed, 1 skipped, 110 deselected.
- `pytest -m "integration or regression" --tb=short`: 88 passed, 870 deselected.
- `pytest -m "training and not slow" --tb=short`: 106 passed, 1 skipped, 851 deselected.
- `pytest tests/ --tb=short`: 957 passed, 1 skipped.

Standalone runtime smoke and production bundle smoke still need to be run after
`artifacts/model_bundle/current` is restored or promoted.
