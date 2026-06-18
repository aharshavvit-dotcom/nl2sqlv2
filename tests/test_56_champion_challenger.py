from __future__ import annotations

from model_selection.champion_challenger import ChampionChallengerRegistry


def test_register_get_promote_and_persist(tmp_path) -> None:
    path = tmp_path / "registry.json"
    registry = ChampionChallengerRegistry(path)
    challenger = registry.register_challenger("neural_ir_model", "artifacts/neural", {"gold_comparison_score": 1.0})
    assert registry.get_current_champion("neural_ir_model") is None

    champion = registry.promote_challenger("neural_ir_model", challenger["challenger_id"])
    loaded = ChampionChallengerRegistry(path).get_current_champion("neural_ir_model")

    assert champion["status"] == "champion"
    assert loaded["challenger_id"] == challenger["challenger_id"]
