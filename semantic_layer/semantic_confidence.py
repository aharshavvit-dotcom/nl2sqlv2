from __future__ import annotations


class SemanticConfidence:
    high_confidence_threshold = 0.82
    minimum_match_threshold = 0.60
    ambiguity_gap = 0.08

    def classify(self, score: float, alternatives: list[dict]) -> dict:
        ambiguous = False
        if score < self.minimum_match_threshold:
            ambiguous = bool(alternatives)
        elif len(alternatives) > 1 and float(alternatives[1].get("score", 0.0)) >= score - self.ambiguity_gap:
            ambiguous = True
        return {
            "score": round(float(score), 4),
            "high_confidence": score >= self.high_confidence_threshold and not ambiguous,
            "ambiguous": ambiguous,
            "requires_clarification": ambiguous or score < self.minimum_match_threshold,
        }
