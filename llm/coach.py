"""Explainable, approval-aware coaching around numerical recommendations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from llm.provider import CoachProvider, NullCoachProvider

DECISION_TYPES = ("draft", "lineup", "waiver", "trade")


def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)


@dataclass(frozen=True)
class RecommendationEvidence:
    """Structured evidence produced by the numerical policy and simulator."""

    decision_type: str
    recommendation: str
    subject: str
    confidence: float
    reasons: tuple[str, ...] = ()
    projected_delta: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        if self.decision_type not in DECISION_TYPES:
            raise ValueError(f"Unknown decision type: {self.decision_type}")
        if not self.recommendation.strip():
            raise ValueError("A recommendation is required.")
        if not self.subject.strip():
            raise ValueError("A recommendation subject is required.")
        object.__setattr__(self, "confidence", _clamp_confidence(self.confidence))

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "recommendation": self.recommendation,
            "subject": self.subject,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "projected_delta": self.projected_delta,
            "context": self.context,
            "human_approval_required": self.human_approval_required,
        }


@dataclass(frozen=True)
class CoachExplanation:
    """Safe explanation returned to the UI or terminal report."""

    summary: str
    reasons: tuple[str, ...]
    cautions: tuple[str, ...]
    confidence: float
    source: str
    human_approval_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FantasyCoach:
    """Explain recommendations without being allowed to create decisions."""

    def __init__(self, provider: CoachProvider | None = None):
        self.provider = provider or NullCoachProvider()

    def explain(self, evidence: RecommendationEvidence) -> CoachExplanation:
        prompt = self.build_prompt(evidence)
        try:
            response = self.provider.complete(prompt)
            return self._parse_response(response, evidence)
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            return self._fallback_explanation(evidence)

    def build_prompt(self, evidence: RecommendationEvidence) -> str:
        payload = json.dumps(evidence.to_prompt_payload(), sort_keys=True)
        return (
            "Explain the following numerical fantasy recommendation. "
            "Do not change the recommendation, invent facts, or propose an "
            "unvalidated transaction. Return JSON with exactly these keys: "
            "summary (string), reasons (array of strings), cautions (array of strings), "
            "confidence (number from 0 to 1).\n\n"
            f"EVIDENCE_JSON={payload}"
        )

    def _parse_response(
        self,
        response: str,
        evidence: RecommendationEvidence,
    ) -> CoachExplanation:
        payload = json.loads(response)
        if not isinstance(payload, dict):
            raise ValueError("Coach response must be a JSON object.")

        summary = payload.get("summary")
        reasons = payload.get("reasons")
        cautions = payload.get("cautions")
        confidence = payload.get("confidence", evidence.confidence)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Coach response has no summary.")
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise ValueError("Coach response reasons must be strings.")
        if not isinstance(cautions, list) or not all(isinstance(item, str) for item in cautions):
            raise ValueError("Coach response cautions must be strings.")

        return CoachExplanation(
            summary=summary.strip(),
            reasons=tuple(reasons[:5]),
            cautions=tuple(cautions[:5]),
            confidence=_clamp_confidence(float(confidence)),
            source="local_llm",
            human_approval_required=True,
        )

    def _fallback_explanation(self, evidence: RecommendationEvidence) -> CoachExplanation:
        delta_text = ""
        if evidence.projected_delta is not None:
            delta_text = f" Projected change: {evidence.projected_delta:+.2f}."
        return CoachExplanation(
            summary=f"{evidence.recommendation}: {evidence.subject}.{delta_text}",
            reasons=evidence.reasons[:5],
            cautions=(
                "Explanation generated without a local language model.",
                "Review and approve the recommendation before applying it.",
            ),
            confidence=evidence.confidence,
            source="deterministic_fallback",
            human_approval_required=True,
        )
