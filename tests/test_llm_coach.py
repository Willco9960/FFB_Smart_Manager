import json

import pytest

from llm.coach import FantasyCoach, RecommendationEvidence
from llm.provider import LocalOpenAICompatibleProvider, NullCoachProvider


def create_evidence() -> RecommendationEvidence:
    return RecommendationEvidence(
        decision_type="waiver",
        recommendation="Add",
        subject="Test Player",
        confidence=0.85,
        reasons=("Projected lineup improvement", "Stable recent usage"),
        projected_delta=3.2,
        context={"week": 7},
    )


class StubProvider:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_evidence_rejects_unknown_decision_type():
    with pytest.raises(ValueError):
        RecommendationEvidence(
            decision_type="start_everyone",
            recommendation="Start",
            subject="Player",
            confidence=0.5,
        )


def test_fallback_is_offline_and_requires_approval():
    explanation = FantasyCoach(NullCoachProvider()).explain(create_evidence())

    assert explanation.source == "deterministic_fallback"
    assert explanation.human_approval_required
    assert explanation.confidence == 0.85
    assert "+3.20" in explanation.summary


def test_local_provider_response_is_parsed_without_changing_recommendation():
    provider = StubProvider(
        json.dumps(
            {
                "summary": "The pickup improves the projected lineup.",
                "reasons": ["Better projected role"],
                "cautions": ["Monitor injury status"],
                "confidence": 0.7,
            }
        )
    )
    explanation = FantasyCoach(provider).explain(create_evidence())

    assert explanation.source == "local_llm"
    assert explanation.reasons == ("Better projected role",)
    assert explanation.human_approval_required
    assert "EVIDENCE_JSON=" in provider.prompts[0]


def test_malformed_local_response_falls_back_safely():
    explanation = FantasyCoach(StubProvider("not json")).explain(create_evidence())

    assert explanation.source == "deterministic_fallback"
    assert explanation.human_approval_required


def test_local_provider_normalizes_chat_endpoint():
    provider = LocalOpenAICompatibleProvider(
        model="local-model",
        base_url="http://127.0.0.1:1234/v1/",
    )

    assert provider.endpoint == "http://127.0.0.1:1234/v1/chat/completions"
