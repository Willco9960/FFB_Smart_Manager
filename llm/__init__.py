"""Optional local language-model support for explainable fantasy decisions."""

from llm.coach import CoachExplanation, FantasyCoach, RecommendationEvidence
from llm.provider import (
    CoachProvider,
    LocalOpenAICompatibleProvider,
    NullCoachProvider,
    create_coach_provider_from_env,
)

__all__ = [
    "CoachExplanation",
    "CoachProvider",
    "FantasyCoach",
    "LocalOpenAICompatibleProvider",
    "NullCoachProvider",
    "RecommendationEvidence",
    "create_coach_provider_from_env",
]
