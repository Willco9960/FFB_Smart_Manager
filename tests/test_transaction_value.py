import math
from pathlib import Path

from agents.hybrid_transaction_agents import HybridWaiverAgent
from evolution.offline_replay import DecisionReplayRecord
from evolution.transaction_value_training import (
    split_transaction_records,
    train_transaction_value_model,
    train_transaction_value_model_with_validation,
)
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import WaiverClaim
from models.league_state_encoder import LEAGUE_STATE_FEATURE_NAMES
from models.modular_manager_policy import ModularPolicyFeatures
from models.transaction_value import (
    TransactionValueNetwork,
    load_transaction_value_model,
    save_transaction_value_model,
)


def features(value: float) -> ModularPolicyFeatures:
    return ModularPolicyFeatures(
        player_values=(value,) * 13,
        state_values=(value,) * len(LEAGUE_STATE_FEATURE_NAMES),
    )


def record(decision_type: str, reward: float) -> DecisionReplayRecord:
    return DecisionReplayRecord(
        season=2021,
        week=4,
        decision_type=decision_type,
        action_key="player",
        features=features(reward / 10.0),
        reward=reward,
        team_name="Team",
        source="test",
    )


def test_transaction_value_model_trains_and_scores():
    model = TransactionValueNetwork()
    loss, count = train_transaction_value_model(
        model,
        [record("waiver", -4.0), record("waiver", 10.0), record("trade", 2.0)],
        epochs=3,
    )

    score, uncertainty = model.score(features(0.5), "waiver")

    assert count == 3
    assert math.isfinite(loss)
    assert isinstance(score, float)
    assert uncertainty > 0.0


def test_transaction_value_model_round_trips(tmp_path: Path):
    path = tmp_path / "transaction_value.pt"
    model = TransactionValueNetwork()
    train_transaction_value_model(model, [record("waiver", 5.0)], epochs=1)
    save_transaction_value_model(model, path)

    loaded = load_transaction_value_model(path)
    original = model.score(features(0.25), "waiver")
    restored = loaded.score(features(0.25), "waiver")

    assert restored == original


def test_transaction_value_split_is_chronological():
    records = [
        record("waiver", 1.0),
        record("waiver", 2.0),
    ]
    records = [
        DecisionReplayRecord(
            season=season,
            week=item.week,
            decision_type=item.decision_type,
            action_key=item.action_key,
            features=item.features,
            reward=item.reward,
            team_name=item.team_name,
            source=item.source,
        )
        for season, item in zip((2020, 2021), records, strict=True)
    ]

    train_records, validation_records, holdout_seasons = split_transaction_records(
        records,
        holdout_seasons=1,
    )

    assert [item.season for item in train_records] == [2020]
    assert [item.season for item in validation_records] == [2021]
    assert holdout_seasons == (2021,)


def test_transaction_value_validation_gate_requires_enough_holdout_data():
    model = TransactionValueNetwork()
    loss, count, validation = train_transaction_value_model_with_validation(
        model,
        [
            DecisionReplayRecord(
                season=season,
                week=4,
                decision_type="waiver",
                action_key=str(season),
                features=features(float(season)),
                reward=float(season),
            )
            for season in (2020, 2021)
        ],
        epochs=1,
        holdout_seasons=1,
        minimum_validation_records=10,
    )

    assert math.isfinite(loss)
    assert count == 1
    assert validation.validation_records == 1
    assert not validation.approved


class NegativeValueModel:
    def score_normalized(self, features, decision_type):
        return -2.0, 0.1


class FixedWaiverAgent:
    def __init__(self, claim):
        self.claim = claim

    def choose_waiver_claim(self, team, available_players, league, week):
        return self.claim

    def get_projected_lineup_score(self, team):
        return sum(player.projected_score for player in team.roster)


def test_hybrid_waiver_can_abstain_when_value_model_is_negative():
    team = Team(
        name="Team",
        roster=[
            Player("QB", "QB", "T", projected_score=20),
            Player("RB1", "RB", "T", projected_score=18),
            Player("RB2", "RB", "T", projected_score=17),
            Player("WR1", "WR", "T", projected_score=16),
            Player("WR2", "WR", "T", projected_score=15),
            Player("TE", "TE", "T", projected_score=12),
        ],
    )
    add = Player("Add", "WR", "T", projected_score=30)
    claim = WaiverClaim("Team", add, team.roster[-1], 4)
    league = League(name="Test", teams=[team])

    result = HybridWaiverAgent(
        neural_agent=FixedWaiverAgent(claim),
        fallback_agent=FixedWaiverAgent(None),
        value_model=NegativeValueModel(),
    ).choose_waiver_claim(team, [add], league, 4)

    assert result is None
