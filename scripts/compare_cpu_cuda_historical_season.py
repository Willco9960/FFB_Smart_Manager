"""Compare the CPU and CUDA season engines on the same historical inputs.

The default comparison disables transactions so draft, weekly lineup scoring,
standings, and playoffs can be checked exactly. ``--transactions`` enables the
experimental tensorized waiver/trade path and reports outcome deltas rather
than claiming action-level parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from agents.genome_draft_agent import GenomeDraftAgent
from agents.trade_agent import GenomeTradeAgent
from agents.waiver_agent import GenomeWaiverAgent
from evolution.genome import create_random_genome
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.player import Player
from fantasy_engine.playoffs import simulate_espn_six_team_playoffs
from fantasy_engine.season import ESPN_TEN_TEAM_DEFAULT_RULES, rank_standings
from fantasy_engine.team import Team
from fantasy_engine.weekly_season_simulation import run_historical_regular_season
from gpu_sim.full_season import run_full_cuda_season
from gpu_sim.historical_adapter import create_historical_cuda_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2021)
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--transactions", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/cpu_cuda_historical_parity.json")
    )
    return parser.parse_args()


def build_cpu_league(inputs) -> League:
    return League(
        name=f"CPU parity {inputs.season}",
        teams=[Team(name=f"Team {index}") for index in range(1, 11)],
        available_players=[
            Player(
                name=player.name,
                position=player.position,
                team=player.team,
                projected_score=player.projected_score,
                actual_score=player.actual_score,
            )
            for player in inputs.players
        ],
    )


class ProjectionShapeDraftAgent:
    """CPU reference for the tensorized projection-plus-shape draft."""

    def __init__(self) -> None:
        self.shape_agent = GenomeDraftAgent(genome=create_random_genome(seed=1))

    def choose_player(self, available_players, team, league):
        eligible = self.shape_agent.get_eligible_players(
            available_players=available_players,
            team=team,
            league=league,
        )
        return max(eligible, key=lambda player: player.projected_score)


def summarize_cpu(inputs, transactions: bool) -> dict[str, object]:
    league = build_cpu_league(inputs)
    run_snake_draft(league=league, rounds=16, draft_agent=ProjectionShapeDraftAgent())
    waiver_agents = None
    trade_agents = None
    if transactions:
        transaction_genome = create_random_genome(seed=1)
        waiver_agents = {
            team.name: GenomeWaiverAgent(
                genome=transaction_genome,
                lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
            )
            for team in league.teams
        }
        trade_agents = {
            team.name: GenomeTradeAgent(
                genome=transaction_genome,
                lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
            )
            for team in league.teams
        }
    regular = run_historical_regular_season(
        league=league,
        performances=list(inputs.performances),
        rules=ESPN_TEN_TEAM_DEFAULT_RULES,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
        waiver_agents=waiver_agents,
        trade_agents=trade_agents,
    )
    playoffs = simulate_espn_six_team_playoffs(
        league=league,
        standings=regular.standings,
        performances=list(inputs.performances),
        rules=ESPN_TEN_TEAM_DEFAULT_RULES,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
    )
    standings = rank_standings(regular.standings)
    return {
        "transactions_enabled": transactions,
        "standings": [
            {
                "team": standing.team_name,
                "wins": standing.wins,
                "losses": standing.losses,
                "ties": standing.ties,
                "points_for": round(standing.points_for, 2),
                "points_against": round(standing.points_against, 2),
            }
            for standing in standings
        ],
        "champion": playoffs.champion.name,
        "weekly_scores": {
            str(week): {name: round(score, 2) for name, score in scores.items()}
            for week, scores in regular.weekly_scores.items()
        },
    }


def summarize_cuda(inputs, transactions: bool) -> dict[str, object]:
    state = inputs.state
    run_full_cuda_season(state, enable_transactions=transactions)
    standings = []
    for team_index in range(state.team_count):
        standings.append(
            {
                "team": f"Team {team_index + 1}",
                "wins": int(state.wins[0, team_index].item()),
                "losses": int(state.losses[0, team_index].item()),
                "ties": int(state.ties[0, team_index].item()),
                "points_for": round(float(state.points_for[0, team_index].item()), 2),
                "points_against": round(float(state.points_against[0, team_index].item()), 2),
            }
        )
    standings.sort(key=lambda item: (item["wins"], item["points_for"]), reverse=True)
    return {
        "transactions_enabled": transactions,
        "standings": standings,
        "champion": f"Team {int(state.champions[0].item()) + 1}",
        "waiver_counts": [int(value[0].item()) for value in state.waiver_counts],
        "trade_counts": [int(value[0].item()) for value in state.trade_counts],
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    inputs = create_historical_cuda_inputs(
        season=args.season,
        players=args.players,
        device=args.device,
    )
    cpu = summarize_cpu(inputs, args.transactions)
    cuda = summarize_cuda(inputs, args.transactions)
    report = {
        "season": args.season,
        "projection_season": inputs.projection_season,
        "players": len(inputs.players),
        "lineup_mode": "ESPN_OFFENSIVE_LINEUP_RULES",
        "transactions": args.transactions,
        "cpu": cpu,
        "cuda": cuda,
        "exact_standings_match": cpu["standings"] == cuda["standings"],
        "exact_champion_match": cpu["champion"] == cuda["champion"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Parity report saved to: {args.output}")


if __name__ == "__main__":
    main()
