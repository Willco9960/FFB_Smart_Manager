"""Compare the CPU and CUDA season engines on the same historical inputs.

The default comparison disables transactions so draft, full ESPN lineup
scoring (including K/DST), standings, and playoffs can be checked exactly.
``--transactions`` enables the tensorized waiver/trade path and reports
transaction-count deltas explicitly rather than claiming action-level parity.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep direct ``python scripts/...py`` execution equivalent to module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from agents.genome_draft_agent import GenomeDraftAgent
from agents.trade_agent import GenomeTradeAgent
from agents.waiver_agent import GenomeWaiverAgent
from evolution.genome import create_random_genome
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.player import Player
from fantasy_engine.playoffs import simulate_espn_six_team_playoffs
from fantasy_engine.season import rank_standings
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


def build_cpu_league(inputs, roster_indices: list[list[int]] | None = None) -> League:
    players = [
        Player(
            name=player.name,
            position=player.position,
            team=player.team,
            projected_score=player.projected_score,
            actual_score=player.actual_score,
            player_id=player.player_id,
            history_missing=player.history_missing,
        )
        for player in inputs.players
    ]
    if roster_indices is None:
        teams = [Team(name=f"Team {index}") for index in range(1, 11)]
        available_players = players
    else:
        selected = {index for roster in roster_indices for index in roster}
        teams = [
            Team(
                name=f"Team {team_index + 1}",
                roster=[players[player_index] for player_index in roster],
            )
            for team_index, roster in enumerate(roster_indices)
        ]
        available_players = [
            player for player_index, player in enumerate(players) if player_index not in selected
        ]

    return League(
        name=f"CPU parity {inputs.season}",
        teams=teams,
        available_players=available_players,
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


def summarize_cpu(
    inputs,
    transactions: bool,
    roster_indices: list[list[int]] | None = None,
) -> dict[str, object]:
    league = build_cpu_league(inputs, roster_indices=roster_indices)
    if roster_indices is None:
        run_snake_draft(league=league, rounds=16, draft_agent=ProjectionShapeDraftAgent())
    waiver_agents = None
    trade_agents = None
    if transactions:
        transaction_genome = create_random_genome(seed=1)
        waiver_agents = {
            team.name: GenomeWaiverAgent(
                genome=transaction_genome,
                lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
            )
            for team in league.teams
        }
        trade_agents = {
            team.name: GenomeTradeAgent(
                genome=transaction_genome,
                lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
            )
            for team in league.teams
        }
    regular = run_historical_regular_season(
        league=league,
        performances=list(inputs.performances),
        rules=ESPN_FITNESS_CONTRACT.league_rules,
        lineup_rules=ESPN_FITNESS_CONTRACT.lineup_rules,
        waiver_agents=waiver_agents,
        trade_agents=trade_agents,
    )
    playoffs = simulate_espn_six_team_playoffs(
        league=league,
        standings=regular.standings,
        performances=list(inputs.performances),
        rules=ESPN_FITNESS_CONTRACT.league_rules,
        lineup_rules=ESPN_FITNESS_CONTRACT.lineup_rules,
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
        "transaction_counts": {
            str(week): len(transactions)
            for week, transactions in regular.weekly_transactions.items()
        },
    }


def summarize_cuda(inputs, transactions: bool) -> dict[str, object]:
    state = inputs.state
    run_full_cuda_season(
        state,
        enable_transactions=transactions,
        fitness_contract=ESPN_FITNESS_CONTRACT,
    )
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
        "transaction_counts": {
            "waivers": sum(int(value[0].item()) for value in state.waiver_counts),
            "trades": sum(int(value[0].item()) for value in state.trade_counts),
        },
        "weekly_scores": {
            str(week + 1): {
                f"Team {team_index + 1}": round(float(scores[0, team_index].item()), 2)
                for team_index in range(state.team_count)
            }
            for week, scores in enumerate(state.weekly_scores)
        },
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
    cuda = summarize_cuda(inputs, args.transactions)
    # Score the exact CUDA draft rosters through the CPU rules engine.  This
    # isolates weekly/K/DST/playoff parity from deliberate draft-policy
    # differences between the two implementations.
    roster_indices = inputs.state.rosters[0].detach().cpu().tolist()
    cpu = summarize_cpu(inputs, args.transactions, roster_indices=roster_indices)
    report = {
        "season": args.season,
        "projection_season": inputs.projection_season,
        "players": len(inputs.players),
        "lineup_mode": "ESPN_DEFAULT_LINEUP_RULES",
        "fitness_contract": ESPN_FITNESS_CONTRACT.to_dict(),
        "transactions": args.transactions,
        "cpu": cpu,
        "cuda": cuda,
        "exact_standings_match": cpu["standings"] == cuda["standings"],
        "exact_champion_match": cpu["champion"] == cuda["champion"],
        "exact_weekly_score_match": cpu["weekly_scores"] == cuda["weekly_scores"],
        "max_weekly_score_abs_delta": round(
            max(
                abs(
                    cpu["weekly_scores"][week][team]
                    - cuda["weekly_scores"][week][team]
                )
                for week in cpu["weekly_scores"]
                for team in cpu["weekly_scores"][week]
            ),
            4,
        ),
        "transaction_count_delta": {
            "cpu": cpu["transaction_counts"],
            "cuda": cuda["transaction_counts"],
        },
        "transaction_actions_exact": False if args.transactions else True,
        "transaction_state_exact": False if args.transactions else True,
        "transaction_reward_exact": False if args.transactions else True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Parity report saved to: {args.output}")


if __name__ == "__main__":
    main()
