"""Compare CPU-authoritative and CUDA historical season execution.

The CPU engine creates the canonical draft and transaction trace. CUDA receives
that roster and replays the trace without drafting or selecting a second
transaction policy. This makes transaction parity a differential test rather
than a comparison of two unrelated heuristics.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

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
from fantasy_engine.transaction_contract import canonical_player_key
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
    *,
    include_trace: bool = False,
) -> dict[str, object]:
    league = build_cpu_league(inputs, roster_indices=roster_indices)
    if roster_indices is None:
        run_snake_draft(league=league, rounds=16, draft_agent=ProjectionShapeDraftAgent())
    initial_roster_indices = [
        [
            next(
                index
                for index, player in enumerate(inputs.players)
                if canonical_player_key(
                    player.player_id or player.name,
                    player.position,
                    player.team,
                )
                == (
                    canonical_player_key(
                        roster_player.player_id or roster_player.name,
                        roster_player.position,
                        roster_player.team,
                    )
                )
            )
            for roster_player in team.roster
        ]
        for team in league.teams
    ]

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
        season=inputs.season,
    )
    playoffs = simulate_espn_six_team_playoffs(
        league=league,
        standings=regular.standings,
        performances=list(inputs.performances),
        rules=ESPN_FITNESS_CONTRACT.league_rules,
        lineup_rules=ESPN_FITNESS_CONTRACT.lineup_rules,
    )
    standings = rank_standings(regular.standings)
    summary: dict[str, object] = {
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
    if include_trace:
        summary["roster_indices"] = initial_roster_indices
        summary["transaction_events"] = [asdict(event) for event in regular.transaction_events]
        summary["transaction_event_objects"] = regular.transaction_events
        summary["transaction_rewards"] = [
            {"week": week, "impacts": [asdict(impact) for impact in impacts]}
            for week, impacts in regular.weekly_transaction_impacts.items()
        ]
        summary["transaction_reward_signature"] = [
            {
                "week": impact.week,
                "transaction_type": impact.transaction_type,
                "team_name": impact.team_name,
                "incoming_points": impact.incoming_points,
                "outgoing_points": impact.outgoing_points,
                "net_points": impact.net_points,
                "reward": impact.reward,
            }
            for impacts in regular.weekly_transaction_impacts.values()
            for impact in impacts
        ]
    return summary


def summarize_cuda(
    inputs,
    transactions: bool,
    *,
    initial_rosters: list[list[int]] | None = None,
    canonical_transaction_events=None,
) -> dict[str, object]:
    state = inputs.state
    initial_roster_tensor = (
        torch.tensor([initial_rosters], dtype=torch.long, device=state.device)
        if initial_rosters is not None
        else None
    )
    run_full_cuda_season(
        state,
        enable_transactions=transactions,
        fitness_contract=ESPN_FITNESS_CONTRACT,
        initial_rosters=initial_roster_tensor,
        canonical_transaction_events=canonical_transaction_events,
    )
    standings = [
        {
            "team": f"Team {team_index + 1}",
            "wins": int(state.wins[0, team_index].item()),
            "losses": int(state.losses[0, team_index].item()),
            "ties": int(state.ties[0, team_index].item()),
            "points_for": round(float(state.points_for[0, team_index].item()), 2),
            "points_against": round(float(state.points_against[0, team_index].item()), 2),
        }
        for team_index in range(state.team_count)
    ]
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
        "transaction_events": [
            [asdict(event) for event in events] for events in state.transaction_events
        ],
        "final_transaction_state_digest": [
            state._transaction_state_digest(scenario)
            for scenario in range(state.scenario_count)
        ],
        "transaction_reward_signature": state.transaction_impacts,
    }


def _public_summary(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "transaction_event_objects"}


def _sorted_reward_signature(summary: dict[str, object]) -> list[dict[str, object]]:
    values = summary.get("transaction_reward_signature", [])
    return sorted(
        values,
        key=lambda item: (
            item["week"],
            item["transaction_type"],
            item["team_name"],
            item["incoming_points"],
            item["outgoing_points"],
            item["net_points"],
        ),
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    inputs = create_historical_cuda_inputs(
        season=args.season,
        players=args.players,
        device=args.device,
    )
    cpu = summarize_cpu(inputs, args.transactions, include_trace=args.transactions)
    roster_indices = cpu.get("roster_indices")
    replay_events = (
        [[event for event in cpu["transaction_event_objects"]]] if args.transactions else None
    )
    cuda_inputs = create_historical_cuda_inputs(
        season=args.season,
        players=args.players,
        device=args.device,
    )
    cuda = summarize_cuda(
        cuda_inputs,
        args.transactions,
        initial_rosters=roster_indices,
        canonical_transaction_events=replay_events,
    )

    cpu_events = cpu.get("transaction_events", [])
    cuda_events = cuda.get("transaction_events", [[]])[0]
    action_exact = cpu_events == cuda_events if args.transactions else True
    state_exact = (
        action_exact
        and all(
            cpu_event["pre_state_digest"] == cuda_event["pre_state_digest"]
            and cpu_event["post_state_digest"] == cuda_event["post_state_digest"]
            for cpu_event, cuda_event in zip(cpu_events, cuda_events, strict=True)
        )
        if args.transactions
        else True
    )
    report = {
        "season": args.season,
        "projection_season": inputs.projection_season,
        "players": len(inputs.players),
        "lineup_mode": "ESPN_DEFAULT_LINEUP_RULES",
        "fitness_contract": ESPN_FITNESS_CONTRACT.to_dict(),
        "transactions": args.transactions,
        "comparison_mode": "cpu_trace_replay" if args.transactions else "exact_parity",
        "cpu": _public_summary(cpu),
        "cuda": cuda,
        "exact_standings_match": cpu["standings"] == cuda["standings"],
        "exact_champion_match": cpu["champion"] == cuda["champion"],
        "exact_weekly_score_match": cpu["weekly_scores"] == cuda["weekly_scores"],
        "transaction_actions_exact": action_exact,
        "transaction_state_exact": state_exact,
        "transaction_reward_exact": (
            _sorted_reward_signature(cpu)
            == sorted(
                cuda.get("transaction_reward_signature", [[]])[0],
                key=lambda item: (
                    item["week"],
                    item["transaction_type"],
                    item["team_name"],
                    item["incoming_points"],
                    item["outgoing_points"],
                    item["net_points"],
                ),
            )
            if args.transactions
            else True
        ),
        "first_divergence": None,
    }
    if args.transactions and not action_exact:
        for index, (cpu_event, cuda_event) in enumerate(
            zip(cpu_events, cuda_events, strict=False)
        ):
            if cpu_event != cuda_event:
                report["first_divergence"] = {
                    "sequence_index": index,
                    "cpu_event": cpu_event,
                    "cuda_event": cuda_event,
                }
                break
        else:
            report["first_divergence"] = {
                "sequence_index": min(len(cpu_events), len(cuda_events)),
                "cpu_event_count": len(cpu_events),
                "cuda_event_count": len(cuda_events),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Parity report saved to: {args.output}")


if __name__ == "__main__":
    main()
