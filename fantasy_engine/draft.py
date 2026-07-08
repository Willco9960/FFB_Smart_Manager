from dataclasses import dataclass
from typing import Protocol

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team


class DraftAgent(Protocol):
    def choose_player(
        self,
        available_players: list[Player],
        team: Team,
        league: League,
    ) -> Player: ...


@dataclass
class DraftPick:
    round_number: int
    pick_number: int
    team_name: str
    player: Player


def get_snake_draft_order(teams: list[Team], round_number: int) -> list[Team]:
    if round_number % 2 == 1:
        return teams

    return list(reversed(teams))


def select_best_available_player(available_players: list[Player]) -> Player:
    return max(available_players, key=lambda player: player.projected_score)


def run_snake_draft(
    league: League,
    rounds: int = 16,
    draft_agent: DraftAgent | None = None,
) -> list[DraftPick]:
    draft_results = []
    pick_number = 1

    for round_number in range(1, rounds + 1):
        draft_order = get_snake_draft_order(league.teams, round_number)

        for team in draft_order:
            if draft_agent is None:
                selected_player = select_best_available_player(league.available_players)
            else:
                selected_player = draft_agent.choose_player(
                    available_players=league.available_players,
                    team=team,
                    league=league,
                )

            team.add_player(selected_player)
            league.available_players.remove(selected_player)

            draft_pick = DraftPick(
                round_number=round_number,
                pick_number=pick_number,
                team_name=team.name,
                player=selected_player,
            )

            draft_results.append(draft_pick)
            pick_number += 1

    return draft_results


def format_draft_results(draft_results: list[DraftPick]) -> str:
    lines = []

    for pick in draft_results:
        line = (
            f"Pick {pick.pick_number}: "
            f"Round {pick.round_number} - "
            f"{pick.team_name} selected "
            f"{pick.player.name} "
            f"({pick.player.position}, {pick.player.projected_score})"
        )

        lines.append(line)

    return "\n".join(lines)
