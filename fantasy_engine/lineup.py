from dataclasses import dataclass

from fantasy_engine.player import Player


@dataclass(frozen=True)
class LineupSlot:
    name: str
    eligible_positions: tuple[str, ...]
    count: int = 1


@dataclass
class StartingLineup:
    players: list[Player]
    missing_slots: list[str]

    def is_complete(self) -> bool:
        return len(self.missing_slots) == 0

    def score(self) -> float:
        return round(sum(player.actual_score for player in self.players), 2)


ESPN_DEFAULT_LINEUP_RULES = (
    LineupSlot(name="QB", eligible_positions=("QB",), count=1),
    LineupSlot(name="RB", eligible_positions=("RB",), count=2),
    LineupSlot(name="WR", eligible_positions=("WR",), count=2),
    LineupSlot(name="TE", eligible_positions=("TE",), count=1),
    LineupSlot(name="FLEX", eligible_positions=("RB", "WR", "TE"), count=1),
    LineupSlot(name="DST", eligible_positions=("DST", "DEF", "D/ST"), count=1),
    LineupSlot(name="K", eligible_positions=("K",), count=1),
)


ESPN_OFFENSIVE_LINEUP_RULES = (
    LineupSlot(name="QB", eligible_positions=("QB",), count=1),
    LineupSlot(name="RB", eligible_positions=("RB",), count=2),
    LineupSlot(name="WR", eligible_positions=("WR",), count=2),
    LineupSlot(name="TE", eligible_positions=("TE",), count=1),
    LineupSlot(name="FLEX", eligible_positions=("RB", "WR", "TE"), count=1),
)


def normalize_position(position: str) -> str:
    if position in ("DEF", "D/ST"):
        return "DST"

    return position


def player_matches_slot(player: Player, lineup_slot: LineupSlot) -> bool:
    player_position = normalize_position(player.position)
    eligible_positions = tuple(
        normalize_position(position) for position in lineup_slot.eligible_positions
    )

    return player_position in eligible_positions


def find_best_player_for_slot(
    available_players: list[Player],
    lineup_slot: LineupSlot,
    selection_score_attribute: str = "actual_score",
    selection_scores: dict[int, float] | None = None,
) -> Player | None:
    eligible_players = []

    for player in available_players:
        if player_matches_slot(player, lineup_slot):
            eligible_players.append(player)

    if not eligible_players:
        return None

    if selection_scores is not None:
        return max(eligible_players, key=lambda player: selection_scores[id(player)])

    return max(eligible_players, key=lambda player: getattr(player, selection_score_attribute))


def build_best_starting_lineup(
    roster: list[Player],
    lineup_rules: tuple[LineupSlot, ...] = ESPN_DEFAULT_LINEUP_RULES,
    require_complete_lineup: bool = True,
    selection_score_attribute: str = "actual_score",
    selection_scores: dict[int, float] | None = None,
) -> StartingLineup:
    available_players = list(roster)
    starting_players = []
    missing_slots = []

    for lineup_slot in lineup_rules:
        for _ in range(lineup_slot.count):
            selected_player = find_best_player_for_slot(
                available_players=available_players,
                lineup_slot=lineup_slot,
                selection_score_attribute=selection_score_attribute,
                selection_scores=selection_scores,
            )

            if selected_player is None:
                missing_slots.append(lineup_slot.name)
                continue

            starting_players.append(selected_player)
            available_players.remove(selected_player)

    starting_lineup = StartingLineup(
        players=starting_players,
        missing_slots=missing_slots,
    )

    if require_complete_lineup and not starting_lineup.is_complete():
        raise ValueError(
            f"Could not build a complete lineup. Missing slots: {', '.join(missing_slots)}"
        )

    return starting_lineup


def score_starting_lineup(
    roster: list[Player],
    lineup_rules: tuple[LineupSlot, ...] = ESPN_DEFAULT_LINEUP_RULES,
    require_complete_lineup: bool = True,
    selection_score_attribute: str = "actual_score",
) -> float:
    starting_lineup = build_best_starting_lineup(
        roster=roster,
        lineup_rules=lineup_rules,
        require_complete_lineup=require_complete_lineup,
        selection_score_attribute=selection_score_attribute,
    )

    return starting_lineup.score()
