from fantasy_engine.player import Player

POSITION_COUNTS = {
    "QB": 24,
    "RB": 48,
    "WR": 60,
    "TE": 24,
    "K": 16,
    "DST": 16,
}


BASE_PROJECTIONS = {
    "QB": 20.0,
    "RB": 15.0,
    "WR": 14.0,
    "TE": 10.0,
    "K": 8.0,
    "DST": 7.0,
}


def create_fake_player_pool() -> list[Player]:
    players = []

    for position, count in POSITION_COUNTS.items():
        base_projection = BASE_PROJECTIONS[position]

        for number in range(1, count + 1):
            projected_score = base_projection - (number * 0.15)
            actual_score = projected_score + ((number % 5) - 2)

            player = Player(
                name=f"{position} Player {number}",
                position=position,
                team=f"FAKE{number % 10}",
                projected_score=round(projected_score, 2),
                actual_score=round(actual_score, 2),
            )

            players.append(player)

    return players
