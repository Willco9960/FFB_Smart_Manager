from fantasy_engine.fake_data import create_fake_player_pool

APP_NAME = "Fantasy Football AI Manager"
APP_VERSION = "0.1.0"
CURRENT_PHASE = "Phase 1 - Basic Fantasy Simulator"


def main():
    players = create_fake_player_pool()

    print(f"{APP_NAME} v{APP_VERSION}")
    print(CURRENT_PHASE)
    print("Status: ready for local development")
    print(f"Fake player pool loaded: {len(players)} players")


if __name__ == "__main__":
    main()
