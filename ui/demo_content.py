"""Placeholder records used to preview the future connected UI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftRecommendation:
    rank: int
    player: str
    position: str
    projection: float
    value: str
    reason: str


@dataclass(frozen=True)
class LineupDecision:
    slot: str
    player: str
    opponent: str
    espn_projection: float
    ai_projection: float
    decision: str
    reason: str


@dataclass(frozen=True)
class WaiverTarget:
    action: str
    player: str
    position: str
    projection: float
    confidence: str
    reason: str


@dataclass(frozen=True)
class TradeIdea:
    proposal: str
    send: str
    receive: str
    target_team: str
    weekly_delta: float
    fairness: int


@dataclass(frozen=True)
class ModelStatus:
    model: str
    status: str
    last_run: str
    detail: str


@dataclass(frozen=True)
class FeedItem:
    source: str
    timestamp: str
    headline: str
    detail: str


DRAFT_RECOMMENDATIONS = (
    DraftRecommendation(
        1, "CeeDee Lamb", "WR", 18.7, "+4.8 value", "Fills WR need before a tier drop."
    ),
    DraftRecommendation(
        2, "Breece Hall", "RB", 17.9, "+3.9 value", "Strong flex fit and excellent workload."
    ),
    DraftRecommendation(
        3, "Trey McBride", "TE", 15.6, "+2.6 value", "Largest projection gap at a shallow position."
    ),
    DraftRecommendation(
        4, "De'Von Achane", "RB", 16.8, "+2.2 value", "High ceiling with manageable risk."
    ),
    DraftRecommendation(
        5, "Jayden Reed", "WR", 14.9, "+1.7 value", "Useful depth and favorable upcoming schedule."
    ),
)

LINEUP_DECISIONS = (
    LineupDecision(
        "QB",
        "Jalen Hurts",
        "vs DAL",
        21.3,
        23.1,
        "START",
        "Rushing floor and strong red-zone matchup.",
    ),
    LineupDecision(
        "RB", "Bijan Robinson", "@ CAR", 17.8, 19.4, "START", "Expected touch volume remains elite."
    ),
    LineupDecision(
        "RB", "James Cook", "vs MIA", 14.2, 15.0, "START", "Receiving work raises his weekly floor."
    ),
    LineupDecision(
        "WR", "A.J. Brown", "vs DAL", 16.5, 18.2, "START", "Target share and matchup both trend up."
    ),
    LineupDecision(
        "WR",
        "Mike Evans",
        "@ NO",
        13.8,
        11.9,
        "SIT RISK",
        "Lower projected pace and secondary coverage.",
    ),
    LineupDecision(
        "TE",
        "Sam LaPorta",
        "@ MIN",
        12.1,
        13.7,
        "START",
        "Reliable route volume in a projected shootout.",
    ),
    LineupDecision(
        "FLEX",
        "Deebo Samuel",
        "@ ARI",
        13.0,
        15.8,
        "START",
        "Designed touches create a strong ceiling.",
    ),
)

WAIVER_TARGETS = (
    WaiverTarget(
        "ADD NOW",
        "Jayden Reed",
        "WR",
        15.8,
        "High",
        "Projected three-week gain; route participation rising.",
    ),
    WaiverTarget(
        "WATCHLIST", "Tyjae Spears", "RB", 12.4, "Medium", "Standalone value with handcuff upside."
    ),
    WaiverTarget(
        "ADD NOW",
        "Pat Freiermuth",
        "TE",
        10.7,
        "Medium",
        "Red-zone role improves during the bye-week stretch.",
    ),
    WaiverTarget(
        "DROP CANDIDATE",
        "Tyler Allgeier",
        "RB",
        7.2,
        "High",
        "Lowest bench value and limited starting path.",
    ),
)

TRADE_IDEAS = (
    TradeIdea("1-for-1", "Mike Evans", "Deebo Samuel", "Sunday Scaries", 2.8, 82),
    TradeIdea(
        "2-for-1", "James Cook + bench WR", "Amon-Ra St. Brown", "The Waiver Wolves", 4.6, 76
    ),
    TradeIdea("Buy-low", "Tight end depth", "Mark Andrews", "Fourth Down Club", 3.1, 79),
)

MODEL_STATUSES = (
    ModelStatus(
        "Draft projection network",
        "Demo loaded",
        "Today, 9:12 AM",
        "Season projection + positional value",
    ),
    ModelStatus(
        "Weekly projection ensemble",
        "Demo loaded",
        "Today, 9:10 AM",
        "Matchup, usage, and uncertainty features",
    ),
    ModelStatus(
        "Manager policy",
        "Demo loaded",
        "Yesterday, 11:48 PM",
        "Draft, lineup, waiver, and trade heads",
    ),
    ModelStatus(
        "ESPN / Sleeper / NFL sync",
        "Demo connected",
        "Today, 9:15 AM",
        "Live connector integration is the next milestone",
    ),
)

SEASON_WEEKS = ("W1", "W3", "W5", "W7", "W9", "W11", "W13", "W15")

LEAGUE_TRENDS = (
    ("Home League", (101, 108, 96, 119, 112, 126, 118, 132), "#5eead4"),
    ("Work League", (92, 99, 105, 101, 110, 106, 115, 121), "#fbbf24"),
    ("Dynasty League", (110, 116, 122, 118, 130, 127, 139, 145), "#a78bfa"),
    ("Friends League", (88, 94, 91, 103, 97, 109, 105, 112), "#60a5fa"),
)

PROJECTED_LEAGUE_POINTS = (
    ("Home", 121.4),
    ("Work", 108.7),
    ("Dynasty", 127.9),
    ("Friends", 103.2),
)

NEWS_UPDATES = (
    FeedItem(
        "TEAM NEWS",
        "8 min ago",
        "A.J. Brown limited in practice",
        "Monitor Friday status; matchup remains favorable if active.",
    ),
    FeedItem(
        "INJURY RADAR",
        "22 min ago",
        "Dallas offensive line downgraded",
        "Reduce confidence in short-yardage backs this week.",
    ),
    FeedItem(
        "MATCHUP ENGINE",
        "41 min ago",
        "Philadelphia implied total moved up",
        "Eagles skill players receive a small projection lift.",
    ),
    FeedItem(
        "BYE WATCH",
        "1 hr ago",
        "Three starters enter bye next week",
        "Waiver scan recommends adding an RB before the deadline.",
    ),
)

SOCIAL_UPDATES = (
    FeedItem(
        "BEAT REPORTER",
        "5 min ago",
        "@GridironBeat expects a larger red-zone role",
        "Confidence: medium · source sentiment positive.",
    ),
    FeedItem(
        "FANTASY COMMUNITY",
        "18 min ago",
        "Analyst consensus moved Reed into WR2 range",
        "Trend: 72% of tracked analysts moved up.",
    ),
    FeedItem(
        "NEWSWIRE",
        "33 min ago",
        "Coach says the backfield remains a committee",
        "Maintain a floor penalty until usage stabilizes.",
    ),
)
