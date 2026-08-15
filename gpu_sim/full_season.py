"""Experimental tensorized full-season stages.

This module keeps league state in tensors and supports the same broad stages as
the reference engine: draft, weekly projected lineups, inverse-standings
waivers, mutually beneficial one-for-one trades, head-to-head standings, and
the ESPN six-team playoff bracket. It is intentionally not wired into the
production trainer until scenario-level parity reports are available.

Transaction decisions are currently a fast tensorized approximation of the
CPU agents. Waiver claims are selected in batched inverse-standings passes and
trades search one-for-one top-K candidates. The CPU engine remains the
behavioral authority until exact action-by-action parity is demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gpu_sim.tensorized_draft import (
    DraftBatchResult,
    run_batched_roster_aware_draft,
    score_batched_lineups,
)


@dataclass
class CudaSeasonState:
    """Mutable tensor state for a batch of simulated leagues."""

    draft_projections: torch.Tensor
    weekly_projections: torch.Tensor
    weekly_actual_points: torch.Tensor
    positions: torch.Tensor
    team_count: int = 10
    roster_size: int = 16
    rosters: torch.Tensor | None = None
    available: torch.Tensor | None = None
    wins: torch.Tensor | None = None
    losses: torch.Tensor | None = None
    ties: torch.Tensor | None = None
    points_for: torch.Tensor | None = None
    points_against: torch.Tensor | None = None
    weekly_scores: list[torch.Tensor] = field(default_factory=list)
    waiver_counts: list[torch.Tensor] = field(default_factory=list)
    trade_counts: list[torch.Tensor] = field(default_factory=list)
    playoff_wins: torch.Tensor | None = None
    champions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.draft_projections.ndim != 2:
            raise ValueError("draft_projections must have shape [scenarios, players].")
        if self.weekly_projections.ndim != 3 or self.weekly_actual_points.ndim != 3:
            raise ValueError("Weekly tensors must have shape [scenarios, weeks, players].")
        if self.weekly_projections.shape != self.weekly_actual_points.shape:
            raise ValueError("Weekly projection and actual tensors must have the same shape.")
        if self.weekly_projections.shape[0] != self.draft_projections.shape[0]:
            raise ValueError("All tensors must share scenario count.")
        if self.weekly_projections.shape[2] != self.draft_projections.shape[1]:
            raise ValueError("All tensors must share player count.")
        if self.positions.shape != (self.draft_projections.shape[1],):
            raise ValueError("positions must describe every player.")
        if self.team_count < 2 or self.roster_size < 1:
            raise ValueError("team_count and roster_size must be positive.")
        if self.rosters is None:
            self.rosters = torch.full(
                (
                    self.scenario_count,
                    self.team_count,
                    self.roster_size,
                ),
                -1,
                dtype=torch.long,
                device=self.device,
            )
        if self.available is None:
            self.available = torch.ones(
                (self.scenario_count, self.player_count),
                dtype=torch.bool,
                device=self.device,
            )
        if self.wins is None:
            self.wins = torch.zeros(
                (self.scenario_count, self.team_count),
                dtype=torch.int32,
                device=self.device,
            )
        if self.losses is None:
            self.losses = torch.zeros_like(self.wins)
        if self.ties is None:
            self.ties = torch.zeros_like(self.wins)
        if self.points_for is None:
            self.points_for = torch.zeros(
                (self.scenario_count, self.team_count),
                dtype=torch.float32,
                device=self.device,
            )
        if self.points_against is None:
            self.points_against = torch.zeros_like(self.points_for)

    @property
    def device(self) -> torch.device:
        return self.draft_projections.device

    @property
    def scenario_count(self) -> int:
        return self.draft_projections.shape[0]

    @property
    def player_count(self) -> int:
        return self.draft_projections.shape[1]

    @property
    def week_count(self) -> int:
        return self.weekly_projections.shape[1]

    def draft(self) -> DraftBatchResult:
        """Draft fixed-size rosters and update free-agent masks."""

        if self.roster_size * self.team_count > self.player_count:
            raise ValueError("Not enough players to fill every roster.")
        result = run_batched_roster_aware_draft(
            self.draft_projections,
            self.positions,
            team_count=self.team_count,
            rounds=self.roster_size,
        )
        self.rosters = result.player_indices
        self.available = torch.ones_like(self.available)
        self.available.scatter_(1, self.rosters.reshape(self.scenario_count, -1), False)
        return result

    def projected_lineup_scores(self, week: int) -> torch.Tensor:
        self._validate_week(week)
        return score_batched_lineups(
            self.weekly_projections[:, week],
            self.weekly_projections[:, week],
            self.positions,
            self._require_rosters(),
        ).scores

    def score_week(self, week: int, matchups: torch.Tensor | None = None) -> torch.Tensor:
        """Score legal projected lineups and update head-to-head standings."""

        self._validate_week(week)
        scores = score_batched_lineups(
            self.weekly_projections[:, week],
            self.weekly_actual_points[:, week],
            self.positions,
            self._require_rosters(),
        ).scores
        self.points_for += scores
        if matchups is None:
            matchups = self.default_matchups(week)
        self._record_matchups(scores, matchups)
        self.weekly_scores.append(scores)
        return scores

    def apply_waivers(self, week: int, minimum_improvement: float = 0.5) -> torch.Tensor:
        """Apply one projection-positive waiver per inverse-standings rank."""

        self._validate_week(week)
        batch = torch.arange(self.scenario_count, device=self.device)
        ranking_value = self.wins.to(torch.float32) * 100000.0 + self.points_for
        priority = torch.argsort(ranking_value, dim=1, descending=False)
        accepted_count = torch.zeros(self.scenario_count, dtype=torch.int32, device=self.device)
        projections = self.weekly_projections[:, week]

        # Select one claim per inverse-standings rank while updating a local
        # availability mask. Lineup legality and improvement are then scored
        # for every rank in two large tensor passes instead of two launches per
        # team. A rejected claim is restored before later ranks are applied.
        working_available = self.available.clone()
        current_rosters_by_rank = self._require_rosters()[batch.unsqueeze(1), priority]
        add_players_by_rank = torch.empty_like(priority)
        drop_slots_by_rank = torch.empty_like(priority)
        for rank in range(self.team_count):
            add_scores = projections.masked_fill(~working_available, float("-inf"))
            add_players = add_scores.argmax(dim=1)
            add_players_by_rank[:, rank] = add_players
            working_available.scatter_(1, add_players.unsqueeze(1), False)
            current_values = projections.gather(
                1,
                current_rosters_by_rank[:, rank],
            )
            drop_slots_by_rank[:, rank] = current_values.argmin(dim=1)

        candidate_rosters_by_rank = current_rosters_by_rank.clone()
        rank_batch = torch.arange(self.scenario_count, device=self.device).unsqueeze(1)
        rank_indices = torch.arange(self.team_count, device=self.device).unsqueeze(0)
        candidate_rosters_by_rank[rank_batch, rank_indices, drop_slots_by_rank] = (
            add_players_by_rank
        )
        flat_rosters = current_rosters_by_rank.reshape(
            self.scenario_count * self.team_count,
            1,
            self.roster_size,
        )
        flat_candidates = candidate_rosters_by_rank.reshape(
            self.scenario_count * self.team_count,
            1,
            self.roster_size,
        )
        flat_projections = (
            projections.unsqueeze(1)
            .expand(
                -1,
                self.team_count,
                -1,
            )
            .reshape(self.scenario_count * self.team_count, self.player_count)
        )
        current_scores = score_batched_lineups(
            flat_projections,
            flat_projections,
            self.positions,
            flat_rosters,
        ).scores.reshape(self.scenario_count, self.team_count)
        candidate_scores = score_batched_lineups(
            flat_projections,
            flat_projections,
            self.positions,
            flat_candidates,
        ).scores.reshape(self.scenario_count, self.team_count)
        accepted = torch.isfinite(working_available.max(dim=1).values).unsqueeze(1) & (
            candidate_scores - current_scores >= minimum_improvement
        )
        accepted_count = accepted.sum(dim=1).to(torch.int32)

        for rank in range(self.team_count):
            update_mask = accepted[:, rank]
            team_indices = priority[:, rank]
            add_players = add_players_by_rank[:, rank]
            drop_slots = drop_slots_by_rank[:, rank]
            old_players = (
                current_rosters_by_rank[:, rank]
                .gather(
                    1,
                    drop_slots.unsqueeze(1),
                )
                .squeeze(1)
            )
            self._require_rosters()[
                batch[update_mask], team_indices[update_mask], drop_slots[update_mask]
            ] = add_players[update_mask]
            self.available[batch[update_mask], add_players[update_mask]] = False
            self.available[batch[update_mask], old_players[update_mask]] = True

        self.waiver_counts.append(accepted_count)
        return accepted_count

    def apply_trades(
        self, week: int, top_k: int = 3, minimum_improvement: float = 0.5
    ) -> torch.Tensor:
        """Apply the best mutually beneficial one-for-one trade per scenario."""

        self._validate_week(week)
        if self.team_count < 2:
            return torch.zeros(self.scenario_count, dtype=torch.int32, device=self.device)
        projections = self.weekly_projections[:, week]
        rosters = self._require_rosters()
        baseline = score_batched_lineups(
            projections,
            projections,
            self.positions,
            rosters,
        ).scores
        top_k = min(top_k, self.roster_size)

        pair_indices = torch.triu_indices(
            self.team_count,
            self.team_count,
            offset=1,
            device=self.device,
        )
        pair_a, pair_b = pair_indices[0], pair_indices[1]
        pair_count = pair_a.shape[0]
        combos = top_k * top_k
        pair_rosters_a = rosters[:, pair_a]
        pair_rosters_b = rosters[:, pair_b]
        pair_projection = projections.unsqueeze(1).expand(-1, pair_count, -1)
        values_a = pair_projection.gather(2, pair_rosters_a)
        values_b = pair_projection.gather(2, pair_rosters_b)
        _, top_slots_a = values_a.topk(top_k, dim=2)
        _, top_slots_b = values_b.topk(top_k, dim=2)
        a_slots = top_slots_a.unsqueeze(3).expand(-1, -1, -1, top_k)
        a_slots = a_slots.reshape(self.scenario_count, pair_count, combos)
        b_slots = top_slots_b.unsqueeze(2).expand(-1, -1, top_k, -1)
        b_slots = b_slots.reshape(self.scenario_count, pair_count, combos)
        base_a = pair_rosters_a.unsqueeze(2).expand(-1, -1, combos, -1).clone()
        base_b = pair_rosters_b.unsqueeze(2).expand(-1, -1, combos, -1).clone()
        a_players = (
            pair_rosters_a.unsqueeze(2)
            .expand(-1, -1, combos, -1)
            .gather(
                3,
                a_slots.unsqueeze(3),
            )
            .squeeze(3)
        )
        b_players = (
            pair_rosters_b.unsqueeze(2)
            .expand(-1, -1, combos, -1)
            .gather(
                3,
                b_slots.unsqueeze(3),
            )
            .squeeze(3)
        )
        flat_a = base_a.reshape(-1, self.roster_size)
        flat_b = base_b.reshape(-1, self.roster_size)
        flat_rows = torch.arange(flat_a.shape[0], device=self.device)
        flat_a[flat_rows, a_slots.reshape(-1)] = b_players.reshape(-1)
        flat_b[flat_rows, b_slots.reshape(-1)] = a_players.reshape(-1)
        repeated_projection = projections.repeat_interleave(pair_count * combos, dim=0)
        score_a = (
            score_batched_lineups(
                repeated_projection,
                repeated_projection,
                self.positions,
                flat_a.unsqueeze(1),
            )
            .scores[:, 0]
            .reshape(self.scenario_count, pair_count, combos)
        )
        score_b = (
            score_batched_lineups(
                repeated_projection,
                repeated_projection,
                self.positions,
                flat_b.unsqueeze(1),
            )
            .scores[:, 0]
            .reshape(self.scenario_count, pair_count, combos)
        )
        baseline_a = baseline[:, pair_a].unsqueeze(2)
        baseline_b = baseline[:, pair_b].unsqueeze(2)
        improvement_a = score_a - baseline_a
        improvement_b = score_b - baseline_b
        valid = (improvement_a >= minimum_improvement) & (improvement_b >= minimum_improvement)
        candidate_gain = (improvement_a + improvement_b).masked_fill(
            ~valid,
            float("-inf"),
        )
        best_gain, flat_choice = candidate_gain.reshape(self.scenario_count, -1).max(dim=1)
        best_pair = flat_choice // combos
        best_combo = flat_choice % combos
        best_a = pair_a[best_pair]
        best_b = pair_b[best_pair]
        best_slot_a = a_slots[
            torch.arange(self.scenario_count, device=self.device), best_pair, best_combo
        ]
        best_slot_b = b_slots[
            torch.arange(self.scenario_count, device=self.device), best_pair, best_combo
        ]

        accepted = torch.isfinite(best_gain)
        batch = torch.arange(self.scenario_count, device=self.device)
        old_a = rosters[batch, best_a, best_slot_a].clone()
        old_b = rosters[batch, best_b, best_slot_b].clone()
        rosters[batch[accepted], best_a[accepted], best_slot_a[accepted]] = old_b[accepted]
        rosters[batch[accepted], best_b[accepted], best_slot_b[accepted]] = old_a[accepted]
        accepted_count = accepted.to(torch.int32)
        self.trade_counts.append(accepted_count)
        return accepted_count

    def run_playoffs(self) -> torch.Tensor:
        """Run the ESPN six-team bracket using weeks 15-17 (1-based)."""

        if self.week_count < 17:
            raise ValueError("At least 17 weekly tensors are required for playoffs.")
        ranking_value = self.wins.to(torch.float32) * 100000.0 + self.points_for
        seeds = torch.argsort(ranking_value, dim=1, descending=True)
        playoff_wins = torch.zeros_like(self.wins)

        week_15 = self._score_without_standings(14)
        first_winner = torch.where(
            week_15.gather(1, seeds[:, 2:3]).squeeze(1)
            >= week_15.gather(1, seeds[:, 5:6]).squeeze(1),
            seeds[:, 2],
            seeds[:, 5],
        )
        second_winner = torch.where(
            week_15.gather(1, seeds[:, 3:4]).squeeze(1)
            >= week_15.gather(1, seeds[:, 4:5]).squeeze(1),
            seeds[:, 3],
            seeds[:, 4],
        )
        playoff_wins.scatter_add_(
            1,
            first_winner.unsqueeze(1),
            torch.ones_like(first_winner.unsqueeze(1), dtype=playoff_wins.dtype),
        )
        playoff_wins.scatter_add_(
            1,
            second_winner.unsqueeze(1),
            torch.ones_like(second_winner.unsqueeze(1), dtype=playoff_wins.dtype),
        )

        week_16 = self._score_without_standings(15)
        seed_one = seeds[:, 0]
        seed_two = seeds[:, 1]
        lowest_remaining = torch.maximum(first_winner, second_winner)
        other_remaining = torch.minimum(first_winner, second_winner)
        semi_one = torch.where(
            week_16.gather(1, seed_one.unsqueeze(1)).squeeze(1)
            >= week_16.gather(1, lowest_remaining.unsqueeze(1)).squeeze(1),
            seed_one,
            lowest_remaining,
        )
        semi_two = torch.where(
            week_16.gather(1, seed_two.unsqueeze(1)).squeeze(1)
            >= week_16.gather(1, other_remaining.unsqueeze(1)).squeeze(1),
            seed_two,
            other_remaining,
        )
        playoff_wins.scatter_add_(
            1,
            semi_one.unsqueeze(1),
            torch.ones_like(semi_one.unsqueeze(1), dtype=playoff_wins.dtype),
        )
        playoff_wins.scatter_add_(
            1,
            semi_two.unsqueeze(1),
            torch.ones_like(semi_two.unsqueeze(1), dtype=playoff_wins.dtype),
        )

        week_17 = self._score_without_standings(16)
        champion = torch.where(
            week_17.gather(1, semi_one.unsqueeze(1)).squeeze(1)
            >= week_17.gather(1, semi_two.unsqueeze(1)).squeeze(1),
            semi_one,
            semi_two,
        )
        playoff_wins.scatter_add_(
            1,
            champion.unsqueeze(1),
            torch.ones_like(champion.unsqueeze(1), dtype=playoff_wins.dtype),
        )
        self.playoff_wins = playoff_wins
        self.champions = champion
        return champion

    def _score_without_standings(self, week: int) -> torch.Tensor:
        self._validate_week(week)
        # The current CPU playoff path uses the roster's preseason
        # ``projected_score`` when no projection service is supplied. Keep this
        # distinction explicit: regular-season weeks use rolling weekly
        # projections, while playoff parity uses draft projections.
        return score_batched_lineups(
            self.draft_projections,
            self.weekly_actual_points[:, week],
            self.positions,
            self._require_rosters(),
        ).scores

    def default_matchups(self, week: int = 0) -> torch.Tensor:
        """Return the CPU engine's rotating round-robin pairings."""

        rotating_teams = list(range(self.team_count))
        for _ in range(week % (self.team_count - 1)):
            rotating_teams = [
                rotating_teams[0],
                rotating_teams[-1],
                *rotating_teams[1:-1],
            ]
        return torch.tensor(
            [
                (rotating_teams[index], rotating_teams[-index - 1])
                for index in range(self.team_count // 2)
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _record_matchups(self, scores: torch.Tensor, matchups: torch.Tensor) -> None:
        for first_team, second_team in matchups.tolist():
            first_scores = scores[:, first_team]
            second_scores = scores[:, second_team]
            self.points_against[:, first_team] += second_scores
            self.points_against[:, second_team] += first_scores
            tied = first_scores == second_scores
            self.wins[:, first_team] += (first_scores > second_scores).to(self.wins.dtype)
            self.wins[:, second_team] += (second_scores > first_scores).to(self.wins.dtype)
            self.losses[:, first_team] += (first_scores < second_scores).to(self.losses.dtype)
            self.losses[:, second_team] += (second_scores < first_scores).to(self.losses.dtype)
            self.ties[:, first_team] += tied.to(self.ties.dtype)
            self.ties[:, second_team] += tied.to(self.ties.dtype)

    def _require_rosters(self) -> torch.Tensor:
        if self.rosters is None or (self.rosters < 0).any():
            raise ValueError("Draft or initialize rosters before scoring a season.")
        return self.rosters

    def _validate_week(self, week: int) -> None:
        if week < 0 or week >= self.week_count:
            raise ValueError(f"Week index {week} is outside 0..{self.week_count - 1}.")


def create_synthetic_season_state(
    scenarios: int = 4,
    players: int = 256,
    *,
    team_count: int = 10,
    roster_size: int = 16,
    weeks: int = 17,
    device: torch.device | str = "cpu",
    seed: int = 20260815,
) -> CudaSeasonState:
    """Create reproducible data for full-stage parity and throughput tests."""

    if players < team_count * roster_size:
        raise ValueError("players must be large enough for all rosters.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draft_projections = torch.rand((scenarios, players), generator=generator) * 500.0
    weekly_projections = torch.rand((scenarios, weeks, players), generator=generator) * 40.0
    weekly_actual = torch.rand((scenarios, weeks, players), generator=generator) * 40.0
    positions = torch.tensor(
        ([0, 1, 1, 2, 2, 3] * ((players + 5) // 6))[:players],
        dtype=torch.long,
    )
    return CudaSeasonState(
        draft_projections=draft_projections.to(device),
        weekly_projections=weekly_projections.to(device),
        weekly_actual_points=weekly_actual.to(device),
        positions=positions.to(device),
        team_count=team_count,
        roster_size=roster_size,
    )


def run_full_cuda_season(
    state: CudaSeasonState,
    *,
    enable_transactions: bool = True,
) -> CudaSeasonState:
    """Execute draft, optional transactions, regular season, and playoffs."""

    state.draft()
    regular_season_weeks = min(14, state.week_count - 3)
    for week in range(regular_season_weeks):
        if enable_transactions:
            state.apply_waivers(week)
            state.apply_trades(week)
        state.score_week(week)
    state.run_playoffs()
    return state
