"""Tensorized full-season stages for CUDA benchmarking and policy training.

The module keeps league state in tensors and supports draft, weekly projected
lineups, inverse-standings waivers, mutually beneficial one-for-one trades,
head-to-head standings, and the ESPN six-team playoff bracket. The shared
fitness contract supplies starter-slot counts, K/DST legality, and reward
weights; CPU replay remains the outcome-level reference for transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT, FitnessContract
from gpu_sim.policy_draft import _build_player_features, run_batched_policy_draft
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
    lineup_policy_gains: list[torch.Tensor] = field(default_factory=list)
    waiver_policy_gains: list[torch.Tensor] = field(default_factory=list)
    trade_policy_gains: list[torch.Tensor] = field(default_factory=list)
    active_policy_network: torch.nn.Module | None = field(default=None, repr=False)
    active_team_policy_networks: list[torch.nn.Module] | None = field(default=None, repr=False)
    active_policy_team_indices: torch.Tensor | None = field(default=None, repr=False)
    draft_floors: torch.Tensor | None = None
    draft_medians: torch.Tensor | None = None
    draft_ceilings: torch.Tensor | None = None
    draft_boom_probabilities: torch.Tensor | None = None
    lineup_position_rules: tuple[tuple[int, ...], ...] = (
        (0,),
        (1,),
        (1,),
        (2,),
        (2,),
        (3,),
        (1, 2, 3),
    )
    contract_digest: str = ESPN_FITNESS_CONTRACT.digest()

    def _score_batched_lineups(
        self,
        selection_points: torch.Tensor,
        actual_points: torch.Tensor,
        positions_or_rosters: torch.Tensor,
        rosters: torch.Tensor | None = None,
    ):
        rosters = positions_or_rosters if rosters is None else rosters
        return score_batched_lineups(
            selection_points,
            actual_points,
            self.positions,
            rosters,
            lineup_position_rules=self.lineup_position_rules,
        )

    def _team_policy(self, team_index: int) -> torch.nn.Module | None:
        if self.active_team_policy_networks is not None:
            return self.active_team_policy_networks[team_index]
        return self.active_policy_network

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
        for name in (
            "draft_floors",
            "draft_medians",
            "draft_ceilings",
            "draft_boom_probabilities",
        ):
            values = getattr(self, name)
            if values is not None and values.shape != self.draft_projections.shape:
                raise ValueError(f"{name} must match draft_projections shape.")
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

    def draft(
        self,
        policy_network: torch.nn.Module | None = None,
        policy_team_indices: torch.Tensor | None = None,
        draft_anchor_weight: float = 0.20,
    ) -> DraftBatchResult:
        """Draft rosters and update free-agent masks.

        With ``policy_network`` supplied, the selected team(s) are controlled
        by the neural policy while the remaining teams use the tensorized
        projection-best baseline or archived team policies.
        """

        if self.roster_size * self.team_count > self.player_count:
            raise ValueError("Not enough players to fill every roster.")
        if policy_network is None:
            if len(self.lineup_position_rules) == 9:
                position_minimums = (1, 4, 4, 1, 1, 1)
                position_maximums = (2, 6, 7, 3, 1, 1)
            else:
                position_minimums = (1, 4, 4, 1)
                position_maximums = (2, 6, 7, 3)
            result = run_batched_roster_aware_draft(
                self.draft_projections,
                (
                    self.positions
                    if len(self.lineup_position_rules) == 9
                    else self.positions.clamp_max(3)
                ),
                team_count=self.team_count,
                rounds=self.roster_size,
                position_minimums=position_minimums,
                position_maximums=position_maximums,
            )
        else:
            result = run_batched_policy_draft(
                self.draft_projections,
                self.positions,
                policy_network,
                team_policy_networks=self.active_team_policy_networks,
                policy_team_indices=policy_team_indices,
                team_count=self.team_count,
                rounds=self.roster_size,
                anchor_weight=draft_anchor_weight,
            )
        # The policy draft kernel runs under inference_mode for throughput;
        # clone before the season mutates rosters through waivers/trades.
        self.rosters = result.player_indices.clone()
        self.available = torch.ones_like(self.available)
        self.available.scatter_(1, self.rosters.reshape(self.scenario_count, -1), False)
        return result

    def projected_lineup_scores(self, week: int) -> torch.Tensor:
        self._validate_week(week)
        return self._score_batched_lineups(
            self.weekly_projections[:, week],
            self.weekly_projections[:, week],
            self.positions,
            self._require_rosters(),
        ).scores

    def _policy_adjusted_points(
        self,
        points: torch.Tensor,
        policy_network: torch.nn.Module | None,
        decision_type: str,
        team_index: int,
        policy_override: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Apply a bounded policy tie-breaker while preserving point scale."""
        policy_network = policy_override or policy_network
        if policy_network is None:
            return points
        available = (
            self.available
            if self.available is not None
            else torch.ones(
                (self.scenario_count, self.player_count), dtype=torch.bool, device=self.device
            )
        )
        rosters = self._require_rosters()
        candidate_mask = None
        if self.active_team_policy_networks is None and self.active_policy_team_indices is not None:
            candidate_mask = self.active_policy_team_indices == team_index
            if not candidate_mask.any():
                return points
            points_for_policy = points[candidate_mask]
            available_for_policy = available[candidate_mask]
            rosters_for_policy = rosters[candidate_mask]
        else:
            points_for_policy = points
            available_for_policy = available
            rosters_for_policy = rosters
        player_features, state_features = _build_player_features(
            projected_points=points_for_policy.masked_fill(~available_for_policy, 0.0),
            positions=self.positions,
            available=available_for_policy,
            rosters=rosters_for_policy,
            team_index=team_index,
            projection_floors=(
                None if self.draft_floors is None else self.draft_floors[candidate_mask]
            )
            if candidate_mask is not None
            else self.draft_floors,
            projection_medians=(
                None if self.draft_medians is None else self.draft_medians[candidate_mask]
            )
            if candidate_mask is not None
            else self.draft_medians,
            projection_ceilings=(
                None if self.draft_ceilings is None else self.draft_ceilings[candidate_mask]
            )
            if candidate_mask is not None
            else self.draft_ceilings,
            boom_probabilities=(
                None
                if self.draft_boom_probabilities is None
                else self.draft_boom_probabilities[candidate_mask]
            )
            if candidate_mask is not None
            else self.draft_boom_probabilities,
        )
        with torch.inference_mode():
            scores = policy_network(
                player_features.reshape(-1, player_features.shape[-1]),
                state_features.reshape(-1, state_features.shape[-1]),
                decision_type=decision_type,
            ).reshape(points_for_policy.shape[0], self.player_count)
        scores = scores.to(points.dtype)
        scores = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
            dim=1, keepdim=True
        ).clamp_min(1e-4)
        adjusted = points_for_policy + (0.25 * scores)
        if candidate_mask is None:
            return adjusted
        result = points.clone()
        result[candidate_mask] = adjusted
        return result

    def _policy_adjusted_points_all_teams(
        self,
        points: torch.Tensor,
        policy_network: torch.nn.Module | None,
        decision_type: str,
    ) -> torch.Tensor:
        """Batch shared-policy inference across all teams in one forward."""
        if policy_network is None:
            return points.unsqueeze(1).expand(-1, self.team_count, -1)
        if (
            self.active_team_policy_networks is not None
            or self.active_policy_team_indices is not None
        ):
            return torch.stack(
                [
                    self._policy_adjusted_points(
                        points, policy_network, decision_type, team_index,
                        self._team_policy(team_index),
                    )
                    for team_index in range(self.team_count)
                ],
                dim=1,
            )
        available = self.available if self.available is not None else torch.ones(
            (self.scenario_count, self.player_count), dtype=torch.bool, device=self.device
        )
        rosters = self._require_rosters()
        player_features = []
        state_features = []
        for team_index in range(self.team_count):
            team_player, team_state = _build_player_features(
                projected_points=points.masked_fill(~available, 0.0),
                positions=self.positions,
                available=available,
                rosters=rosters,
                team_index=team_index,
                projection_floors=self.draft_floors,
                projection_medians=self.draft_medians,
                projection_ceilings=self.draft_ceilings,
                boom_probabilities=self.draft_boom_probabilities,
            )
            player_features.append(team_player)
            state_features.append(team_state)
        flat_player = torch.cat(player_features, dim=0)
        flat_state = torch.cat(state_features, dim=0)
        with torch.inference_mode():
            scores = policy_network(
                flat_player.reshape(-1, flat_player.shape[-1]),
                flat_state.reshape(-1, flat_state.shape[-1]),
                decision_type=decision_type,
            ).reshape(self.team_count, self.scenario_count, self.player_count)
        scores = scores.to(points.dtype).permute(1, 0, 2)
        scores = (scores - scores.mean(dim=2, keepdim=True)) / scores.std(
            dim=2, keepdim=True
        ).clamp_min(1e-4)
        return points.unsqueeze(1) + (0.25 * scores)

    def score_week(
        self,
        week: int,
        matchups: torch.Tensor | None = None,
        policy_network: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Score legal projected lineups and update head-to-head standings."""

        self._validate_week(week)
        policy_network = policy_network or self.active_policy_network
        projections = self.weekly_projections[:, week]
        policy_projections = self._policy_adjusted_points_all_teams(
            projections, policy_network, "lineup"
        )
        scores = self._score_batched_lineups(
            policy_projections,
            self.weekly_actual_points[:, week],
            self.positions,
            self._require_rosters(),
        ).scores
        self.points_for += scores
        if matchups is None:
            matchups = self.default_matchups(week)
        self._record_matchups(scores, matchups)
        self.weekly_scores.append(scores)
        if policy_network is not None:
            baseline = score_batched_lineups(
                projections.unsqueeze(1).expand(-1, self.team_count, -1),
                self.weekly_actual_points[:, week],
                self.positions,
                self._require_rosters(),
            ).scores
            lineup_gain = scores - baseline
            if self.active_policy_team_indices is not None:
                team_ids = torch.arange(self.team_count, device=self.device).unsqueeze(0)
                candidate_mask = team_ids == self.active_policy_team_indices.unsqueeze(1)
                lineup_gain = torch.where(
                    candidate_mask, lineup_gain, torch.zeros_like(lineup_gain)
                )
            self.lineup_policy_gains.append(lineup_gain)
        return scores

    def apply_waivers(
        self,
        week: int,
        minimum_improvement: float = 0.5,
        policy_network: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Apply one projection-positive waiver per inverse-standings rank."""

        self._validate_week(week)
        policy_network = policy_network or self.active_policy_network
        batch = torch.arange(self.scenario_count, device=self.device)
        ranking_value = self.wins.to(torch.float32) * 100000.0 + self.points_for
        priority = torch.argsort(ranking_value, dim=1, descending=False)
        accepted_count = torch.zeros(self.scenario_count, dtype=torch.int32, device=self.device)
        counterfactual_gain = torch.zeros(self.scenario_count, device=self.device)
        projections = self.weekly_projections[:, week]
        for rank in range(self.team_count):
            priority = torch.argsort(
                self.wins.to(torch.float32) * 100000.0 + self.points_for,
                dim=1,
                descending=False,
                stable=True,
            )
            team_indices = priority[:, rank]
            policy_projection_by_team = self._policy_adjusted_points_all_teams(
                projections, policy_network, "waiver"
            )
            team_projections = policy_projection_by_team[
                batch, team_indices
            ]
            available_before = self.available.clone()
            had_available_player = available_before.any(dim=1)
            add_scores = team_projections.masked_fill(~available_before, float("-inf"))
            add_players = torch.argsort(add_scores, dim=1, descending=True, stable=True)[:, 0]
            selected_valid = had_available_player & available_before.gather(
                1, add_players.unsqueeze(1)
            ).squeeze(1)
            current_rosters = self._require_rosters()[batch, team_indices]
            current_values = team_projections.gather(1, current_rosters)
            drop_slots = torch.argsort(current_values, dim=1, stable=True)[:, 0]
            candidate_rosters = current_rosters.clone()
            candidate_rosters[batch, drop_slots] = add_players
            current_scores = self._score_batched_lineups(
                team_projections,
                team_projections,
                self.positions,
                current_rosters.unsqueeze(1),
            ).scores[:, 0]
            candidate_scores = self._score_batched_lineups(
                team_projections,
                team_projections,
                self.positions,
                candidate_rosters.unsqueeze(1),
            ).scores[:, 0]
            accepted = selected_valid & (
                candidate_scores - current_scores >= minimum_improvement
            )
            old_players = current_rosters.gather(1, drop_slots.unsqueeze(1)).squeeze(1)
            self._require_rosters()[
                batch[accepted], team_indices[accepted], drop_slots[accepted]
            ] = add_players[accepted]
            self.available[batch[accepted], add_players[accepted]] = False
            self.available[batch[accepted], old_players[accepted]] = True
            accepted_count += accepted.to(torch.int32)
            rank_gain = (candidate_scores - current_scores).clamp_min(0.0) * accepted
            if self.active_policy_team_indices is not None:
                rank_gain = torch.where(
                    team_indices == self.active_policy_team_indices,
                    rank_gain,
                    torch.zeros_like(rank_gain),
                )
            counterfactual_gain += rank_gain

        self.waiver_counts.append(accepted_count)
        if policy_network is not None:
            self.waiver_policy_gains.append(counterfactual_gain)
        return accepted_count

    def apply_trades(
        self,
        week: int,
        top_k: int | None = None,
        minimum_improvement: float = 0.5,
        policy_network: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Apply the best mutually beneficial one-for-one trade per scenario."""

        self._validate_week(week)
        policy_network = policy_network or self.active_policy_network
        if self.team_count < 2:
            return torch.zeros(self.scenario_count, dtype=torch.int32, device=self.device)
        projections = self.weekly_projections[:, week]
        policy_projection_by_team = self._policy_adjusted_points_all_teams(
            projections, policy_network, "trade"
        )
        rosters = self._require_rosters()
        baseline = torch.stack(
            [
                self._score_batched_lineups(
                    policy_projection_by_team[:, team_index],
                    policy_projection_by_team[:, team_index],
                    self.positions,
                    rosters[:, team_index : team_index + 1],
                ).scores[:, 0]
                for team_index in range(self.team_count)
            ],
            dim=1,
        )
        top_k = self.roster_size if top_k is None else min(top_k, self.roster_size)
        if top_k < 1:
            raise ValueError("top_k must be positive when supplied.")

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
        pair_projection = policy_projection_by_team[:, pair_a]
        values_a = pair_projection.gather(2, pair_rosters_a)
        values_b = pair_projection.gather(2, pair_rosters_b)
        top_slots_a = torch.argsort(values_a, dim=2, descending=True, stable=True)[
            :, :, :top_k
        ]
        top_slots_b = torch.argsort(values_b, dim=2, descending=True, stable=True)[
            :, :, :top_k
        ]
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
        repeated_projection = (
            pair_projection.unsqueeze(2).expand(-1, -1, combos, -1).reshape(-1, self.player_count)
        )
        repeated_projection_b = (
            policy_projection_by_team[:, pair_b]
            .unsqueeze(2)
            .expand(-1, -1, combos, -1)
            .reshape(-1, self.player_count)
        )
        score_a = (
            self._score_batched_lineups(
                repeated_projection_b,
                repeated_projection_b,
                self.positions,
                flat_a.unsqueeze(1),
            )
            .scores[:, 0]
            .reshape(self.scenario_count, pair_count, combos)
        )
        score_b = (
            self._score_batched_lineups(
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
        if policy_network is not None:
            trade_gain = best_gain.clamp_min(0.0)
            if self.active_team_policy_networks is not None:
                team_gains = torch.zeros(
                    self.scenario_count,
                    self.team_count,
                    device=self.device,
                )
                team_gains.scatter_add_(1, best_a.unsqueeze(1), trade_gain.unsqueeze(1))
                team_gains.scatter_add_(1, best_b.unsqueeze(1), trade_gain.unsqueeze(1))
                trade_gain = team_gains
            elif self.active_policy_team_indices is not None:
                candidate_trade = (best_a == self.active_policy_team_indices) | (
                    best_b == self.active_policy_team_indices
                )
                trade_gain = torch.where(candidate_trade, trade_gain, torch.zeros_like(trade_gain))
            self.trade_policy_gains.append(trade_gain)
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
        first_seed_rank = (seeds == first_winner.unsqueeze(1)).to(torch.int64).argmax(dim=1)
        second_seed_rank = (seeds == second_winner.unsqueeze(1)).to(torch.int64).argmax(dim=1)
        lower_seed_is_first = first_seed_rank > second_seed_rank
        lowest_remaining = torch.where(lower_seed_is_first, first_winner, second_winner)
        other_remaining = torch.where(lower_seed_is_first, second_winner, first_winner)
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
        projections = self.draft_projections
        policy_network = self.active_policy_network
        policy_projections = self._policy_adjusted_points_all_teams(
            projections, policy_network, "lineup"
        )
        return torch.stack(
            [
                self._score_batched_lineups(
                    policy_projections[:, team_index],
                    self.weekly_actual_points[:, week],
                    self.positions,
                    self._require_rosters()[:, team_index : team_index + 1],
                ).scores[:, 0]
                for team_index in range(self.team_count)
            ],
            dim=1,
        )

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
    policy_network: torch.nn.Module | None = None,
    team_policy_networks: list[torch.nn.Module] | None = None,
    policy_team_indices: torch.Tensor | None = None,
    draft_anchor_weight: float = 0.20,
    fitness_contract: FitnessContract | None = None,
) -> CudaSeasonState:
    """Execute draft, transactions, regular season, and playoffs.

    ``policy_network`` enables policy-controlled action heads for draft,
    lineup, waiver, and trade decisions. The exact CPU transaction replay is
    still the authority for promotion comparisons.
    """

    if fitness_contract is not None:
        expected_contract_digest = fitness_contract.digest()
        if state.contract_digest not in ("", expected_contract_digest):
            raise ValueError("CUDA state fitness contract does not match the requested contract.")
        state.contract_digest = expected_contract_digest
        position_ids = {
            "QB": 0,
            "RB": 1,
            "WR": 2,
            "TE": 3,
            "DST": 4,
            "DEF": 4,
            "D/ST": 4,
            "K": 5,
        }
        state.lineup_position_rules = tuple(
            tuple(position_ids[position] for position in slot.eligible_positions)
            for slot in fitness_contract.lineup_rules
            for _ in range(slot.count)
        )
    if team_policy_networks is not None:
        state.active_team_policy_networks = team_policy_networks
        state.active_policy_network = team_policy_networks[0]
        state.active_policy_team_indices = None
    elif policy_network is not None:
        state.active_team_policy_networks = None
        if policy_team_indices is None:
            policy_team_indices = torch.zeros(
                state.scenario_count,
                dtype=torch.long,
                device=state.device,
            )
        policy_team_indices = policy_team_indices.to(state.device)
        state.active_policy_team_indices = policy_team_indices
        state.active_policy_network = policy_network
    else:
        state.active_team_policy_networks = None
        state.active_policy_team_indices = None
        state.active_policy_network = None
    state.draft(
        policy_network=policy_network
        or (team_policy_networks[0] if team_policy_networks else None),
        policy_team_indices=policy_team_indices,
        draft_anchor_weight=draft_anchor_weight,
    )
    if team_policy_networks is None:
        state.active_policy_network = policy_network
    regular_season_weeks = min(14, state.week_count - 3)
    for week in range(regular_season_weeks):
        if enable_transactions:
            state.apply_trades(week, policy_network=policy_network)
            state.apply_waivers(week, policy_network=policy_network)
        state.score_week(week, policy_network=policy_network)
    state.run_playoffs()
    return state
