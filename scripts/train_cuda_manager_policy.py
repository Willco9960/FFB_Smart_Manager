"""Train the modular manager policy with the CUDA full-season simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

from evolution.preflight import run_training_preflight
from evolution.promotion_gate import evaluate_promotion_gate
from fantasy_engine.historical_loader import get_player_stats_raw_path
from gpu_sim.historical_adapter import create_historical_cuda_inputs
from gpu_sim.policy_training import (
    CudaGenerationMetrics,
    save_cuda_policy_checkpoint,
    save_cuda_training_state,
    summarize_cuda_throughput,
    train_cuda_policy_population,
    validate_cuda_training_state_manifest,
)
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    load_modular_policy_network,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--selection", type=int, default=4)
    parser.add_argument("--scenario-repeats", type=int, default=8)
    parser.add_argument("--projection-noise", type=float, default=0.015)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--mutation-strength", type=float, default=0.02)
    parser.add_argument("--final-mutation-strength", type=float, default=0.005)
    parser.add_argument("--draft-anchor-weight", type=float, default=0.20)
    parser.add_argument("--risk-penalty", type=float, default=0.10)
    parser.add_argument(
        "--scenario-refresh-generations",
        type=int,
        default=0,
        help="Refresh common-random-number scenarios periodically; 0 keeps one bank.",
    )
    parser.add_argument(
        "--season-subsample-size",
        type=int,
        default=0,
        help="Deterministically rotate this many training seasons per generation; 0 uses all.",
    )
    parser.add_argument(
        "--season-replay-interval",
        type=int,
        default=0,
        help="When using a leading extra season, replay all seasons every N generations.",
    )
    parser.add_argument(
        "--full-policy-mutation",
        action="store_true",
        help="Evolve shared encoders as well as decision/value heads.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden width for a new/from-scratch policy; existing checkpoints define theirs.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore --initial-policy and create a new policy using --hidden-size or 128.",
    )
    parser.add_argument(
        "--compile-policy",
        action="store_true",
        help="Use torch.compile reduce-overhead for repeated CUDA policy forwards.",
    )
    parser.add_argument(
        "--disable-population-batching",
        action="store_true",
        help="Use sequential policy evaluation for debugging or CPU parity.",
    )
    parser.add_argument(
        "--batched-policy-heads",
        action="store_true",
        help=(
            "Use the parity-tested flattened CUDA population route for all manager heads. "
            "Without this flag, exact per-policy head evaluation remains the default."
        ),
    )
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=0,
        help="One chronological unseen season; use 0 with --holdout-seasons or to disable.",
    )
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        nargs="+",
        default=(2024, 2025),
        help="Multiple chronological unseen seasons evaluated independently.",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Evaluate candidates against a frozen population/opponent archive.",
    )
    parser.add_argument(
        "--self-play-interval",
        type=int,
        default=1,
        help="Run routed self-play every N generations; 1 means every generation.",
    )
    parser.add_argument(
        "--opponent-archive-size",
        type=int,
        default=64,
        help="Maximum frozen self-play opponent archive size.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        default=Path("data/models/modular_manager_policy.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/models/cuda_manager_policy.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/cuda_manager_training_state.pt"),
        help="Full-population checkpoint written after every generation.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume the population checkpoint written by --checkpoint.",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/cuda_manager_training.json"))
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=None,
        help="Historical CPU/CUDA parity report used for promotion-readiness evidence.",
    )
    parser.add_argument(
        "--require-promotion-ready",
        action="store_true",
        help="Refuse to start unless the supplied parity report proves exact full-rule parity.",
    )
    parser.add_argument(
        "--multi-seed-report",
        type=Path,
        default=None,
        help="Frozen multi-seed holdout report used as an additional promotion gate.",
    )
    parser.add_argument(
        "--require-multi-seed-promotion",
        action="store_true",
        help="Block promotion unless every evaluated seed beats the initial policy.",
    )
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Allow a pre-manifest checkpoint with an explicit safety warning.",
    )
    parser.add_argument(
        "--disable-transactions",
        action="store_true",
        help="Disable CUDA waiver/trade stages for a draft-only ablation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch/CUDA algorithms and disable TF32 for qualification runs.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-capable PyTorch device is available.")
    return torch.device(requested)


def resolve_holdout_seasons(
    holdout_season: int,
    holdout_seasons: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    """Normalize one or more chronological holdout seasons."""
    if holdout_seasons is not None:
        if holdout_season != 0:
            raise ValueError("Specify either --holdout-season or --holdout-seasons, not both.")
        normalized = tuple(holdout_seasons)
    elif holdout_season == 0:
        normalized = ()
    else:
        normalized = (holdout_season,)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Holdout seasons must be unique.")
    if any(season < 0 for season in normalized):
        raise ValueError("Holdout seasons must be zero or positive.")
    return normalized


def validate_player_count(players: int, team_count: int = 10, roster_size: int = 16) -> None:
    """Reject player pools that cannot fill the declared league rosters."""
    minimum_players = team_count * roster_size
    if players < minimum_players:
        raise ValueError(
            f"players must be at least {minimum_players} for {team_count} "
            f"teams with {roster_size}-player rosters."
        )


def validate_opponent_archive_size(size: int) -> None:
    if size < 1:
        raise ValueError("opponent archive size must be positive.")


def validate_season_window(
    start_season: int,
    end_season: int,
    holdout_season: int | tuple[int, ...],
) -> None:
    """Reject training windows that make any declared holdout invalid."""
    if end_season < start_season:
        raise ValueError("end-season must be >= start-season")
    holdouts = (
        (holdout_season,)
        if isinstance(holdout_season, int)
        else tuple(holdout_season)
    )
    if any(season < 0 for season in holdouts):
        raise ValueError("Holdout season must be zero or a positive season.")
    if any(season > 0 and season <= end_season for season in holdouts):
        raise ValueError("Holdout season must be after every training season.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, object]:
    """Record revision and dirty state without making Git a hard dependency."""
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "-C", str(repository), "diff", "HEAD", "--binary"],
            capture_output=True,
            check=True,
        ).stdout + dirty.encode("utf-8")
        return {
            "revision": revision,
            "dirty": bool(dirty),
            "working_tree_diff_sha256": hashlib.sha256(dirty_output).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "working_tree_diff_sha256": None}


def load_parity_evidence(
    path: Path | None,
    *,
    transactions_enabled: bool,
    require_promotion_ready: bool,
    expected_players: int | None = None,
) -> dict[str, object]:
    """Load and validate a historical parity artifact without rerunning it."""
    if path is None:
        evidence = {
            "status": "missing",
            "promotion_ready": False,
            "reason": "no parity report supplied",
        }
        if require_promotion_ready:
            raise ValueError("--require-promotion-ready requires --parity-report.")
        return evidence
    if not path.exists():
        raise FileNotFoundError(f"Parity report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = all(
        bool(payload.get(key))
        for key in (
            "exact_standings_match",
            "exact_champion_match",
            "exact_weekly_score_match",
        )
    )
    mode_matches = bool(payload.get("transactions")) == transactions_enabled
    player_count_matches = expected_players is None or payload.get("players") == expected_players
    transaction_trace_exact = bool(
        payload.get("transaction_actions_exact")
        and payload.get("transaction_state_exact")
        and payload.get("transaction_reward_exact")
    )
    promotion_ready = (
        exact
        and mode_matches
        and player_count_matches
        and payload.get("max_weekly_score_abs_delta") == 0.0
        and (not transactions_enabled or transaction_trace_exact)
    )
    evidence = {
        "status": "verified" if promotion_ready else "failed",
        "promotion_ready": promotion_ready,
        "path": str(path),
        "sha256": _sha256_file(path),
        "transactions_enabled": payload.get("transactions"),
        "mode_matches_training": mode_matches,
        "player_count": payload.get("players"),
        "player_count_matches_training": player_count_matches,
        "exact_standings_match": bool(payload.get("exact_standings_match")),
        "exact_champion_match": bool(payload.get("exact_champion_match")),
        "exact_weekly_score_match": bool(payload.get("exact_weekly_score_match")),
        "max_weekly_score_abs_delta": payload.get("max_weekly_score_abs_delta"),
        "transaction_actions_exact": bool(payload.get("transaction_actions_exact")),
        "transaction_state_exact": bool(payload.get("transaction_state_exact")),
        "transaction_reward_exact": bool(payload.get("transaction_reward_exact")),
        "transaction_trace_exact": transaction_trace_exact,
    }
    if require_promotion_ready and not promotion_ready:
        raise ValueError(
            "Parity report does not prove exact full-rule parity for the training transaction mode."
        )
    return evidence


def load_multi_seed_evidence(
    path: Path | None,
    *,
    require_promotion_ready: bool,
    expected_holdout_seasons: tuple[int, ...] | None = None,
    expected_transactions_enabled: bool | None = None,
    expected_initial_policy_sha256: str | None = None,
) -> dict[str, object]:
    """Load and independently validate independent-seed holdout evidence."""
    if path is None:
        if require_promotion_ready:
            raise ValueError("--require-multi-seed-promotion requires --multi-seed-report.")
        return {"provided": False, "promotion_ready_multi_seed": False}
    if not path.exists():
        raise FileNotFoundError(f"Multi-seed report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_initial_policy_sha256 is not None:
        actual_initial_policy_sha256 = payload.get("initial_policy_sha256")
        if actual_initial_policy_sha256 != expected_initial_policy_sha256:
            raise ValueError(
                "Multi-seed report initial-policy hash does not match the active initial policy."
            )
    claimed_ready = bool(payload.get("promotion_ready_multi_seed", False))
    seeds = payload.get("seeds", [])
    by_season = payload.get("by_season", {})
    report_seasons = tuple(payload.get("holdout_seasons", ()))
    expected_seasons = tuple(
        expected_holdout_seasons
        or report_seasons
        or tuple(int(season) for season in by_season)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Multi-seed report must contain unique independent seeds.")
    if not by_season:
        raise ValueError("Multi-seed report must contain at least one holdout season.")
    if expected_holdout_seasons is not None and set(report_seasons) != set(expected_seasons):
        raise ValueError("Multi-seed report holdout seasons do not match training expectations.")
    if (
        expected_transactions_enabled is not None
        and payload.get("transactions") != expected_transactions_enabled
    ):
        raise ValueError("Multi-seed report transaction mode does not match training expectations.")
    incomplete = [
        season for season in expected_seasons
        if by_season.get(str(season), {}).get("seed_count") != len(seeds)
    ]
    if incomplete:
        raise ValueError(
            "Multi-seed report seed count is incomplete for: " + ", ".join(map(str, incomplete))
        )
    rows = payload.get("rows", [])
    expected_pairs = {(season, seed) for season in expected_seasons for seed in seeds}
    actual_pairs = {(row.get("season"), row.get("seed")) for row in rows}
    if actual_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError(
            "Multi-seed report rows do not cover every holdout/seed pair exactly once."
        )
    for row in rows:
        for key in ("delta_vs_initial", "risk_adjusted_delta_vs_initial"):
            if not math.isfinite(float(row.get(key, float("nan")))):
                raise ValueError(f"Multi-seed report row has invalid {key}.")
    recomputed_ready = bool(rows) and all(
        float(row["delta_vs_initial"]) > 0
        and float(row["risk_adjusted_delta_vs_initial"]) > 0
        for row in rows
    )
    if claimed_ready != recomputed_ready:
        raise ValueError("Multi-seed report readiness claim does not match its rows.")
    evidence = {
        "provided": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "promotion_ready_multi_seed": recomputed_ready,
        "by_season": by_season,
        "seeds": seeds,
        "holdout_seasons": list(expected_seasons),
        "transactions": payload.get("transactions"),
    }
    if require_promotion_ready and not recomputed_ready:
        raise ValueError("Multi-seed report does not prove all-seed promotion readiness.")
    return evidence


def build_run_manifest(
    args: argparse.Namespace,
    initial_policy,
    preflight_summary: dict[str, object],
    parity_evidence: dict[str, object],
) -> dict:
    """Capture data and search identity needed for an auditable resume."""
    data_seasons = set(range(args.start_season - 1, args.end_season + 1))
    for holdout_season in args.holdout_seasons:
        data_seasons.update((holdout_season - 1, holdout_season))
    data_files = {}
    for season in sorted(data_seasons):
        path = get_player_stats_raw_path(season)
        if not path.exists():
            raise FileNotFoundError(f"Historical input missing for season {season}: {path}")
        data_files[str(season)] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {
        "schema_version": 2,
        "code_identity": _git_identity(),
        "initial_policy": (
            {
                "path": str(args.initial_policy),
                "size": args.initial_policy.stat().st_size,
                "sha256": _sha256_file(args.initial_policy),
            }
            if args.initial_policy.exists() and not args.from_scratch
            else None
        ),
        "preflight": preflight_summary,
        "parity_evidence": parity_evidence,
        "training_seasons": list(range(args.start_season, args.end_season + 1)),
        "holdout_seasons": list(args.holdout_seasons),
        "holdout_season": args.holdout_seasons[0] if len(args.holdout_seasons) == 1 else 0,
        "data_files": data_files,
        "population": args.population,
        "selection": args.selection,
        "scenario_repeats": args.scenario_repeats,
        "projection_noise": args.projection_noise,
        "players": args.players,
        "mutation_strength": args.mutation_strength,
        "final_mutation_strength": args.final_mutation_strength,
        "draft_anchor_weight": args.draft_anchor_weight,
        "risk_penalty": args.risk_penalty,
        "transactions_enabled": not args.disable_transactions,
        "batched_policy_heads": args.batched_policy_heads,
        "population_batching": not args.disable_population_batching,
        "compile_policy": args.compile_policy,
        "scenario_refresh_generations": args.scenario_refresh_generations,
        "season_subsample_size": args.season_subsample_size,
        "season_replay_interval": args.season_replay_interval,
        "full_policy_mutation": args.full_policy_mutation,
        "self_play": args.self_play,
        "opponent_archive_size": args.opponent_archive_size,
        "self_play_interval": args.self_play_interval,
        "hidden_size": initial_policy.hidden_size,
        "player_feature_count": initial_policy.player_feature_count,
        "state_feature_count": initial_policy.state_feature_count,
        "parameter_count": sum(
            parameter.numel() for parameter in initial_policy.parameters()
        ),
        "seed": args.seed,
        "deterministic_effective": bool(args.deterministic or args.require_promotion_ready),
        "tf32_enabled": not bool(args.deterministic or args.require_promotion_ready),
    }


def load_historical_states(args: argparse.Namespace, device: torch.device):
    seasons = list(range(args.start_season, args.end_season + 1))

    def load(season: int):
        return create_historical_cuda_inputs(
            season=season,
            players=args.players,
            device=device,
        ).state

    workers = max(1, min(args.loader_workers, len(seasons)))
    if workers == 1:
        return [load(season) for season in seasons]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(load, seasons))


def load_holdout_states(args: argparse.Namespace, device: torch.device):
    return [
        (
            season,
            create_historical_cuda_inputs(
                season=season,
                players=args.players,
                device=device,
            ).state,
        )
        for season in args.holdout_seasons
    ]


def evaluate_cuda_promotion_readiness(
    *,
    parity_ready: bool,
    multi_seed_ready: bool,
    transactions_disabled: bool,
    self_play: bool,
    self_play_interval: int,
    holdouts: list[dict[str, object]] | None,
) -> dict[str, object]:
    """Return a conservative promotion decision from completed evidence."""
    reasons: list[str] = []
    if not parity_ready:
        reasons.append("historical CPU/CUDA parity is not exact")
    if not multi_seed_ready:
        reasons.append("independent multi-seed holdout gate is not satisfied")
    if transactions_disabled:
        reasons.append("transactions are disabled; this is a draft-only ablation")
    if not self_play:
        reasons.append("competitive self-play evaluation is disabled")
    if self_play_interval != 1:
        reasons.append("self-play is not evaluated every generation")
    completed_holdouts = holdouts or []
    if len(completed_holdouts) < 2:
        reasons.append("at least two unseen holdout seasons are required")
    if completed_holdouts and not all(
        float(item.get("candidate_delta_vs_initial", 0.0)) > 0
        and float(item.get("candidate_risk_adjusted_delta_vs_initial", 0.0)) > 0
        for item in completed_holdouts
    ):
        reasons.append("candidate does not improve raw and risk-adjusted fitness on every holdout")
    if completed_holdouts and not all(
        "candidate" in item
        and "initial_policy" in item
        and "wins" in item["candidate"]
        and "wins" in item["initial_policy"]
        for item in completed_holdouts
    ):
        reasons.append("paired promotion evidence is missing candidate and baseline wins")
    elif completed_holdouts and not reasons:
        paired_gate = evaluate_promotion_gate(
            candidate_fitness=[
                float(item["candidate_delta_vs_initial"])
                for item in completed_holdouts
            ],
            baseline_fitness=[0.0] * len(completed_holdouts),
            candidate_wins=[float(item["candidate"]["wins"]) for item in completed_holdouts],
            baseline_wins=[float(item["initial_policy"]["wins"]) for item in completed_holdouts],
        )
        if not paired_gate.promoted:
            reasons.extend(paired_gate.reasons)
    if not completed_holdouts:
        reasons.append("holdout evaluation is pending")
    return {"promotion_ready": not reasons, "reasons": reasons}


def evaluation_to_dict(evaluation) -> dict[str, float]:
    return {
        "fitness": evaluation.fitness,
        "fitness_stddev": evaluation.fitness_stddev,
        "risk_adjusted_fitness": evaluation.risk_adjusted_fitness,
        "wins": evaluation.wins,
        "points_for": evaluation.points_for,
        "playoff_rate": evaluation.playoff_rate,
        "championship_rate": evaluation.championship_rate,
        "transaction_reward": evaluation.transaction_reward,
        "lineup_efficiency": evaluation.lineup_efficiency,
        "elapsed_seconds": evaluation.elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    args.holdout_seasons = resolve_holdout_seasons(
        args.holdout_season,
        args.holdout_seasons,
    )
    validate_season_window(args.start_season, args.end_season, args.holdout_seasons)
    validate_player_count(args.players)
    validate_opponent_archive_size(args.opponent_archive_size)
    if args.self_play_interval < 1:
        raise ValueError("self-play interval must be positive.")
    if args.season_subsample_size < 0:
        raise ValueError("season subsample size cannot be negative.")
    if args.season_replay_interval < 0:
        raise ValueError("season replay interval cannot be negative.")
    if (
        args.season_subsample_size
        and args.season_subsample_size > args.end_season - args.start_season + 1
    ):
        raise ValueError("season subsample size cannot exceed the training-season count.")
    device = resolve_device(args.device)
    deterministic = bool(args.deterministic or args.require_promotion_ready)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    parity_evidence = load_parity_evidence(
        args.parity_report,
        transactions_enabled=not args.disable_transactions,
        require_promotion_ready=args.require_promotion_ready,
        expected_players=args.players,
    )
    multi_seed_evidence = load_multi_seed_evidence(
        args.multi_seed_report,
        require_promotion_ready=args.require_multi_seed_promotion,
        expected_holdout_seasons=args.holdout_seasons,
        expected_transactions_enabled=not args.disable_transactions,
        expected_initial_policy_sha256=(
            _sha256_file(args.initial_policy) if args.initial_policy.exists() else None
        ),
    )
    training_seasons = tuple(range(args.start_season, args.end_season + 1))
    print("Running mandatory historical preflight...", flush=True)
    preflight = run_training_preflight(training_seasons, device="cpu")
    preflight_summary = {
        "approved": preflight.approved,
        "data_ready": preflight.data_ready,
        "policy_heads_ready": preflight.policy_heads_ready,
        "contract_ready": preflight.contract_ready,
        "training_seasons": list(training_seasons),
    }
    required_preflight_flags = (
        "approved",
        "data_ready",
        "policy_heads_ready",
        "contract_ready",
    )
    if not all(preflight_summary[key] for key in required_preflight_flags):
        raise RuntimeError(f"CUDA training preflight failed: {preflight_summary}")

    if args.from_scratch:
        initial_policy = ModularManagerPolicyNetwork(hidden_size=args.hidden_size or 128)
    elif args.initial_policy.exists():
        initial_policy = load_modular_policy_network(args.initial_policy)
        if args.hidden_size is not None and args.hidden_size != initial_policy.hidden_size:
            raise ValueError(
                "--hidden-size cannot differ from the existing initial-policy checkpoint; "
                "use --from-scratch for a new architecture."
            )
    else:
        initial_policy = ModularManagerPolicyNetwork(hidden_size=args.hidden_size or 64)
    initial_policy = initial_policy.to(device)
    run_manifest = build_run_manifest(
        args,
        initial_policy,
        preflight_summary,
        parity_evidence,
    )
    print(f"CUDA manager training device: {device}", flush=True)
    print(
        f"Seasons: {args.start_season}-{args.end_season} | population={args.population} "
        f"generations={args.generations} repeats={args.scenario_repeats}",
        flush=True,
    )
    print(
        "Population routing: "
        + (
            "flattened CUDA policy heads"
            if args.batched_policy_heads
            else "exact per-policy heads"
        ),
        flush=True,
    )
    print("Loading historical CUDA states...", flush=True)
    states = load_historical_states(args, device)
    holdout_states = load_holdout_states(args, device)
    resume_state = None
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_state.get("run_manifest") is None and args.allow_legacy_resume:
            warnings.warn(
                "Resuming a pre-manifest checkpoint; data and search identity "
                "cannot be verified.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            validate_cuda_training_state_manifest(resume_state, run_manifest)
        print(
            f"Resuming after generation {resume_state['generation']} from {args.resume}",
            flush=True,
        )
    configuration = vars(args).copy()
    configuration["holdout_seasons"] = list(args.holdout_seasons)
    configuration["deterministic_effective"] = deterministic
    configuration["tf32_enabled"] = (
        bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False
    )
    initial_promotion = evaluate_cuda_promotion_readiness(
        parity_ready=bool(parity_evidence["promotion_ready"]),
        multi_seed_ready=bool(multi_seed_evidence["promotion_ready_multi_seed"]),
        transactions_disabled=args.disable_transactions,
        self_play=args.self_play,
        self_play_interval=args.self_play_interval,
        holdouts=None,
    )
    promotion_reasons = list(initial_promotion["reasons"])
    promotion_ready = bool(initial_promotion["promotion_ready"])
    report = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "configuration": configuration | {"device": str(device)},
        "preflight": preflight.to_dict(),
        "parity_evidence": parity_evidence,
        "multi_seed_evidence": multi_seed_evidence,
        "promotion_readiness": {
            "status": "eligible" if promotion_ready else "blocked",
            "reasons": promotion_reasons,
        },
        "run_manifest": run_manifest,
        "generations": [],
        "holdout": None,
        "resumed_from": str(args.resume) if args.resume is not None else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def on_generation(
        metrics: CudaGenerationMetrics,
        best_policy: ModularManagerPolicyNetwork,
    ) -> None:
        checkpoint = save_cuda_policy_checkpoint(
            best_policy,
            args.output,
            [*metrics_history, metrics],
            run_manifest=run_manifest,
        )
        record = metrics.to_dict() | {"checkpoint": str(checkpoint)}
        metrics_history.append(metrics)
        report["generations"].append(record)
        report["updated_at"] = datetime.now().astimezone().isoformat()
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(
            f"[CUDA generation {metrics.generation}/{metrics.generations}] "
            f"best={metrics.best_fitness:.2f} avg={metrics.average_fitness:.2f} "
            f"risk_adj={metrics.best_risk_adjusted_fitness:.2f} "
            f"std={metrics.best_fitness_stddev:.2f} "
            f"wins={metrics.best_wins:.2f} playoffs={metrics.best_playoff_rate:.1%} "
            f"championships={metrics.best_championship_rate:.1%} "
            f"GPH={metrics.generations_per_hour:.2f} "
            f"elapsed={metrics.elapsed_seconds / 3600:.2f}h",
            flush=True,
        )

    metrics_history: list[CudaGenerationMetrics] = []
    if resume_state is not None:
        metrics_history = [CudaGenerationMetrics(**item) for item in resume_state["metrics"]]
    def save_training_checkpoint(
        generation,
        population,
        best_policy,
        metrics,
        rng,
        opponent_archive,
    ):
        save_cuda_training_state(
            args.checkpoint,
            generation=generation,
            population=population,
            best_policy=best_policy,
            metrics=metrics,
            rng_state=rng.getstate(),
            run_manifest=run_manifest,
            opponent_archive=opponent_archive,
        )

    best_policy, metrics_history = train_cuda_policy_population(
        initial_policy=initial_policy,
        historical_states=states,
        population_size=args.population,
        generations=args.generations,
        selection_count=args.selection,
        mutation_strength=args.mutation_strength,
        final_mutation_strength=args.final_mutation_strength,
        scenario_repeats=args.scenario_repeats,
        projection_noise=args.projection_noise,
        enable_transactions=not args.disable_transactions,
        seed=args.seed,
        draft_anchor_weight=args.draft_anchor_weight,
        risk_penalty=args.risk_penalty,
        compile_policy=args.compile_policy,
        batch_population=not args.disable_population_batching,
        exact_policy_head_parity=not args.batched_policy_heads,
        adapter_only=not args.full_policy_mutation,
        self_play=args.self_play,
        opponent_archive_size=args.opponent_archive_size,
        self_play_interval=args.self_play_interval,
        scenario_refresh_generations=args.scenario_refresh_generations,
        season_subsample_size=args.season_subsample_size,
        season_replay_interval=args.season_replay_interval,
        require_complete_fitness_contract=args.require_promotion_ready,
        resume_state=resume_state,
        generation_callback=on_generation,
        checkpoint_callback=save_training_checkpoint,
        run_manifest=run_manifest,
    )
    # The callback writes the full resumable population checkpoint each
    # generation; this final write records the terminal best-policy artifact.
    save_cuda_policy_checkpoint(
        best_policy,
        args.output,
        metrics_history,
        run_manifest=run_manifest,
    )
    report["status"] = "complete"
    report["completed_at"] = datetime.now().astimezone().isoformat()
    report["generations"] = [metric.to_dict() for metric in metrics_history]
    report["output"] = str(args.output)
    best_metric = max(metrics_history, key=lambda metric: metric.best_risk_adjusted_fitness)
    final_metric = metrics_history[-1]
    report["optimization_summary"] = {
        "best_risk_adjusted_generation": best_metric.generation,
        "best_risk_adjusted_fitness": best_metric.best_risk_adjusted_fitness,
        "final_generation": final_metric.generation,
        "final_risk_adjusted_fitness": final_metric.best_risk_adjusted_fitness,
        "final_minus_best_risk_adjusted_fitness": (
            final_metric.best_risk_adjusted_fitness - best_metric.best_risk_adjusted_fitness
        ),
    }
    if holdout_states:
        from gpu_sim.policy_training import evaluate_cuda_policy

        holdout_reports = []
        for holdout_index, (season, holdout_state) in enumerate(holdout_states):
            evaluation_kwargs = {
                "scenario_repeats": max(args.scenario_repeats, 8),
                "projection_noise": args.projection_noise,
                "enable_transactions": not args.disable_transactions,
                "seed": args.seed + 900_000 + holdout_index * 10_000,
                "draft_anchor_weight": args.draft_anchor_weight,
                "risk_penalty": args.risk_penalty,
                "compile_policy": args.compile_policy,
            }
            candidate = evaluate_cuda_policy(best_policy, [holdout_state], **evaluation_kwargs)
            initial = evaluate_cuda_policy(initial_policy, [holdout_state], **evaluation_kwargs)
            projection_baseline = evaluate_cuda_policy(None, [holdout_state], **evaluation_kwargs)
            holdout_reports.append(
                {
                    "season": season,
                    "candidate": evaluation_to_dict(candidate),
                    "initial_policy": evaluation_to_dict(initial),
                    "projection_baseline": evaluation_to_dict(projection_baseline),
                    "candidate_delta_vs_initial": candidate.fitness - initial.fitness,
                    "candidate_delta_vs_projection": (
                        candidate.fitness - projection_baseline.fitness
                    ),
                    "candidate_risk_adjusted_delta_vs_initial": (
                        candidate.risk_adjusted_fitness - initial.risk_adjusted_fitness
                    ),
                    "candidate_risk_adjusted_delta_vs_projection": (
                        candidate.risk_adjusted_fitness
                        - projection_baseline.risk_adjusted_fitness
                    ),
                }
            )
            print(
                f"Holdout {season}: candidate={candidate.fitness:.2f} "
                f"initial={initial.fitness:.2f} projection={projection_baseline.fitness:.2f} "
                f"candidate_std={candidate.fitness_stddev:.2f}",
                flush=True,
            )
        report["holdouts"] = holdout_reports
        report["holdout"] = holdout_reports[0] if len(holdout_reports) == 1 else None
        report["promotion_evidence"] = {
            "minimum_unseen_seasons": 2,
            "unseen_seasons": [item["season"] for item in holdout_reports],
            "eligible_for_two_season_review": len(holdout_reports) >= 2,
            "opponent_mode": (
                "frozen_self_play_archive"
                if args.self_play
                else "projection_baseline_only"
            ),
            "competitive_opponent_evaluation_complete": args.self_play,
            "candidate_beats_initial_all_seasons": all(
                item["candidate_delta_vs_initial"] > 0 for item in holdout_reports
            ),
            "candidate_beats_projection_all_seasons": all(
                item["candidate_delta_vs_projection"] > 0 for item in holdout_reports
            ),
            "candidate_risk_adjusted_beats_initial_all_seasons": all(
                item["candidate_risk_adjusted_delta_vs_initial"] > 0
                for item in holdout_reports
            ),
            "candidate_risk_adjusted_beats_projection_all_seasons": all(
                item["candidate_risk_adjusted_delta_vs_projection"] > 0
                for item in holdout_reports
            ),
        }
        final_promotion = evaluate_cuda_promotion_readiness(
            parity_ready=bool(parity_evidence["promotion_ready"]),
            multi_seed_ready=bool(multi_seed_evidence["promotion_ready_multi_seed"]),
            transactions_disabled=args.disable_transactions,
            self_play=args.self_play,
            self_play_interval=args.self_play_interval,
            holdouts=holdout_reports,
        )
        report["promotion_readiness"] = {
            "status": "eligible" if final_promotion["promotion_ready"] else "blocked",
            "reasons": final_promotion["reasons"],
        }
    if metrics_history:
        training_seasons_count = args.end_season - args.start_season + 1
        report["throughput_summary"] = summarize_cuda_throughput(
            metrics_history,
            population=args.population,
            training_seasons=training_seasons_count,
            scenario_repeats=args.scenario_repeats,
            warmup_generations=min(5, len(metrics_history) - 1),
        )
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"CUDA manager policy saved to: {args.output}", flush=True)
    print(f"Training report saved to: {args.report}", flush=True)


if __name__ == "__main__":
    main()
