# AI Engineering Audit

**Review date:** 2026-08-16  
**Reviewed revision:** `0fee25f`  
**Scope:** projections, manager policies, CPU simulator, CUDA simulator,
evolution, historical data, evaluation, and reproducibility.

## Executive verdict

This project has a strong research foundation: rules are separated from
decisions, the CPU simulator is the behavioral reference, chronological
holdouts are documented, long runs are resumable, and the CUDA path now has
population batching, common random numbers, risk-aware selection, and parity
tests.

The current CUDA flagship is **not yet a complete neural fantasy manager**.
It trains the draft head while CUDA lineups, waivers, and trades use
tensorized baseline heuristics. Its opponents are projection-best baselines,
not independently learning managers. The largest risk is therefore optimizing
a simplified objective and mistaking simulator progress for real manager skill.

The correct order is:

```text
simulation truth and data coverage
    -> valid projections
    -> policy-controlled actions
    -> competitive population training
    -> holdout-gated selection
    -> throughput optimization
```

Do not spend another large overnight budget until the P0 gates below are
measured.

## What is done well

### Separation of concerns

`fantasy_engine/` owns roster legality, scoring, schedules, transactions, and
playoffs. `models/` produces projections or action scores. `agents/` chooses
actions. `evolution/` runs populations and evaluation. This is the right shape
for a decision system because a model cannot be allowed to violate league
rules.

### Historical cutoff discipline

`docs/DATA_AND_REPRODUCIBILITY.md` explicitly distinguishes draft, weekly
lineup, waiver, and trade information boundaries. The weekly feature builder
filters prior weeks for recent and defensive signals. Chronological validation
and unseen holdouts are preferred throughout the repository.

### Useful CPU research instrumentation

The modular CPU trainer records fitness variance, baseline-relative scores,
transaction rewards, population diversity, generation checkpoints, scenario
labels, and transaction ablations. It also supports bounded process
parallelism. This is the right instrumentation for meaningful experiments.

### Serious CUDA safeguards

The current CUDA trainer includes:

- population evaluation through a `torch.func.vmap` ensemble;
- common randomized scenario banks;
- risk-adjusted fitness;
- chronological holdout reporting;
- deterministic Python/PyTorch RNG checkpoints;
- full population resume checkpoints;
- a sequential debugging fallback;
- CUDA-focused parity tests.

The measured smoke result was approximately 1,320 generations/hour batched
versus 357 generations/hour sequential on the same small workload. That is a
measured optimization, not a theoretical claim.

### Transparency

The repository documents simplifications, keeps reports separate from source,
uses Tee-based logs, and records ADRs. The explicit Spec-Driven Development
boundary is appropriate and makes generated changes reviewable.

## Critical problems

### P0-1: CUDA is not full-manager training

Evidence:

- `gpu_sim/policy_draft.py` calls the network with
  `decision_type="draft"`.
- `gpu_sim/full_season.py` uses tensorized projection heuristics for weekly
  lineups, waivers, and trades.
- `models/modular_manager_policy.py` contains draft, lineup, waiver, and trade
  heads, but CUDA does not call the latter three.

Consequence: a CUDA run can improve drafting while leaving the most important
in-season decisions unchanged.

Required fix:

1. Add policy-controlled CUDA lineup actions with legal action masks.
2. Add policy-controlled waiver add/drop actions and transaction logs.
3. Add policy-controlled trade proposal and acceptance actions.
4. Add counterfactual transaction-value attribution to CUDA state.
5. Compare every neural head against its CPU behavioral equivalent before using
   it in flagship training.

### P0-2: CUDA is not true self-play

`evaluate_cuda_policy()` gives one candidate control of one rotating team while
nine teams use projection-best behavior. Population batching makes this faster,
but it does not change the opponent model.

Consequence: the policy learns to exploit a fixed baseline instead of adapting
to strong managers.

Required fix:

- Maintain projection, ADP, random, scarcity, risk-averse, upside, genome, and
  archived neural opponents.
- Evaluate leagues where every team has an explicit policy identity.
- Maintain an Elo or win-rate matrix and sample opponents by difficulty.
- Keep a frozen test league outside mutation and selection.

### P0-3: The player pool has survivorship bias

`fantasy_engine/leakage_safe_player_pool.py` intersects projection-season and
actual-season player keys. `gpu_sim/historical_adapter.py` then truncates that
intersection to the top projected players.

This excludes rookies, returning players, team changes, and identity changes:
exactly the cases a real draft manager must handle.

Required fix:

- Use stable player IDs instead of `(name, position)` as primary identity.
- Build the draft universe from all players known before the draft.
- Represent missing history with masks and priors.
- Add rookie, free-agent, team-change, and injury-status features.
- Test that prior-season-absent players can enter the draft pool.

### P0-4: Projection targets are too weak

The draft projection model uses previous-season points, a two-year average, team
change, and position indicators. The weekly model is richer, but still mainly
regresses point means.

Fantasy decisions need distributions and rankings, not only means.

Required fix:

- Predict quantiles or a distribution: floor, median, ceiling, and boom/bust
  probability.
- Add ranking losses alongside Huber/MAE.
- Calibrate probabilities chronologically.
- Report top-12/top-24 accuracy, rank correlation, calibration error, and lineup
  regret—not only MAE.

### P0-5: CPU and CUDA objectives differ

CPU full-season fitness includes transaction attribution and the full rules
engine. CUDA uses a separate tensorized formula and often offensive-only
lineups. CUDA playoff scoring also intentionally differs from parts of the CPU
projection path.

Consequence: a policy can improve CUDA score while getting worse under the
actual league rules.

Required fix: create one versioned `FitnessContract` containing lineup slots,
K/DST, scoring settings, matchup/tie rules, waiver priority, trade legality,
playoff bracket, reward weights, and transaction attribution. Both CPU and CUDA
must pass the same golden scenario suite.

## Important problems

### Full-weight mutation is inefficient

The CUDA evolutionary loop mutates every neural parameter and crosses parameter
tensors independently. This can destroy useful co-adapted features and spends
evaluations rediscovering basic behavior.

Recommended approach:

- behavior-clone or supervised-pretrain every action head;
- use gradient training for projections and action imitation;
- evolve a smaller strategy adapter, gating vector, or low-rank delta;
- use evolutionary search for exploration and policy optimization for
  fine-tuning;
- retain elites and inject a controlled immigrant fraction.

### The reward is high variance and incomplete

Fitness rewards wins, points, playoff qualification, playoff wins, and
championships. It does not sufficiently reward lineup efficiency, replacement
value, opponent difficulty, or transaction value in the CUDA path.

Use a multi-objective report:

```text
baseline-relative weekly wins
+ lineup efficiency
+ replacement-value gain
+ transaction value
+ playoff performance
- season variance
- invalid actions
- exploitability against the opponent league
```

Do not collapse this into one number until each component is validated.

### Holdout is audited, not gated

The CUDA script reports 2025 after training, but selection can still choose a
policy that fails the holdout. Promotion must require improvement over the
initial model and simple baselines on unseen seasons with uncertainty intervals.

Promotion gate:

- beat projection/ADP baselines on at least two unseen seasons;
- no significant regression in weekly wins or lineup efficiency;
- bootstrap or confidence interval reported;
- transaction ablation does not explain away the result;
- replay and simulator parity gates pass.

### Feature lineage is incomplete

Feature names exist, but checkpoints do not fully encode schema version, source
hashes, player-ID mapping, scoring-rule hash, and normalization provenance.

Add a `FeatureManifest` containing feature names/order, source checksums,
decision cutoff, scoring settings, identity-map version, code revision, and
normalization statistics. Reject incompatible checkpoints.

### Special teams and irregular rules remain incomplete

The CUDA path is frequently run with offensive-only scoring while the target
league includes K and DST. Until K/DST, ties, postponed games, player status,
and league-specific settings pass parity tests, results must be labeled
experimental rather than flagship-quality.

## Recommended target architecture

```text
data registry + stable player IDs + feature manifests
                         |
             projection ensemble with quantiles
                         |
              exact league environment
                         |
       shared encoder + masked action heads
           draft | lineup | waiver | trade | value
                         |
      opponent archive + population league + Elo
                         |
       replay + counterfactual transaction value
                         |
 supervised pretraining -> offline fine-tuning
                         |
      self-play/PBT/ES over low-rank adapters
                         |
 chronological validation -> frozen holdout gate
                         |
              CUDA batched production trainer
```

The local language model should remain an explanation and research assistant.
It should not directly choose or submit transactions; structured policy outputs
must pass the rules engine.

## Zero-waste training protocol

Before every long run:

1. Run the golden CPU/CUDA parity suite, including K/DST and transactions.
2. Run a 10-minute benchmark and record GPH, VRAM, GPU utilization, and peak
   memory.
3. Run an ablation matrix: projection, ADP, current policy, genome, neural,
   and hybrid transaction arms.
4. Require validation improvement over baseline before overnight compute.
5. Start with a full population checkpoint, Tee log, report, fixed seed,
   holdout season, and declared promotion gate.

Every overnight run should answer one question, such as:

- Does policy-controlled lineup scoring improve weekly wins?
- Does an opponent archive reduce exploitability?
- Do quantile projections improve lineup regret?
- Do low-rank mutations improve holdout results per GPU-hour?

If a run cannot answer a question with a predeclared metric, it is not a useful
training run.

## Prioritized roadmap

### P0 — correctness before scale

1. Stable IDs and a non-survivorship draft universe.
2. Versioned shared fitness/rules contract.
3. CPU/CUDA golden parity for full lineups, K/DST, waivers, trades, and
   playoffs.
4. Holdout promotion gate and baseline-relative reports.

### P1 — complete manager intelligence

1. Policy-controlled CUDA lineup, waiver, and trade heads.
2. True multi-policy opponent league with archive/Elo.
3. Quantile/ranking projection models with injury, role, and matchup features.
4. Counterfactual transaction attribution and action masks.

### P2 — efficiency and scale

1. Low-rank policy adapters instead of full-weight mutation.
2. Replay-based offline pretraining and policy fine-tuning.
3. Population batching across policies, seasons, and opponent leagues.
4. CUDA graphs/fused kernels only after profiler evidence and parity tests.

## Bottom line

The project is strong in organization, reproducibility, simulation
instrumentation, and measured CUDA optimization. The weakness is that training
throughput has advanced faster than behavioral fidelity: the CUDA flagship
currently learns draft behavior in a simplified opponent and rules environment.

The next meaningful improvement is not another larger overnight run. It is a
parity-and-completeness milestone: train all four decision heads against a real
opponent population, then promote models only when they beat baselines on
unseen seasons under the exact league contract.
