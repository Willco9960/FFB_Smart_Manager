[CmdletBinding()]
param(
    [int]$StartSeason = 2001,
    [int]$EndSeason = 2023,
    [int]$Population = 16,
    [int]$Generations = 400,
    [int]$ScenarioRepeats = 4,
    [int]$ScenarioRefreshGenerations = 25,
    [int]$SeasonSubsampleSize = 0,
    [int]$SeasonReplayInterval = 0,
    [int]$SelfPlayInterval = 1,
    [int]$Players = 256,
    [switch]$SelfPlay,
    [switch]$FullPolicyMutation,
    [switch]$BatchedPolicyHeads,
    [switch]$Deterministic,
    [switch]$CompilePolicy,
    [ValidateSet("promotion", "exploratory-draft")]
    [string]$RunMode = "promotion",
    [string]$RunId = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [int]$OpponentArchiveSize = 64,
    [string]$MultiSeedReport = "reports\multi_seed_final_architecture_calibration.json",
    [string]$ParityReport = "reports\parity_transactions_pipeline_rewardfixed_20260820.json",
    [int]$MaxGpuUtilization = 25,
    [int]$MaxGpuMemoryMiB = 3000,
    [int]$MaxGpuTemperatureC = 80
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Stop-ProcessTree {
    param([int]$ProcessId)
    if ($ProcessId -le 0) {
        return
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        & taskkill.exe /PID $ProcessId /T /F | Out-Null
    }
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}

if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    throw "RunId must contain only letters, numbers, underscore, and hyphen."
}

if ($RunMode -eq "promotion" -and -not $SelfPlay.IsPresent) {
    throw "Promotion mode requires -SelfPlay for candidate/opponent evaluation."
}

if ($SelfPlayInterval -lt 1) {
    throw "SelfPlayInterval must be positive."
}

if ($RunMode -eq "promotion" -and $SelfPlayInterval -ne 1) {
    throw "Promotion mode requires self-play every generation (SelfPlayInterval=1)."
}

# Prevent an inherited global package path from changing the project environment.
$env:PYTHONPATH = $null

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuRows = @(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits)
    foreach ($row in $gpuRows) {
        $parts = $row -split "," | ForEach-Object { [int]($_.Trim()) }
        if ($parts.Count -ge 3) {
            $utilization = $parts[0]
            $memoryMiB = $parts[1]
            $temperatureC = $parts[2]
            if ($utilization -gt $MaxGpuUtilization -or
                $memoryMiB -gt $MaxGpuMemoryMiB -or
                $temperatureC -gt $MaxGpuTemperatureC) {
                throw ("GPU is busy or hot (utilization={0}%, memory={1} MiB, temperature={2} C). " +
                    "Stop GTA/other GPU workloads before starting the overnight CUDA run.") -f `
                    $utilization, $memoryMiB, $temperatureC
            }
        }
    }
}

$prefix = "frontier_${StartSeason}_${EndSeason}_overnight_${RunId}"
$output = Join-Path $repo "data\models\${prefix}.pt"
$checkpoint = Join-Path $repo "data\models\${prefix}_state.pt"
$report = Join-Path $repo "reports\${prefix}.json"

foreach ($path in @($output, $checkpoint, $report)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing run artifact: $path"
    }
}

$parityReport = Join-Path $repo $ParityReport

if ($RunMode -eq "promotion") {
    if (-not (Test-Path -LiteralPath $parityReport)) {
        throw "Promotion parity report is missing: $parityReport"
    }
    $parity = Get-Content -LiteralPath $parityReport -Raw | ConvertFrom-Json
    foreach ($field in @(
        "exact_standings_match_count",
        "exact_champion_match_count",
        "exact_weekly_score_match_count",
        "transaction_actions_exact_count",
        "transaction_state_exact_count",
        "transaction_reward_exact_count"
    )) {
        if ([int]$parity.$field -lt 4) {
            throw "Promotion parity report is not exact across 2021-2024: $field=$($parity.$field)"
        }
    }
    if (-not [bool]$parity.transactions) {
        throw "Promotion parity report does not prove transaction-enabled parity."
    }
    if ([int]$parity.players -ne $Players) {
        throw "Promotion parity player count ($($parity.players)) does not match requested Players ($Players)."
    }
    if (-not (Test-Path -LiteralPath $MultiSeedReport)) {
        throw "Promotion multi-seed report is missing: $MultiSeedReport"
    }
}

if ($RunMode -eq "promotion") {
    $multiSeed = Get-Content -LiteralPath $MultiSeedReport -Raw | ConvertFrom-Json
    if (-not [bool]$multiSeed.promotion_ready_multi_seed) {
        throw "Promotion multi-seed report is not ready: promotion_ready_multi_seed=$($multiSeed.promotion_ready_multi_seed)"
    }
}

$arguments = @(
    "-u",
    "-m", "scripts.train_cuda_manager_policy",
    "--device", "cuda",
    "--start-season", $StartSeason,
    "--end-season", $EndSeason,
    "--population", $Population,
    "--generations", $Generations,
    "--selection", "4",
    "--scenario-repeats", $ScenarioRepeats,
    "--projection-noise", "0.015",
    "--loader-workers", "1",
    "--scenario-refresh-generations", $ScenarioRefreshGenerations,
    "--season-subsample-size", $SeasonSubsampleSize,
    "--season-replay-interval", $SeasonReplayInterval,
    "--self-play-interval", $SelfPlayInterval,
    "--players", $Players,
    "--holdout-season", "0",
    "--holdout-seasons", "2024", "2025",
    "--opponent-archive-size", $OpponentArchiveSize,
    "--output", $output,
    "--checkpoint", $checkpoint,
    "--report", $report
)

if ($SelfPlay) {
    $arguments += "--self-play"
}

if ($FullPolicyMutation) {
    $arguments += "--full-policy-mutation"
}

if ($BatchedPolicyHeads) {
    $arguments += "--batched-policy-heads"
}

if ($Deterministic) {
    $arguments += "--deterministic"
}

if ($CompilePolicy) {
    $arguments += "--compile-policy"
}

if ($RunMode -eq "exploratory-draft") {
    $arguments += "--disable-transactions"
}

if ($RunMode -eq "promotion") {
    $arguments += @(
        "--parity-report", $parityReport,
        "--require-promotion-ready",
        "--multi-seed-report", $MultiSeedReport,
        "--require-multi-seed-promotion"
    )
}

Write-Host "Starting frontier overnight run from $repo"
Write-Host ("Run mode: {0}; run id: {1}" -f $RunMode, $RunId)
Write-Host ("Training seasons: {0}-{1}; holdouts: 2024, 2025; population: {2}; generations: {3}" -f `
    $StartSeason, $EndSeason, $Population, $Generations)
Write-Host ("Self-play: {0}; transactions: {1}; separate output: {2}" -f `
    $SelfPlay.IsPresent, ($RunMode -eq "promotion"), $output)

$child = $null
try {
    $child = Start-Process -FilePath $python -ArgumentList $arguments -NoNewWindow -PassThru
    $child.WaitForExit()
    $exitCode = $child.ExitCode
    if ($null -eq $exitCode) {
        # Windows can expose a null ExitCode after WaitForExit when the
        # process has already been reaped.  The report is authoritative for
        # this wrapper's post-run status in that case.
        if (Test-Path -LiteralPath $report) {
            $completed = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
            if ($completed.status -eq "complete") {
                $exitCode = 0
            }
        }
    }
    if ($exitCode -ne 0) {
        throw "Frontier overnight run failed with exit code $exitCode"
    }
}
finally {
    if ($null -ne $child) {
        Stop-ProcessTree -ProcessId $child.Id
    }
}

Write-Host "Frontier overnight run completed. Inspect $report before any promotion decision."
