"""Versioned provenance for model features and normalization.

Checkpoints are only reproducible when the feature order, cutoff, identity
mapping, scoring rules, and normalization are recorded beside the weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureManifest:
    feature_names: tuple[str, ...]
    schema_version: str = "1"
    decision_cutoff: str = ""
    scoring_settings: tuple[tuple[str, float], ...] = ()
    identity_map_version: str = "stable-player-id-v1"
    normalization_means: tuple[float, ...] = ()
    normalization_standard_deviations: tuple[float, ...] = ()
    source_checksums: tuple[tuple[str, str], ...] = ()
    code_revision: str = ""

    def __post_init__(self) -> None:
        if len(self.normalization_means) != len(self.normalization_standard_deviations):
            raise ValueError("Normalization means and standard deviations must align.")
        if self.normalization_means and len(self.feature_names) != len(self.normalization_means):
            raise ValueError("Normalization statistics must match feature count.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "feature_names": list(self.feature_names),
            "scoring_settings": [list(item) for item in self.scoring_settings],
            "source_checksums": [list(item) for item in self.source_checksums],
            "normalization_means": list(self.normalization_means),
            "normalization_standard_deviations": list(self.normalization_standard_deviations),
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        expected = checkpoint.get("feature_manifest")
        if expected is None:
            raise ValueError("Checkpoint is missing feature_manifest provenance.")
        expected_digest = checkpoint.get("feature_manifest_digest")
        if expected_digest != self.digest():
            raise ValueError(
                "Checkpoint feature manifest is incompatible with the requested model."
            )


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_feature_manifest(
    feature_names: tuple[str, ...],
    means: tuple[float, ...] = (),
    standard_deviations: tuple[float, ...] = (),
    decision_cutoff: str = "",
    scoring_settings: dict[str, float] | None = None,
    source_paths: tuple[Path, ...] = (),
    code_revision: str = "",
) -> FeatureManifest:
    return FeatureManifest(
        feature_names=feature_names,
        decision_cutoff=decision_cutoff,
        scoring_settings=tuple(sorted((scoring_settings or {}).items())),
        normalization_means=means,
        normalization_standard_deviations=standard_deviations,
        source_checksums=tuple(
            sorted((str(path), checksum_file(path)) for path in source_paths if path.exists())
        ),
        code_revision=code_revision,
    )


def validate_checkpoint_manifest(checkpoint: dict[str, Any]) -> None:
    """Validate the self-consistency of provenance embedded in a checkpoint."""
    manifest = checkpoint.get("feature_manifest")
    digest = checkpoint.get("feature_manifest_digest")
    if manifest is None and digest is None:
        return
    if manifest is None or digest is None:
        raise ValueError("Checkpoint has incomplete feature manifest provenance.")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != digest:
        raise ValueError("Checkpoint feature manifest digest does not match its contents.")


def validate_feature_names(
    checkpoint: dict[str, Any],
    expected_names: tuple[str, ...],
) -> None:
    manifest = checkpoint.get("feature_manifest")
    if manifest is None:
        return
    if tuple(manifest.get("feature_names", ())) != expected_names:
        raise ValueError("Checkpoint feature names are incompatible with this code revision.")
