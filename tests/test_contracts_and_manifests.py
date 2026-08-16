import pytest

from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT
from models.feature_manifest import FeatureManifest, validate_checkpoint_manifest


def test_fitness_contract_digest_is_stable():
    assert ESPN_FITNESS_CONTRACT.digest() == ESPN_FITNESS_CONTRACT.digest()
    assert len(ESPN_FITNESS_CONTRACT.digest()) == 64


def test_feature_manifest_rejects_incompatible_checkpoint():
    manifest = FeatureManifest(feature_names=("a", "b"))
    checkpoint = {
        "feature_manifest": manifest.to_dict(),
        "feature_manifest_digest": manifest.digest(),
    }
    manifest.validate_checkpoint(checkpoint)

    incompatible = FeatureManifest(feature_names=("a", "c"))
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.validate_checkpoint(checkpoint)


def test_feature_manifest_requires_provenance():
    manifest = FeatureManifest(feature_names=("a",))
    with pytest.raises(ValueError, match="missing feature_manifest"):
        manifest.validate_checkpoint({})


def test_checkpoint_manifest_rejects_tampering():
    manifest = FeatureManifest(feature_names=("a",))
    checkpoint = {
        "feature_manifest": manifest.to_dict(),
        "feature_manifest_digest": "bad",
    }
    with pytest.raises(ValueError, match="digest"):
        validate_checkpoint_manifest(checkpoint)
