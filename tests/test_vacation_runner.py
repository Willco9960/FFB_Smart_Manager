import json
from argparse import Namespace
from pathlib import Path

from scripts.run_modular_vacation_training import write_manifest
from scripts.show_modular_run_status import resolve_manifest_path


def test_write_manifest_replaces_partial_file_atomically(tmp_path):
    manifest_path = tmp_path / "run" / "manifest.json"
    write_manifest(manifest_path, {"status": "running", "completed_segments": 2})

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "status": "running",
        "completed_segments": 2,
    }
    assert not manifest_path.with_suffix(".json.tmp").exists()


def test_status_command_resolves_run_id_manifest():
    manifest_path = resolve_manifest_path(Namespace(run_id="demo", manifest=None))

    assert manifest_path == Path("data/models/vacation_runs/demo/manifest.json")
