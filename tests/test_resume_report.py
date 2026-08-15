import json

from scripts.resume_modular_manager_policy import load_resume_report


def test_load_resume_report_preserves_generations_from_interrupted_attempt(tmp_path):
    report_path = tmp_path / "segment.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "running",
                "started_at": "old",
                "generations": [{"generation_number": 51}],
                "stages": {"replay": {"records": 10}},
                "final_evaluation": {"selected_generation": 51},
            }
        ),
        encoding="utf-8",
    )

    report = load_resume_report(
        report_path=report_path,
        state_checkpoint=tmp_path / "state.pt",
        transaction_mode="genome",
        transaction_value_output=tmp_path / "value.pt",
        state_completed_generations=55,
        additional_generations=5,
        population_size=24,
    )

    assert report["generations"] == [{"generation_number": 51}]
    assert report["stages"] == {"replay": {"records": 10}}
    assert report["resume_attempts"] == 1
    assert "final_evaluation" not in report
    assert report["configuration"]["target_generations_after_resume"] == 60
