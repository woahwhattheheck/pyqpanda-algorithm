from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "main.yml"


def _automated_test_step() -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = "      - name: Run Automated Tests\n"
    assert marker in workflow, "Automated Testing job must keep an explicit test step"
    remainder = workflow.split(marker, 1)[1]
    return remainder.split("\n      - name:", 1)[0]


def test_installed_wheel_suite_is_a_single_fail_closed_gate():
    step = _automated_test_step()

    assert step.count("pytest -v") == 1, (
        "the installed wheel should be tested once so reporting passes cannot disagree"
    )
    assert "|| true" not in step, "pytest failures must propagate to the CI job"

    for report_flag in (
        "--junitxml=",
        "--alluredir=",
        "--html=",
        "--self-contained-html",
    ):
        assert report_flag in step, f"authoritative pytest pass must retain {report_flag}"
