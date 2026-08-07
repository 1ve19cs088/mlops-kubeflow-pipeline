"""
Tests for dashboard/deployment_pipeline_status.py.

Uses a tmp_path standing in for the project root so "file present" vs
"file absent" can be tested directly, without touching this repo's
real files.
"""

from dashboard.deployment_pipeline_status import get_deployment_pipeline_status


def test_reports_not_configured_when_nothing_exists(tmp_path):
    status = get_deployment_pipeline_status(project_root=tmp_path)

    assert status["training"] == "Not Configured"
    assert status["registry"] == "Not Configured"
    assert status["docker_build"] == "Not Configured"


def test_reports_available_when_the_relevant_files_exist(tmp_path):
    (tmp_path / "src" / "pipeline").mkdir(parents=True)
    (tmp_path / "src" / "pipeline" / "run_pipeline.py").write_text("")
    (tmp_path / "src" / "tracking").mkdir(parents=True)
    (tmp_path / "src" / "tracking" / "mlflow_tracking.py").write_text("")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "Dockerfile.serving").write_text("")
    (tmp_path / "docker" / "Dockerfile.pipeline").write_text("")

    status = get_deployment_pipeline_status(project_root=tmp_path)

    assert status["training"] == "Available"
    assert status["registry"] == "Available"
    assert status["docker_build"] == "Available"


def test_docker_build_requires_both_dockerfiles(tmp_path):
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "Dockerfile.serving").write_text("")

    status = get_deployment_pipeline_status(project_root=tmp_path)

    assert status["docker_build"] == "Not Configured"


def test_deployment_and_current_deployment_are_honestly_not_yet_built(tmp_path):
    status = get_deployment_pipeline_status(project_root=tmp_path)

    assert status["deployment"] == "Future Stage"
    assert status["current_deployment"] == "Not Configured"


def test_default_project_root_reflects_this_real_repository():
    status = get_deployment_pipeline_status()

    assert status["training"] == "Available"
    assert status["registry"] == "Available"
    assert status["docker_build"] == "Available"
