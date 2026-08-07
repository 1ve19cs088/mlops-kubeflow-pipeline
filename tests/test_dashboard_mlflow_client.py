"""
Tests for the dashboard's MLflow client wrapper.

Mocks the underlying mlflow.MlflowClient entirely — these tests verify
that MlflowRegistryClient calls the right underlying methods and
shapes their results correctly, not MLflow's own read/write behavior
(already covered by tests/test_mlflow_tracking.py against a real,
isolated registry). No real MLflow store is touched, nothing is
written to the project root.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dashboard.mlflow_client import MlflowRegistryClient, get_mlflow_client


def _fake_registered_model(name, last_updated_timestamp):
    return SimpleNamespace(
        name=name,
        creation_timestamp=last_updated_timestamp,
        last_updated_timestamp=last_updated_timestamp,
        description=None,
        tags={},
    )


def _fake_model_version(name, version, current_stage="None", run_id="run-1"):
    return SimpleNamespace(
        name=name,
        version=version,
        current_stage=current_stage,
        run_id=run_id,
        source=f"models:/m-{version}",
        creation_timestamp=1000 + version,
    )


def _client_with_mock(mock_mlflow_client):
    """
    Builds an MlflowRegistryClient whose underlying mlflow.MlflowClient
    is never actually constructed — patches the class itself so the
    real constructor (which touches disk even for a throwaway sqlite
    path) never runs at all, guaranteeing zero filesystem writes.
    """

    with patch(
        "dashboard.mlflow_client.MlflowClient", return_value=mock_mlflow_client
    ):
        return MlflowRegistryClient(tracking_uri="sqlite:///unused.db")


def test_get_registered_models_sorts_by_last_updated_desc():
    mock_client = MagicMock()
    mock_client.search_registered_models.return_value = [
        _fake_registered_model("model-a", last_updated_timestamp=100),
        _fake_registered_model("model-b", last_updated_timestamp=200),
    ]
    client = _client_with_mock(mock_client)

    models = client.get_registered_models()

    assert [m.name for m in models] == ["model-b", "model-a"]


def test_get_model_versions_sorts_newest_first():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _fake_model_version("iris-model", version=1),
        _fake_model_version("iris-model", version=3),
        _fake_model_version("iris-model", version=2),
    ]
    client = _client_with_mock(mock_client)

    versions = client.get_model_versions("iris-model")

    assert [v.version for v in versions] == [3, 2, 1]
    mock_client.search_model_versions.assert_called_once_with("name='iris-model'")


def test_get_latest_version_returns_highest_version_number():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _fake_model_version("iris-model", version=1),
        _fake_model_version("iris-model", version=2),
    ]
    client = _client_with_mock(mock_client)

    latest = client.get_latest_version("iris-model")

    assert latest.version == 2


def test_get_latest_version_returns_none_when_no_versions_exist():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = []
    client = _client_with_mock(mock_client)

    assert client.get_latest_version("no-such-model") is None


def test_get_production_version_returns_version_in_production_stage():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _fake_model_version("iris-model", version=1, current_stage="None"),
        _fake_model_version("iris-model", version=2, current_stage="Production"),
    ]
    client = _client_with_mock(mock_client)

    production = client.get_production_version("iris-model")

    assert production.version == 2


def test_get_production_version_returns_none_when_nothing_promoted():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _fake_model_version("iris-model", version=1, current_stage="None"),
    ]
    client = _client_with_mock(mock_client)

    assert client.get_production_version("iris-model") is None


def test_get_run_metrics_returns_dict_of_metrics():
    mock_client = MagicMock()
    mock_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"test_accuracy": 0.95}, params={})
    )
    client = _client_with_mock(mock_client)

    metrics = client.get_run_metrics("run-123")

    assert metrics == {"test_accuracy": 0.95}
    mock_client.get_run.assert_called_once_with("run-123")


def test_get_run_parameters_returns_dict_of_params():
    mock_client = MagicMock()
    mock_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            metrics={}, params={"algorithm": "RandomForestClassifier"}
        )
    )
    client = _client_with_mock(mock_client)

    params = client.get_run_parameters("run-123")

    assert params == {"algorithm": "RandomForestClassifier"}


def test_get_run_tags_returns_dict_of_tags():
    mock_client = MagicMock()
    mock_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            metrics={}, params={}, tags={"mlflow.source.git.commit": "abc123"}
        )
    )
    client = _client_with_mock(mock_client)

    tags = client.get_run_tags("run-123")

    assert tags == {"mlflow.source.git.commit": "abc123"}
    mock_client.get_run.assert_called_once_with("run-123")


def test_get_git_commit_returns_the_tag_value_when_present():
    mock_client = MagicMock()
    mock_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            metrics={}, params={}, tags={"mlflow.source.git.commit": "abc123"}
        )
    )
    client = _client_with_mock(mock_client)

    assert client.get_git_commit("run-123") == "abc123"


def test_get_git_commit_returns_none_when_the_tag_is_absent():
    mock_client = MagicMock()
    mock_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={}, params={}, tags={})
    )
    client = _client_with_mock(mock_client)

    assert client.get_git_commit("run-123") is None


def test_list_run_artifacts_returns_top_level_file_infos():
    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        SimpleNamespace(path="training_report.json", is_dir=False, file_size=42),
        SimpleNamespace(path="model", is_dir=True, file_size=None),
    ]
    client = _client_with_mock(mock_client)

    artifacts = client.list_run_artifacts("run-123")

    assert [a.path for a in artifacts] == ["training_report.json", "model"]
    mock_client.list_artifacts.assert_called_once_with("run-123")


def test_get_artifact_bytes_reads_the_downloaded_file(tmp_path):
    downloaded_file = tmp_path / "training_report.json"
    downloaded_file.write_bytes(b'{"algorithm": "RandomForestClassifier"}')
    client = _client_with_mock(MagicMock())

    with patch(
        "dashboard.mlflow_client.mlflow.artifacts.download_artifacts",
        return_value=str(downloaded_file),
    ) as mock_download:
        content = client.get_artifact_bytes("run-123", "training_report.json")

    assert content == b'{"algorithm": "RandomForestClassifier"}'
    mock_download.assert_called_once_with(
        run_id="run-123",
        artifact_path="training_report.json",
        tracking_uri="sqlite:///unused.db",
    )


def test_get_mlflow_client_returns_registry_client_instance():
    client = get_mlflow_client()

    assert isinstance(client, MlflowRegistryClient)
