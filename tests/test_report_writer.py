"""
Tests for the generic JSON report writer.
"""

import json
from pathlib import Path

from src.utils.report_writer import save_json_report


def test_save_json_report_writes_expected_content(tmp_path):
    data = {"accuracy": 0.95, "labels": ["a", "b"]}

    report_path = save_json_report(tmp_path, "report.json", data)

    assert report_path == tmp_path / "report.json"
    assert json.loads(report_path.read_text()) == data


def test_save_json_report_creates_missing_directory(tmp_path):
    nested_dir = tmp_path / "nested" / "dir"

    report_path = save_json_report(nested_dir, "report.json", {"x": 1})

    assert report_path.exists()


def test_save_json_report_serializes_non_native_values_via_default_str(tmp_path):
    data = {"a_path": Path("/some/path")}

    report_path = save_json_report(tmp_path, "report.json", data)

    content = json.loads(report_path.read_text())
    assert content["a_path"] == "/some/path"
