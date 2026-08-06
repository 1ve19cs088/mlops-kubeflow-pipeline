"""
Generic JSON report/artifact writer.

Used by any pipeline stage that needs to persist a JSON-serializable
report (validation reports, evaluation metrics, etc.) to the artifacts
directory.
"""

import json
from pathlib import Path
from typing import Any


def save_json_report(directory: Path, filename: str, data: dict[str, Any]) -> Path:
    """
    Persist a dictionary as a JSON report inside the given directory.

    Args:
        directory: Target directory (created if missing).
        filename: Report filename, e.g. "validation_report.json".
        data: JSON-serializable report contents.

    Returns:
        Path to the written report file.
    """

    directory.mkdir(parents=True, exist_ok=True)

    report_path = directory / filename

    with open(report_path, "w") as file:
        json.dump(data, file, indent=2, default=str)

    return report_path
