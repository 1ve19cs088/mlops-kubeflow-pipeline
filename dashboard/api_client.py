"""
Thin HTTP client for the existing FastAPI serving API.

Every dashboard page consumes the API exclusively through this
module — no prediction/validation/metrics logic is duplicated here,
only HTTP calls to endpoints that already exist under app/.
"""

import os

import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


class ApiUnavailableError(Exception):
    """Raised when the existing FastAPI service can't be reached at all."""


class ApiValidationError(Exception):
    """Raised when the existing API rejects a request (4xx) — carries its detail."""

    def __init__(self, status_code: int, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API returned {status_code}: {detail}")


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _get(self, path: str) -> dict:
        try:
            response = httpx.get(f"{self.base_url}{path}", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            raise ApiUnavailableError(str(error)) from error

    def _post(self, path: str, payload: dict, timeout: float = 10.0) -> dict:
        try:
            response = httpx.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        except httpx.HTTPError as error:
            raise ApiUnavailableError(str(error)) from error

        if response.status_code >= 500:
            raise ApiUnavailableError(f"API returned {response.status_code}")
        if response.status_code >= 400:
            raise ApiValidationError(response.status_code, response.json().get("detail"))

        return response.json()

    def get_health(self) -> dict:
        return self._get("/v1/health")

    def get_metadata(self) -> dict:
        return self._get("/v1/metadata")

    def get_metrics(self) -> dict:
        return self._get("/v1/metrics")

    def predict(self, payload: dict) -> dict:
        return self._post("/v1/predict", payload)

    def predict_batch(self, records: list) -> dict:
        return self._post("/v1/predict/batch", {"records": records}, timeout=30.0)


def get_api_client() -> ApiClient:
    """FastAPI dependency provider — overridable in tests."""

    return ApiClient()
