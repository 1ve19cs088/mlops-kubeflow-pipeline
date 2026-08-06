"""
API request/response schemas for the model serving layer.

Only PredictionRequest/BatchPredictionRequest are built dynamically —
their shape (which named fields exist) genuinely varies per dataset,
driven by the training contract's feature_names/feature_dtypes. Every
response model below is an ordinary, statically-defined Pydantic
class: their shape doesn't vary by dataset, only the values do.
"""

from pydantic import BaseModel, ConfigDict, create_model

DTYPE_TO_PYTHON_TYPE: dict[str, type] = {
    "float64": float,
    "float32": float,
    "int64": int,
    "int32": int,
    "bool": bool,
    "object": str,
    "str": str,
    "string": str,
}


def _python_type_for(dtype: str) -> type:
    """
    Map a pandas dtype string (as recorded in training_report.json's
    feature_dtypes) to a Python/Pydantic field type.

    Falls back to `str` for any dtype not in the table, rather than
    raising — an unrecognized dtype should degrade to permissive
    validation, not crash schema construction at startup.
    """

    return DTYPE_TO_PYTHON_TYPE.get(dtype, str)


def build_prediction_request_model(contract: dict) -> type[BaseModel]:
    """
    Build a Pydantic model for a single prediction request, with one
    required field per feature the model was trained on.

    extra="forbid" is deliberate: without it, Pydantic v2 silently
    ignores fields it doesn't recognize, so a client typo (e.g.
    "sepal_lenght_cm") would be dropped instead of flagged, and could
    be mistaken for influencing the prediction when it never did.

    Args:
        contract: The training contract (training_report.json contents).

    Returns:
        A dynamically constructed PredictionRequest Pydantic class.
    """

    fields = {
        feature_name: (
            _python_type_for(contract["feature_dtypes"][feature_name]),
            ...,
        )
        for feature_name in contract["feature_names"]
    }

    return create_model(
        "PredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def build_batch_request_model(prediction_model: type[BaseModel]) -> type[BaseModel]:
    """
    Build a Pydantic model wrapping a list of prediction requests.

    Args:
        prediction_model: The dynamically built PredictionRequest model.

    Returns:
        A dynamically constructed BatchPredictionRequest Pydantic class.
    """

    return create_model(
        "BatchPredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        records=(list[prediction_model], ...),
    )


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    algorithm: str
    framework: str
    model_version: str
    trained_at: str
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    target_column: str
    class_labels: list[str]


class PredictionResponse(BaseModel):
    prediction: str | float


class BatchPredictionResponse(BaseModel):
    predictions: list[str | float]


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class ClassMetrics(BaseModel):
    num_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class TestMetrics(ClassMetrics):
    confusion_matrix: ConfusionMatrix


class MetricsResponse(BaseModel):
    dataset: str
    algorithm: str
    model_version: str
    trained_at: str
    evaluated_at: str
    train: ClassMetrics
    test: TestMetrics
