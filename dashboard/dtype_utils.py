"""
Helpers for turning HTML form / CSV string values into the correctly
typed values the existing /v1/predict API expects, based on
feature_dtypes from /v1/metadata.

This is UI-side marshaling only — it never decides what a valid
prediction is, only how to shape a string into the right JSON type
for whatever features the current model's contract declares. Nothing
here is Iris-specific: driven entirely by feature_dtypes, so a
six-feature dataset renders six correctly-typed inputs automatically.
"""

NUMBER_DTYPES = {"float64", "float32", "int64", "int32"}
FLOAT_DTYPES = {"float64", "float32"}
INT_DTYPES = {"int64", "int32"}


def html_input_type(dtype: str) -> str:
    if dtype in NUMBER_DTYPES:
        return "number"
    if dtype == "bool":
        return "checkbox"
    return "text"


def coerce_value(raw_value, dtype: str):
    if dtype in FLOAT_DTYPES:
        return float(raw_value)
    if dtype in INT_DTYPES:
        return int(float(raw_value))
    if dtype == "bool":
        return str(raw_value).strip().lower() in ("1", "true", "on", "yes")
    return raw_value
