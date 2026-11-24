"""Inspect and prepare CSV datasets without guessing away their contents.

The original project sample is a small synthetic URL/label CSV. For a real
87-feature project dataset, point this module at that file explicitly (or place
it in ``dataset/``). This module never fabricates missing features or silently
deduplicates records.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .feature_extraction import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
REPORT_PATH = ROOT / "reports" / "dataset_inspection.json"
EXPECTED_INPUT_FEATURES = 87
TARGET_NAME_HINTS = {
    "label", "class", "target", "result", "status", "type", "category",
    "phishing", "is_phishing", "class_label", "target_label", "legitimate",
}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def identify_target_column(frame: pd.DataFrame) -> str:
    """Find a plausible label column; fail clearly when the data is ambiguous."""
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("A labeled dataset needs at least one input column and one target column.")
    normalized = {column: re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") for column in frame.columns}
    named = [column for column, name in normalized.items() if name in TARGET_NAME_HINTS]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise ValueError(f"Several columns look like target labels: {named}. Specify the target column explicitly.")

    candidates = []
    for column in frame.columns:
        values = frame[column].dropna()
        if 1 < values.nunique() <= min(20, max(2, len(frame) // 10)):
            text_values = {str(value).strip().lower() for value in values.unique()}
            if text_values & {"phishing", "phish", "legitimate", "benign", "malicious", "good", "bad"}:
                candidates.append(column)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Several columns contain class-like values: {candidates}. Specify the target column explicitly.")
    raise ValueError("Could not identify a target label column. Pass target_column explicitly after inspecting the CSV columns.")


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def inspect_csv(path: str | Path, target_column: str | None = None) -> dict:
    path = Path(path)
    frame = _read_csv(path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")
    target = target_column or identify_target_column(frame)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not in {path.name}.")
    input_columns = [column for column in frame.columns if column != target]
    url_names = {"url", "uri", "website", "link", "domain_url"}
    raw_url_columns = [column for column in input_columns if re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") in url_names]
    feature_columns = [column for column in input_columns if column not in raw_url_columns]
    numeric_feature_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    url_only = len(input_columns) == 1 and bool(raw_url_columns)
    url_plus_features = bool(raw_url_columns) and bool(feature_columns)
    class_counts = frame[target].dropna().value_counts(dropna=False)
    input_features = len(input_columns)
    return {
        "file": path.name,
        "path": str(path),
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": [str(column) for column in frame.columns],
        "target_column": str(target),
        "input_feature_count": input_features,
        "raw_url_column_count": len(raw_url_columns),
        "extracted_feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_feature_columns),
        "has_exactly_87_input_features": len(feature_columns) == EXPECTED_INPUT_FEATURES,
        "has_exactly_87_numeric_features": len(numeric_feature_columns) == EXPECTED_INPUT_FEATURES,
        "data_type": "raw_urls" if url_only else ("raw_url_plus_extracted_features" if url_plus_features else "extracted_features"),
        "raw_url_columns": [str(column) for column in raw_url_columns],
        "missing_values_by_column": {str(column): int(count) for column, count in frame.isna().sum().items()},
        "total_missing_values": int(frame.isna().sum().sum()),
        "duplicate_records": int(frame.duplicated(keep="first").sum()),
        "class_distribution": {str(_json_value(label)): int(count) for label, count in class_counts.items()},
        "actual_label_values": [_json_value(value) for value in frame[target].dropna().drop_duplicates().tolist()],
        "rows_missing_target": int(frame[target].isna().sum()),
        "rows_with_complete_target": int(frame[target].notna().sum()),
        "notes": [
            "Duplicate rows are reported and retained; no records are silently removed.",
            "Missing target labels cannot be imputed and are excluded only from supervised train/test splits.",
            "Missing input values are imputed in the training pipeline using training-fold statistics.",
            "A raw-URL-only CSV may be converted to the prototype lexical features. If precomputed feature columns are present, the raw URL identifier is excluded and the precomputed values are retained unchanged.",
        ],
    }


def inspect_dataset(dataset_dir: str | Path = DATASET_DIR, report_path: str | Path = REPORT_PATH) -> dict:
    """Inspect every CSV directly inside dataset_dir and save a JSON report."""
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.rglob("*.csv")) if dataset_dir.exists() else []
    report: dict[str, Any] = {"dataset_directory": str(dataset_dir), "csv_files_found": len(files), "datasets": [], "errors": []}
    for path in files:
        try:
            report["datasets"].append(inspect_csv(path))
        except Exception as exc:
            report["errors"].append({"file": path.name, "error": str(exc)})
    report["status"] = "inspected" if report["datasets"] else "missing_or_uninspectable"
    report["required_real_dataset_missing"] = not any(
        item["shape"]["rows"] >= 1000 and item["extracted_feature_count"] == EXPECTED_INPUT_FEATURES
        for item in report["datasets"]
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _features_and_target(frame: pd.DataFrame, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series, str]:
    target = target_column or identify_target_column(frame)
    if target not in frame:
        raise ValueError(f"Target column {target!r} is missing from dataset.")
    if frame[target].isna().all():
        raise ValueError("Target column has no labeled rows.")
    usable = frame.loc[frame[target].notna()].copy()
    y = usable.pop(target)
    url_names = {"url", "uri", "website", "link", "domain_url"}
    raw_url_columns = [column for column in usable.columns if re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") in url_names]
    feature_columns = [column for column in usable.columns if column not in raw_url_columns]
    if raw_url_columns and not feature_columns and len(raw_url_columns) == 1:
        url_column = raw_url_columns[0]
        rows = []
        for value in usable[url_column]:
            if pd.isna(value) or not str(value).strip():
                rows.append({name: pd.NA for name in FEATURE_NAMES})
            else:
                rows.append(extract_features(str(value)))
        usable = pd.DataFrame(rows, columns=FEATURE_NAMES)
    elif raw_url_columns:
        # Raw URLs are identifiers, not precomputed features. Never infer or invent
        # values for the extracted feature columns.
        usable = usable.drop(columns=raw_url_columns)
    if usable.empty:
        raise ValueError("No input features remain after separating the target column.")
    # Reject columns that are entirely missing: they contain no usable signal.
    all_missing = [str(column) for column in usable.columns if usable[column].isna().all()]
    if all_missing:
        raise ValueError(f"Input columns are entirely missing and cannot be imputed: {all_missing}")
    return usable, y, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        category_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", category_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_train_test(
    path: str | Path,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Return raw split features, labels, and an imputation/encoding transformer.

    Duplicate rows are retained as requested. Stratification is used when every
    class has enough observations to appear in both partitions.
    """
    frame = _read_csv(path)
    X, y, target = _features_and_target(frame, target_column)
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and int(counts.min()) >= 2 else None
    if stratify is None:
        raise ValueError("Stratified split requires at least two classes with at least two labeled rows each.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )
    preprocessor = build_preprocessor(X_train)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "preprocessor": preprocessor,
        "target_column": target,
        "feature_columns": list(X.columns),
        "label_values": [_json_value(value) for value in y.drop_duplicates().tolist()],
        "duplicate_records_retained": int(frame.duplicated(keep="first").sum()),
        "stratified": True,
    }


def load_dataset(path: str | Path, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Compatibility loader returning features and labels (no row deletion)."""
    frame = _read_csv(path)
    X, y, _ = _features_and_target(frame, target_column)
    return X, y


if __name__ == "__main__":
    result = inspect_dataset()
    print(json.dumps(result, indent=2, ensure_ascii=False))
