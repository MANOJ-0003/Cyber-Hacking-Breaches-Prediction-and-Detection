"""Offline URL prediction using the exact schema recorded during training."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd

from .feature_extraction import FEATURE_NAMES, extract_features, normalize_url, explain_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"
DEFAULT_METADATA_PATH = ROOT / "models" / "feature_metadata.json"
PROBABILITY_CAVEAT = (
    "Model probabilities are estimates from this trained classifier, not calibrated "
    "or guaranteed security scores. A low risk result does not prove a URL is safe."
)


class PredictionError(ValueError):
    """Raised when saved artifacts or their schema cannot safely score a URL."""


def load_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> dict:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise PredictionError(f"Feature metadata is missing: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionError(f"Could not read feature metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise PredictionError("Feature metadata must be a JSON object.")
    return metadata


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load the saved joblib pipeline; never auto-trains or accesses the network."""
    model_path = Path(path)
    if not model_path.is_file():
        raise PredictionError(
            f"Trained model is missing: {model_path}. Training is paused until the "
            "real dataset and matching 87-feature extraction definitions are provided."
        )
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise PredictionError(f"Could not load the saved model: {exc}") from exc


def load_artifacts(model_path: str | Path = DEFAULT_MODEL_PATH,
                   metadata_path: str | Path = DEFAULT_METADATA_PATH):
    """Load both model and metadata as a convenient independent API."""
    return load_model(model_path), load_metadata(metadata_path)


def _metadata_feature_order(metadata: dict) -> list[str]:
    names = metadata.get("input_feature_order") or metadata.get("input_feature_names")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise PredictionError("Metadata does not contain an ordered input_feature_order list.")
    if len(names) != len(set(names)):
        raise PredictionError("Metadata contains duplicate input feature names.")
    return names


def _label_equal(left, right) -> bool:
    return left == right or str(left) == str(right)


def _class_mapping(metadata: dict, model) -> tuple[object, object, list]:
    mapping = metadata.get("label_mapping")
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("classifier"), "classes_", None)
    if classes is None or len(classes) != 2:
        raise PredictionError("The loaded classifier must expose exactly two trained classes.")
    if not isinstance(mapping, dict) or "phishing" not in mapping or "legitimate" not in mapping:
        raise PredictionError(
            "Metadata is missing the dataset-confirmed label_mapping for phishing and legitimate."
        )
    phishing_label = mapping["phishing"]
    legitimate_label = mapping["legitimate"]
    if not any(_label_equal(phishing_label, item) for item in classes):
        raise PredictionError("Metadata phishing label is not present in the trained model classes.")
    if not any(_label_equal(legitimate_label, item) for item in classes):
        raise PredictionError("Metadata legitimate label is not present in the trained model classes.")
    if _label_equal(phishing_label, legitimate_label):
        raise PredictionError("Metadata maps phishing and legitimate to the same label.")
    return phishing_label, legitimate_label, list(classes)


def _risk_category(phishing_probability: float) -> str:
    # Presentation bands only; they are not calibrated safety thresholds.
    if phishing_probability < 0.33:
        return "Low Risk"
    if phishing_probability < 0.67:
        return "Medium Risk"
    return "High Risk"


def predict_url(
    value: str,
    model=None,
    metadata: dict | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> dict:
    """Validate and score a submitted URL without visiting it.

    Optional model and metadata arguments make the module independently usable
    with an already-loaded estimator, for example in tests or another service.
    """
    # Validate first, so malformed input is rejected even if model artifacts are absent.
    normalized_url = normalize_url(value)
    features = extract_features(value)
    if metadata is None:
        metadata = load_metadata(metadata_path)
    if model is None:
        model = load_model(model_path)

    feature_order = _metadata_feature_order(metadata)
    if feature_order != FEATURE_NAMES:
        missing = [name for name in feature_order if name not in features]
        unexpected = [name for name in features if name not in feature_order]
        raise PredictionError(
            "The URL extractor does not match the model's exact training feature order. "
            f"Missing extractor features: {missing}; unexpected extractor features: {unexpected}. "
            "Provide the actual 87-feature extraction code or documentation."
        )
    model_feature_order = getattr(model, "feature_names_in_", None)
    if model_feature_order is not None and list(model_feature_order) != feature_order:
        raise PredictionError("The loaded model's feature order differs from its metadata.")

    phishing_label, legitimate_label, classes = _class_mapping(metadata, model)
    row = pd.DataFrame([{name: features[name] for name in feature_order}], columns=feature_order)
    try:
        predicted_label = model.predict(row)[0]
        if not hasattr(model, "predict_proba"):
            raise PredictionError("The Random Forest model does not support probability estimates.")
        probability_values = model.predict_proba(row)[0]
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(f"Model prediction failed: {exc}") from exc
    if len(probability_values) != len(classes):
        raise PredictionError("Model probability output does not match its trained classes.")
    probabilities = {str(label): float(prob) for label, prob in zip(classes, probability_values)}
    phishing_probability = next(
        float(prob) for label, prob in zip(classes, probability_values)
        if _label_equal(label, phishing_label)
    )
    if _label_equal(predicted_label, phishing_label):
        predicted_class = "Phishing"
    elif _label_equal(predicted_label, legitimate_label):
        predicted_class = "Legitimate"
    else:
        raise PredictionError(f"Model returned an unmapped class: {predicted_label!r}.")

    submitted = value.strip()
    score_percent = round(phishing_probability * 100, 1)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "submitted_url": submitted,
        "normalized_url": normalized_url,
        "predicted_class": predicted_class,
        "class_probabilities": {
            "Phishing": probabilities[str(phishing_label)],
            "Legitimate": probabilities[str(legitimate_label)],
        },
        "estimated_phishing_probability": phishing_probability,
        "risk_category": _risk_category(phishing_probability),
        "analysis_timestamp": timestamp,
        "probability_caveat": PROBABILITY_CAVEAT,
        "features": features,
        "explanations": explain_features(features),
        # Backwards-compatible keys consumed by the existing Flask templates.
        "url": submitted,
        "label": "phishing" if predicted_class == "Phishing" else "likely legitimate",
        "risk_score": score_percent,
    }
