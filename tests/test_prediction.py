from datetime import datetime

import numpy as np
import pytest

from src.feature_extraction import FEATURE_NAMES
from src.prediction import (
    PredictionError, load_metadata, load_model, predict_url,
)


class DummyClassifier:
    """Small estimator stand-in; no training or network access is performed."""
    classes_ = np.array(["legit", "phish"])
    feature_names_in_ = np.array(FEATURE_NAMES)

    def __init__(self, probabilities=(0.15, 0.85)):
        self.probabilities = probabilities
        self.seen_columns = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return np.array(["phish" if self.probabilities[1] >= self.probabilities[0] else "legit"])

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return np.array([self.probabilities])


def metadata():
    return {
        "input_feature_order": list(FEATURE_NAMES),
        "label_mapping": {"phishing": "phish", "legitimate": "legit"},
    }


def test_prediction_result_contains_required_fields_and_probability_order():
    model = DummyClassifier()
    result = predict_url("https://example.com/login", model=model, metadata=metadata())

    assert result["submitted_url"] == "https://example.com/login"
    assert result["predicted_class"] == "Phishing"
    assert result["class_probabilities"] == {"Phishing": 0.85, "Legitimate": 0.15}
    assert result["estimated_phishing_probability"] == 0.85
    assert result["risk_category"] == "High Risk"
    assert result["probability_caveat"]
    datetime.fromisoformat(result["analysis_timestamp"])
    assert model.seen_columns == FEATURE_NAMES


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.1, "Low Risk"), (0.5, "Medium Risk"), (0.9, "High Risk")],
)
def test_risk_category_uses_probability_presentation_bands(probability, expected):
    model = DummyClassifier(probabilities=(1 - probability, probability))
    result = predict_url("example.org", model=model, metadata=metadata())
    assert result["risk_category"] == expected


def test_invalid_url_is_rejected_before_artifact_loading():
    with pytest.raises(ValueError, match="Only HTTP and HTTPS"):
        predict_url("javascript:alert(1)")


def test_incompatible_feature_metadata_is_rejected():
    bad_metadata = metadata()
    bad_metadata["input_feature_order"] = ["different_feature"]
    with pytest.raises(PredictionError, match="does not match"):
        predict_url("https://example.com", model=DummyClassifier(), metadata=bad_metadata)


def test_missing_dataset_confirmed_label_mapping_is_rejected():
    bad_metadata = {"input_feature_order": list(FEATURE_NAMES)}
    with pytest.raises(PredictionError, match="label_mapping"):
        predict_url("https://example.com", model=DummyClassifier(), metadata=bad_metadata)


def test_load_model_reports_missing_artifact(tmp_path):
    with pytest.raises(PredictionError, match="Trained model is missing"):
        load_model(tmp_path / "missing.pkl")


def test_load_metadata_reports_missing_or_invalid_artifact(tmp_path):
    with pytest.raises(PredictionError, match="Feature metadata is missing"):
        load_metadata(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    with pytest.raises(PredictionError, match="Could not read feature metadata"):
        load_metadata(broken)
