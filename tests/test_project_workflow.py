"""Integration checks; any injected prediction below is test-only wiring data."""
from pathlib import Path
import re

import pytest

from src.data_preprocessing import inspect_csv, load_dataset, prepare_train_test
from src.feature_extraction import extract_features, normalize_url
from src.model_training import train
from src.prediction import PredictionError, load_model, predict_url

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "phishing_dataset.csv"
FULL_DATASET = ROOT / "dataset" / "dataset_phishing.csv"


def test_imports_and_real_dataset_loading():
    X, y = load_dataset(DATASET)
    report = inspect_csv(DATASET)
    assert report["shape"] == {"rows": 60, "columns": 2}
    assert report["target_column"] == "label"
    assert report["actual_label_values"] == [0, 1]
    assert X.shape == (60, 13)
    assert y.value_counts().to_dict() == {0: 30, 1: 30}


def test_preprocessing_and_stratified_split_preserve_rows():
    split = prepare_train_test(DATASET, test_size=0.2, random_state=42)
    assert split["stratified"] is True
    assert len(split["X_train"]) == 48
    assert len(split["X_test"]) == 12
    assert split["y_train"].value_counts().to_dict() == {0: 24, 1: 24}
    transformed = split["preprocessor"].fit_transform(split["X_train"])
    assert transformed.shape == (48, 13)


def test_downloaded_dataset_has_url_plus_87_numeric_features():
    report = inspect_csv(FULL_DATASET)
    assert report["shape"] == {"rows": 11430, "columns": 89}
    assert report["target_column"] == "status"
    assert report["input_feature_count"] == 88  # 87 features plus the raw URL column
    assert report["raw_url_columns"] == ["url"]
    assert report["extracted_feature_count"] == 87
    assert report["numeric_feature_count"] == 87
    assert report["class_distribution"] == {"legitimate": 5715, "phishing": 5715}
    assert report["total_missing_values"] == 0
    assert report["duplicate_records"] == 0


def test_downloaded_dataset_preprocessing_drops_url_identifier_and_stratifies():
    split = prepare_train_test(FULL_DATASET, test_size=0.2, random_state=42)
    assert len(split["X_train"]) == 9144
    assert len(split["X_test"]) == 2286
    assert len(split["feature_columns"]) == 87
    assert "url" not in split["feature_columns"]
    assert split["y_train"].value_counts().to_dict() == {"legitimate": 4572, "phishing": 4572}
    assert split["y_test"].value_counts().to_dict() == {"legitimate": 1143, "phishing": 1143}


def test_expected_url_cases_validate_and_extract_features():
    legitimate_example = "https://www.wikipedia.org"
    suspicious_example = "https://account-check.example.com/login?session=demo"
    special_example = "http://user@example.com:8080/a?x=1&y=2#part"
    long_label_example = "https://" + ("a" * 60) + ".example.com/path"

    assert extract_features(legitimate_example)["uses_https"] == 1
    suspicious_features = extract_features(suspicious_example)
    assert suspicious_features["has_suspicious_word"] == 1
    # This test does not claim that suspicious URL text must predict phishing.
    assert extract_features(special_example)["num_at"] == 1
    assert extract_features(special_example)["has_port"] == 1
    assert extract_features(long_label_example)["hostname_length"] > 50
    for invalid in ("", "javascript:alert(1)", "https:///missing-host", "http://example.com:99999"):
        with pytest.raises(ValueError):
            normalize_url(invalid)


def test_small_demo_dataset_is_not_mistaken_for_the_training_dataset(tmp_path):
    model_path = tmp_path / "random_forest_model.pkl"
    metadata_path = tmp_path / "feature_metadata.json"
    with pytest.raises(ValueError, match="at least 100 labeled URL records"):
        train(dataset_path=DATASET, model_path=model_path, metadata_path=metadata_path,
              reports_dir=tmp_path / "reports")
    assert not model_path.exists()
    assert not metadata_path.exists()


def test_full_dataset_trains_matching_offline_url_model(tmp_path):
    model_path = tmp_path / "random_forest_model.pkl"
    metadata_path = tmp_path / "feature_metadata.json"
    metadata = train(dataset_path=FULL_DATASET, model_path=model_path, metadata_path=metadata_path,
                     reports_dir=tmp_path / "reports", tune=False, random_state=42)
    assert model_path.exists() and metadata_path.exists()
    assert metadata["model_scope"] == "offline_url_lexical_features"
    assert metadata["dataset_precomputed_features_used"] is False
    assert len(metadata["input_feature_order"]) == 13
    assert metadata["evaluation"]["testing_accuracy"] >= 0
    from src.prediction import load_artifacts
    model, saved_metadata = load_artifacts(model_path, metadata_path)
    assert len(model.feature_names_in_) == 13
    assert saved_metadata["label_mapping"]["phishing"] == "phishing"
    prediction = predict_url("https://example.com", model=model, metadata=saved_metadata)
    assert prediction["predicted_class"] in {"Phishing", "Legitimate"}
    assert 0 <= prediction["estimated_phishing_probability"] <= 1


def test_prediction_refuses_to_score_without_artifacts(tmp_path):
    with pytest.raises(PredictionError, match="Trained model is missing"):
        load_model(tmp_path / "not-trained.pkl")
    with pytest.raises(PredictionError, match="Trained model is missing"):
        predict_url("https://www.wikipedia.org", metadata={
            "input_feature_order": list(extract_features("example.org")),
            "label_mapping": {"phishing": "phishing", "legitimate": "legitimate"},
        }, model_path=tmp_path / "not-trained.pkl")


@pytest.fixture
def web_client(tmp_path, monkeypatch):
    import app as flask_app

    monkeypatch.setattr(flask_app, "DATABASE_PATH", tmp_path / "scans.db")
    monkeypatch.setattr(flask_app, "FEATURE_METADATA_PATH", tmp_path / "missing-metadata.json")
    flask_app.init_db()
    flask_app.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return flask_app, flask_app.app.test_client()


def csrf_value(response):
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match, "Rendered form should contain a CSRF token"
    return match.group(1).decode()


def test_flask_startup_navigation_and_empty_chart_states(web_client):
    flask_app, client = web_client
    paths = ["/", "/scan", "/dashboard", "/threat-analytics", "/history", "/model-performance", "/about"]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        assert b"Sentinel" in response.data or b"SENTINEL" in response.data
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert b"No completed scan data" in client.get("/dashboard").data
    assert b"No trained model metrics" in client.get("/model-performance").data
    assert b"No feature importance data" in client.get("/model-performance").data
    assert client.get("/result/987654").status_code == 404


def test_invalid_submission_and_missing_model_do_not_create_scan_rows(web_client, monkeypatch):
    flask_app, client = web_client
    form = client.get("/scan")
    token = csrf_value(form)
    invalid = client.post("/scan", data={"csrf_token": token, "url": "javascript:alert(1)"})
    assert invalid.status_code == 400

    def missing_model(_url):
        raise PredictionError("fixture: missing model")

    monkeypatch.setattr(flask_app, "predict_url", missing_model)
    form = client.get("/scan")
    unavailable = client.post("/scan", data={"csrf_token": csrf_value(form), "url": "https://example.com"})
    assert unavailable.status_code == 503
    assert b"offline URL model is not trained yet" in unavailable.data
    with flask_app.database() as db:
        assert db.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 0


def test_csrf_and_oversized_or_empty_input_are_rejected(web_client):
    _, client = web_client
    assert client.post("/scan", data={"url": "https://example.com"}).status_code == 400
    form = client.get("/scan")
    token = csrf_value(form)
    empty = client.post("/scan", data={"csrf_token": token, "url": ""})
    assert empty.status_code == 400
    huge = client.post("/scan", data={"csrf_token": token, "url": "a" * 9000})
    assert huge.status_code == 413


def test_scan_storage_search_dashboard_and_chart_use_sqlite_rows(web_client, monkeypatch):
    flask_app, client = web_client
    fixture_prediction = {
        "submitted_url": "https://example.com/test?x=1",
        "predicted_class": "Phishing",
        "estimated_phishing_probability": 0.73,
        "risk_category": "High Risk",
        "analysis_timestamp": "2026-01-02T03:04:05+00:00",
        "probability_caveat": "Test fixture only; no production prediction is created.",
        "risk_score": 73.0,
        "explanations": [],
    }
    monkeypatch.setattr(flask_app, "predict_url", lambda _url: fixture_prediction)
    form = client.get("/scan")
    response = client.post("/scan", data={
        "csrf_token": csrf_value(form), "url": fixture_prediction["submitted_url"],
    })
    assert response.status_code == 200
    assert b"PHISHING DETECTED" in response.data

    dashboard = client.get("/dashboard")
    assert b">1<" in dashboard.data
    assert b">73.0%<" in dashboard.data
    assert b"Daily scan activity" in dashboard.data
    assert b"CLASS DISTRIBUTION" in dashboard.data
    history = client.get("/history?q=example.com")
    assert b"example.com/test" in history.data

    with flask_app.database() as db:
        row = db.execute("SELECT * FROM scans").fetchone()
    assert row["prediction"] == "Phishing"
    assert row["phishing_probability"] == pytest.approx(0.73)
    assert row["risk_category"] == "High Risk"


def test_saved_model_metadata_drives_metrics_and_feature_charts(web_client, monkeypatch):
    flask_app, client = web_client
    test_metadata = {
        "evaluation": {"training_accuracy": 0.8, "testing_accuracy": 0.7,
                       "precision": 0.75, "recall": 0.65, "f1_score": 0.69},
        "feature_importance": {"url_length": 0.4, "domain_depth": 0.2},
    }
    monkeypatch.setattr(flask_app, "read_model_metadata", lambda: test_metadata)
    response = client.get("/model-performance")
    assert response.status_code == 200
    assert b"80.0%" in response.data and b"70.0%" in response.data
    assert b"url_length" in response.data and b"domain_depth" in response.data


def test_frontend_resources_include_responsive_layout_and_scan_animation():
    base = (ROOT / "templates" / "base.html").read_text(encoding="utf-8")
    stylesheet = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    scanner = (ROOT / "templates" / "scan.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert 'name="viewport"' in base
    assert "@media(max-width:760px)" in stylesheet
    assert "prefers-reduced-motion:reduce" in stylesheet
    assert "data-scan-progress" in scanner and "spinner" in scanner
    assert "data-scan-form" in script and "classList.add(\"visible\")" in script
