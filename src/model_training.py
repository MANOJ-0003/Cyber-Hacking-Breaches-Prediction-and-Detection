"""Leakage-safe Random Forest training for offline URL lexical screening.

The supplied 87-column dataset also includes page-content and external-service
signals that cannot be reproduced without visiting URLs. When raw URLs are
available, this trainer derives the shared offline lexical feature schema from
those URLs and trains a compatible, explicitly scoped model. It never pretends
that the result uses all 87 dataset features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .data_preprocessing import EXPECTED_INPUT_FEATURES, build_preprocessor, inspect_csv, load_dataset
from .feature_extraction import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "dataset_phishing.csv"
DEFAULT_MODEL = ROOT / "models" / "random_forest_model.pkl"
DEFAULT_METADATA = ROOT / "models" / "feature_metadata.json"
DEFAULT_REPORTS = ROOT / "reports"
RANDOM_STATE = 42


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _positive_label(classes, requested=None):
    if requested is not None:
        if requested not in classes:
            raise ValueError(f"positive_label {requested!r} is not present in target labels {list(classes)!r}.")
        return requested
    phishing = {"phishing", "phish", "malicious"}
    matches = [value for value in classes if str(value).strip().lower() in phishing]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        "Cannot identify the phishing label from the dataset's encoding. "
        f"Observed labels: {list(classes)!r}. Supply positive_label explicitly "
        "after confirming the dataset's label mapping."
    )


def _check_dataset_contract(dataset_path: Path, inspection: dict) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    if inspection["shape"]["rows"] < 100:
        raise ValueError("Training requires at least 100 labeled URL records.")
    frame = pd.read_csv(dataset_path, low_memory=False)
    url_columns = inspection.get("raw_url_columns", [])
    if url_columns:
        url_column = url_columns[0]
        target_column = inspection["target_column"]
        if frame[url_column].isna().any() or frame[target_column].isna().any():
            raise ValueError("Training paused: raw URLs and target labels must be present for every training row.")
        rows = []
        encoded_whitespace_rows = []
        for row_number, url in enumerate(frame[url_column].astype(str), start=2):
            if any(character.isspace() for character in url):
                # Dataset strings can contain literal Unicode spaces. Convert
                # them to URL percent escapes, matching how a browser-safe URL
                # must be submitted to the scanner.
                url = "".join(quote(character, safe="") if character.isspace() else character for character in url)
                encoded_whitespace_rows.append(row_number)
            try:
                rows.append(extract_features(url))
            except ValueError as exc:
                raise ValueError(f"Training paused: invalid URL in CSV row {row_number}: {exc}") from exc
        features = pd.DataFrame(rows, columns=FEATURE_NAMES)
        target = frame[target_column].copy()
        if target.nunique() != 2:
            raise ValueError(f"Expected two target classes; found {target.nunique()}: {target.unique().tolist()!r}.")
        return features, target, list(FEATURE_NAMES), {
            "input_rows": len(frame),
            "rows_with_whitespace_percent_encoded": encoded_whitespace_rows,
            "precomputed_features_used": False,
        }

    if inspection.get("extracted_feature_count") != EXPECTED_INPUT_FEATURES:
        raise ValueError("Training requires raw URLs or a documented extracted feature schema.")
    if inspection.get("numeric_feature_count") != EXPECTED_INPUT_FEATURES:
        raise ValueError("Training paused: all 87 extracted model features must be numeric or have documented preprocessing.")
    features, target = load_dataset(dataset_path, inspection["target_column"])
    extractor_names = list(FEATURE_NAMES)
    if len(extractor_names) != EXPECTED_INPUT_FEATURES:
        raise ValueError("Training paused: no raw URL values or compatible documented 87-feature extractor are available.")
    if len(set(extractor_names)) != len(extractor_names):
        raise ValueError("Training paused: feature_extraction.py contains duplicate feature names.")
    missing = [name for name in extractor_names if name not in features.columns]
    extra = [name for name in features.columns if name not in extractor_names]
    if missing or extra:
        raise ValueError(
            "Training paused: dataset columns and extractor schema differ. "
            f"Missing from dataset: {missing}; not defined by extractor: {extra}."
        )
    # This establishes the canonical order for training and records it in metadata.
    features = features.loc[:, extractor_names]
    if target.nunique() != 2:
        raise ValueError(f"Expected two target classes; found {target.nunique()}: {target.unique().tolist()!r}.")
    return features, target, extractor_names, {
        "input_rows": len(features),
        "rows_with_whitespace_percent_encoded": [],
        "precomputed_features_used": True,
    }


def _write_confusion_plot(matrix: np.ndarray, labels: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted label", ylabel="Actual label",
           title="Confusion matrix (held-out test set)")
    ax.tick_params(axis="x", labelrotation=25)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_importance_plot(names: list[str], values: np.ndarray, output: Path) -> None:
    order = np.argsort(values)[-20:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([names[i] for i in order], values[order], color="#14765f")
    ax.set(xlabel="Random Forest feature importance", title="Top feature importances")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def train(
    dataset_path: str | Path = DEFAULT_DATASET,
    model_path: str | Path = DEFAULT_MODEL,
    metadata_path: str | Path = DEFAULT_METADATA,
    reports_dir: str | Path = DEFAULT_REPORTS,
    target_column: str | None = None,
    positive_label=None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    tune: bool = True,
) -> dict:
    """Train, evaluate and save a pipeline; fail before writing artifacts if incompatible.

    The preprocessing steps are fitted inside the estimator pipeline, so CV folds
    and the final model learn imputation/encoding values only from their training
    data. The 20% stratified test partition is never used to tune parameters.
    """
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    reports_dir = Path(reports_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Training paused: provide the real labeled CSV at "
            "dataset/phishing_dataset.csv (approximately 5,457 rows, 87 input "
            "features, and one target column)."
        )
    inspection = inspect_csv(dataset_path, target_column)
    X, y, feature_names, dataset_notes = _check_dataset_contract(dataset_path, inspection)
    counts = y.value_counts()
    if len(counts) != 2 or int(counts.min()) < 2:
        raise ValueError("Stratified 80:20 training requires two classes with at least two labeled rows each.")
    positive = _positive_label(y.unique().tolist(), positive_label)
    # Use exact class membership and avoid assuming that 1 always means phishing.
    scorer = make_scorer(f1_score, pos_label=positive, zero_division=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(
            random_state=random_state, class_weight="balanced", n_jobs=-1,
        )),
    ])
    min_train_class = int(y_train.value_counts().min())
    search_details = {"used": False, "cv_folds": 0, "best_params": {}}
    if tune and min_train_class >= 2:
        folds = min(5, min_train_class)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            pipeline,
            param_grid={
                "classifier__n_estimators": [150, 300],
                "classifier__max_features": ["sqrt", 0.5],
                "classifier__min_samples_leaf": [1, 2],
            },
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        search_details = {
            "used": True,
            "cv_folds": folds,
            "scoring": f"F1 for phishing label {positive!r}",
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
        }
    else:
        fitted = pipeline.fit(X_train, y_train)

    predicted_train = fitted.predict(X_train)
    predicted_test = fitted.predict(X_test)
    labels = list(fitted.named_steps["classifier"].classes_)
    precision = precision_score(y_test, predicted_test, pos_label=positive, zero_division=0)
    recall = recall_score(y_test, predicted_test, pos_label=positive, zero_division=0)
    f1 = f1_score(y_test, predicted_test, pos_label=positive, zero_division=0)
    matrix = confusion_matrix(y_test, predicted_test, labels=labels)
    report_dict = classification_report(y_test, predicted_test, labels=labels,
                                        target_names=[str(item) for item in labels],
                                        output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predicted_test, labels=labels,
                                        target_names=[str(item) for item in labels], zero_division=0)

    preprocessor = fitted.named_steps["preprocessing"]
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    classifier = fitted.named_steps["classifier"]
    importances = classifier.feature_importances_
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_path = reports_dir / "confusion_matrix.png"
    importance_path = reports_dir / "feature_importance.png"
    evaluation_path = reports_dir / "model_evaluation.txt"
    _write_confusion_plot(matrix, [str(item) for item in labels], confusion_path)
    _write_importance_plot(transformed_names, importances, importance_path)

    metrics = {
        "training_accuracy": float(accuracy_score(y_train, predicted_train)),
        "testing_accuracy": float(accuracy_score(y_test, predicted_test)),
        "precision": float(precision), "recall": float(recall), "f1_score": float(f1),
        "positive_label": _plain(positive),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [_plain(item) for item in labels],
        "classification_report": _plain(report_dict),
    }
    evaluation_path.write_text(
        "Random Forest evaluation\n"
        f"Dataset: {dataset_path.name}\nRows: {len(X)}\nInput features: {len(feature_names)}\n"
        f"Split: stratified {1-test_size:.0%}/{test_size:.0%}\nRandom state: {random_state}\n"
        f"Positive (phishing) label: {positive!r}\n\n"
        f"Training accuracy: {metrics['training_accuracy']:.4f}\n"
        f"Testing accuracy: {metrics['testing_accuracy']:.4f}\n"
        f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\n\n"
        f"Parameter search: {json.dumps(search_details, indent=2, default=str)}\n\n"
        f"Confusion matrix (labels {labels!r}):\n{matrix}\n\n"
        f"Classification report:\n{report_text}\n",
        encoding="utf-8",
    )
    metadata = {
        "model_type": "sklearn.pipeline.Pipeline(RandomForestClassifier)",
        "model_scope": "offline_url_lexical_features",
        "dataset_precomputed_features_used": False,
        "dataset_preprocessing_notes": dataset_notes,
        "scope_note": (
            "Trained on lexical features recomputed from raw URL text using the same offline extractor used at inference. "
            "The dataset's 87 precomputed URL, page-content, and external-service features were not used."
        ),
        "dataset_file": dataset_path.name,
        "target_column": inspection["target_column"],
        "input_feature_names": feature_names,
        "input_feature_order": feature_names,
        "transformed_feature_names": transformed_names,
        "label_values": [_plain(item) for item in labels],
        "label_mapping": {
            "phishing": _plain(positive),
            "legitimate": _plain(next(item for item in labels if item != positive)),
            "source": "confirmed by semantic target value or explicit positive_label argument",
        },
        "positive_label": _plain(positive),
        "preprocessing": {
            "numeric_missing_values": "median imputer fitted on training data",
            "categorical_missing_values": "most-frequent imputer fitted on training data",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore') fitted on training data",
            "pipeline_included_in_model_file": True,
        },
        "split": {"test_size": test_size, "stratified": True, "random_state": random_state},
        "parameter_search": search_details,
        "evaluation": metrics,
        "feature_importance": {
            name: float(value) for name, value in zip(transformed_names, importances)
        },
        "artifacts": {
            "model": str(model_path),
            "confusion_matrix": str(confusion_path),
            "feature_importance": str(importance_path),
            "evaluation_report": str(evaluation_path),
        },
    }
    # Persist a single complete preprocessing + classifier pipeline for inference.
    joblib.dump(fitted, model_path)
    metadata_path.write_text(json.dumps(_plain(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    try:
        result = train()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    evaluation = result["evaluation"]
    print("Training complete; metrics calculated from the held-out dataset.")
    print(f"Training accuracy: {evaluation['training_accuracy']:.3f}")
    print(f"Testing accuracy: {evaluation['testing_accuracy']:.3f}")
    print(f"Precision / recall / F1: {evaluation['precision']:.3f} / {evaluation['recall']:.3f} / {evaluation['f1_score']:.3f}")
