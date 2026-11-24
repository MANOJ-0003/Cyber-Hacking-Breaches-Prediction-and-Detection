"""Local Flask application for offline phishing URL screening."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import json
import logging
import secrets
import sqlite3

from flask import (
    Flask, abort, flash, redirect, render_template, request, session, url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import DATABASE_PATH, FEATURE_METADATA_PATH, MAX_CONTENT_LENGTH, SECRET_KEY
from src.feature_extraction import normalize_url
from src.prediction import PredictionError, predict_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # Local HTTP at 127.0.0.1; use HTTPS if deployed.
)


@contextmanager
def database():
    """Open a short-lived SQLite connection and always commit/close safely."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with database() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                submitted_url TEXT NOT NULL,
                prediction TEXT NOT NULL CHECK (prediction IN ('Phishing', 'Legitimate')),
                phishing_probability REAL CHECK (
                    phishing_probability IS NULL OR
                    (phishing_probability >= 0 AND phishing_probability <= 1)
                ),
                risk_category TEXT NOT NULL CHECK (
                    risk_category IN ('Low Risk', 'Medium Risk', 'High Risk')
                ),
                timestamp TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp DESC)")


def save_scan(result: dict) -> int:
    """Persist only a successful prediction using bound SQL parameters."""
    prediction = result.get("predicted_class")
    probability = result.get("estimated_phishing_probability")
    risk = result.get("risk_category")
    if prediction not in ("Phishing", "Legitimate"):
        raise ValueError("Prediction result contains an unknown class.")
    if probability is not None and not 0 <= float(probability) <= 1:
        raise ValueError("Prediction probability is outside its valid range.")
    if risk not in ("Low Risk", "Medium Risk", "High Risk"):
        raise ValueError("Prediction result contains an unknown risk category.")
    timestamp = result.get("analysis_timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database() as db:
        cursor = db.execute(
            "INSERT INTO scans (submitted_url, prediction, phishing_probability, risk_category, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (result["submitted_url"], prediction, probability, risk, timestamp),
        )
        return int(cursor.lastrowid)


@app.context_processor
def inject_csrf_token():
    def csrf_token():
        token = session.get("csrf_token")
        if token is None:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token
    return {"csrf_token": csrf_token}


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        expected = session.get("csrf_token")
        supplied = request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            abort(400, description="The form expired or could not be verified. Reload the page and try again.")


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; form-action 'self'; base-uri 'self'; frame-ancestors 'none'",
    )
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/scan")
def scan_page():
    return render_template("scan.html")


@app.post("/scan")
@app.post("/analyze")  # Keep the earlier form endpoint usable.
def scan_url():
    submitted_url = request.form.get("url", "")
    try:
        normalize_url(submitted_url)
        result = predict_url(submitted_url)
    except ValueError as exc:
        # URL validation feedback is safe and useful; model/artifact details stay server-side.
        if isinstance(exc, PredictionError):
            logger.warning("Scan unavailable: %s", exc)
            flash(
                "Scan unavailable: the compatible offline URL model is not trained yet. "
                "Run the training command from the project README, then restart the app.",
                "error",
            )
        else:
            flash(str(exc), "error")
        return render_template("scan.html", submitted_url=submitted_url.strip()), 400 if not isinstance(exc, PredictionError) else 503
    except Exception:
        logger.exception("Unexpected URL scan failure")
        flash("The URL could not be analyzed right now. Please try again later.", "error")
        return render_template("scan.html", submitted_url=submitted_url.strip()), 503

    try:
        scan_id = save_scan(result)
    except (KeyError, TypeError, ValueError):
        logger.exception("Prediction result could not be stored")
        flash("The analysis completed, but its result could not be saved.", "error")
        return render_template("scan.html", submitted_url=submitted_url.strip()), 500
    return render_template("result.html", result=result, scan_id=scan_id)


@app.get("/result/<int:scan_id>")
def result_page(scan_id: int):
    with database() as db:
        row = db.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    if row is None:
        abort(404)
    result = {
        "submitted_url": row["submitted_url"],
        "predicted_class": row["prediction"],
        "estimated_phishing_probability": row["phishing_probability"],
        "risk_category": row["risk_category"],
        "analysis_timestamp": row["timestamp"],
        "probability_caveat": "Model probabilities are estimates, not guaranteed security scores.",
        "url": row["submitted_url"],
        "label": "phishing" if row["prediction"] == "Phishing" else "likely legitimate",
        "risk_score": round(row["phishing_probability"] * 100, 1) if row["phishing_probability"] is not None else None,
        "explanations": [],
    }
    return render_template("result.html", result=result, scan_id=scan_id)


@app.get("/dashboard")
def dashboard():
    with database() as db:
        total = db.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        phishing = db.execute("SELECT COUNT(*) FROM scans WHERE prediction = ?", ("Phishing",)).fetchone()[0]
        legitimate = db.execute("SELECT COUNT(*) FROM scans WHERE prediction = ?", ("Legitimate",)).fetchone()[0]
        recent = db.execute("SELECT * FROM scans ORDER BY timestamp DESC, scan_id DESC LIMIT 8").fetchall()
        daily_rows = db.execute(
            "SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS total "
            "FROM scans WHERE substr(timestamp, 1, 10) >= ? GROUP BY day",
            ((datetime.now(timezone.utc).date() - timedelta(days=6)).isoformat(),),
        ).fetchall()
    phishing_percent = round(100 * phishing / total, 1) if total else 0.0
    daily_lookup = {row["day"]: row["total"] for row in daily_rows}
    dates = [datetime.now(timezone.utc).date() - timedelta(days=offset) for offset in range(6, -1, -1)]
    daily = [{"day": item.strftime("%a"), "date": item.isoformat(), "count": daily_lookup.get(item.isoformat(), 0)} for item in dates]
    daily_max = max((item["count"] for item in daily), default=0)
    statistics = {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "phishing_percent": phishing_percent,
        "legitimate_percent": round(100 * legitimate / total, 1) if total else 0.0,
    }
    distribution = {
        "phishing": phishing,
        "legitimate": legitimate,
        "phishing_percent": phishing_percent,
        "legitimate_percent": round(100 * legitimate / total, 1) if total else 0.0,
    }
    return render_template("dashboard.html", stats=statistics, recent=recent,
                           distribution=distribution, daily=daily, daily_max=daily_max)


def read_model_metadata() -> dict | None:
    try:
        with open(FEATURE_METADATA_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/threat-analytics")
def threat_analytics():
    with database() as db:
        phishing = db.execute("SELECT COUNT(*) FROM scans WHERE prediction = ?", ("Phishing",)).fetchone()[0]
        legitimate = db.execute("SELECT COUNT(*) FROM scans WHERE prediction = ?", ("Legitimate",)).fetchone()[0]
        daily_rows = db.execute(
            "SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS total "
            "FROM scans WHERE substr(timestamp, 1, 10) >= ? GROUP BY day",
            ((datetime.now(timezone.utc).date() - timedelta(days=13)).isoformat(),),
        ).fetchall()
    lookup = {row["day"]: row["total"] for row in daily_rows}
    dates = [datetime.now(timezone.utc).date() - timedelta(days=offset) for offset in range(13, -1, -1)]
    daily = [{"day": item.strftime("%d %b"), "date": item.isoformat(), "count": lookup.get(item.isoformat(), 0)} for item in dates]
    daily_max = max((item["count"] for item in daily), default=0)
    total = phishing + legitimate
    distribution = {
        "phishing": phishing,
        "legitimate": legitimate,
        "total": total,
        "phishing_percent": round(100 * phishing / total, 1) if total else 0.0,
        "legitimate_percent": round(100 * legitimate / total, 1) if total else 0.0,
    }
    return render_template("threat_analytics.html", daily=daily, daily_max=daily_max, distribution=distribution)


@app.get("/model-performance")
def model_performance():
    metadata = read_model_metadata()
    evaluation = metadata.get("evaluation", {}) if metadata else {}
    metric_names = [
        ("training_accuracy", "Training accuracy"),
        ("testing_accuracy", "Test accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 score"),
    ]
    metrics = []
    for key, label in metric_names:
        value = evaluation.get(key)
        if isinstance(value, (int, float)):
            metrics.append({"name": label, "value": max(0.0, min(1.0, float(value)))})
    importance_map = metadata.get("feature_importance", {}) if metadata else {}
    features = []
    if isinstance(importance_map, dict):
        pairs = [(str(name), float(value)) for name, value in importance_map.items()
                 if isinstance(value, (int, float)) and value >= 0]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        maximum = max((value for _, value in pairs[:10]), default=0)
        features = [{"name": name, "value": value,
                     "width": (value / maximum * 100) if maximum else 0}
                    for name, value in pairs[:10]]
    return render_template("model_performance.html", metrics=metrics, features=features,
                           metadata_available=metadata is not None)


@app.get("/history")
def history():
    query = request.args.get("q", "").strip()[:200]
    with database() as db:
        if query:
            escaped = query.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            rows = db.execute(
                "SELECT * FROM scans WHERE submitted_url LIKE ? ESCAPE '!' "
                "ORDER BY timestamp DESC, scan_id DESC LIMIT 250",
                (f"%{escaped}%",),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM scans ORDER BY timestamp DESC, scan_id DESC LIMIT 250"
            ).fetchall()
    return render_template("history.html", rows=rows, query=query)


@app.get("/about")
def about():
    return render_template("about.html")


@app.errorhandler(400)
def bad_request(error):
    return render_template("error.html", code=400, message=getattr(error, "description", "Request could not be verified.")), 400


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404, message="The page you requested was not found."), 404


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return render_template("error.html", code=413, message="The request is too large."), 413


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
