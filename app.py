from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash

from feature_extraction import (
    extract_features,
    normalize_url,
    is_ip_url,
    has_at_symbol,
    is_shortener,
)

db = SQLAlchemy()
login_manager = LoginManager()


@dataclass
class PredictionResult:
    label: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(60), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True, nullable=False)

    raw_url = db.Column(db.String(300), nullable=False)
    normalized_url = db.Column(db.String(400), nullable=False)

    label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    reason = db.Column(db.String(200), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Train using train_model.py first.")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["columns"]


def predict_url(url: str, model, model_columns) -> Tuple[Optional[PredictionResult], Optional[str]]:
    # Hard rules first
    if is_ip_url(url):
        return PredictionResult("Phishing Website", None, "URL uses IP address"), None
    if has_at_symbol(url):
        return PredictionResult("Phishing Website", None, "URL contains '@'"), None
    if is_shortener(url):
        return PredictionResult("Phishing Website", None, "URL is a shortener"), None

    try:
        feats = extract_features(url, model_columns)
        X = pd.DataFrame([feats], columns=model_columns)

        y = int(model.predict(X)[0])  # -1 phishing, 1 legit
        label = "Legitimate Website" if y == 1 else "Phishing Website"

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 2)

        return PredictionResult(label, confidence, None), None
    except Exception as e:
        return None, f"Error while processing URL: {e}"


def save_history(user_id: int, raw_url: str, normalized_url: str, result: PredictionResult) -> None:
    raw_url = (raw_url or "")[:300]
    normalized_url = (normalized_url or "")[:400]
    reason = (result.reason or "")[:200] if result.reason else None

    row = History(
        user_id=user_id,
        raw_url=raw_url,
        normalized_url=normalized_url,
        label=result.label,
        confidence=result.confidence,
        reason=reason,
    )
    db.session.add(row)
    db.session.commit()


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-in-env")

    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///phishing_app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "login"

    model, model_columns = load_model("model.pkl")

    with app.app_context():
        db.create_all()

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.query.get(int(user_id))

    @app.route("/", methods=["GET"])
    def home():
        return redirect(url_for("dashboard" if current_user.is_authenticated else "login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = (request.form.get("username") or "").strip().lower()
            password = request.form.get("password") or ""
            confirm = request.form.get("confirm") or ""

            if len(username) < 3:
                flash("Username must be at least 3 characters.", "error")
                return render_template("auth_register.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters.", "error")
                return render_template("auth_register.html")

            if password != confirm:
                flash("Passwords do not match.", "error")
                return render_template("auth_register.html")

            if User.query.filter_by(username=username).first():
                flash("Username already exists.", "error")
                return render_template("auth_register.html")

            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash("Account created. Please login.", "success")
            return redirect(url_for("login"))

        return render_template("auth_register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = (request.form.get("username") or "").strip().lower()
            password = request.form.get("password") or ""

            user = User.query.filter_by(username=username).first()
            if not user or not user.check_password(password):
                flash("Invalid username or password.", "error")
                return render_template("auth_login.html")

            login_user(user)
            return redirect(url_for("dashboard"))

        return render_template("auth_login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.route("/dashboard", methods=["GET", "POST"])
    @login_required
    def dashboard():
        result: Optional[PredictionResult] = None
        error: Optional[str] = None

        if request.method == "POST":
            raw_url = (request.form.get("url") or "").strip()
            url = normalize_url(raw_url)

            if not url:
                error = "Please enter a valid URL (example: https://youtube.com)."
            else:
                result, error = predict_url(url, model, model_columns)
                if result is not None:
                    save_history(current_user.id, raw_url, url, result)

        q = (request.args.get("q") or "").strip()
        label_filter = (request.args.get("label") or "").strip()

        history_query = History.query.filter_by(user_id=current_user.id)

        if label_filter in {"Legitimate Website", "Phishing Website"}:
            history_query = history_query.filter(History.label == label_filter)

        if q:
            like = f"%{q}%"
            history_query = history_query.filter(
                (History.raw_url.ilike(like)) | (History.normalized_url.ilike(like))
            )

        history = history_query.order_by(History.created_at.desc()).limit(50).all()

        total = History.query.filter_by(user_id=current_user.id).count()
        phishing_count = History.query.filter_by(user_id=current_user.id, label="Phishing Website").count()
        legit_count = History.query.filter_by(user_id=current_user.id, label="Legitimate Website").count()

        return render_template(
            "dashboard.html",
            result=result,
            error=error,
            history=history,
            stats={"total": total, "phishing": phishing_count, "legit": legit_count},
            q=q,
            label_filter=label_filter,
        )

    @app.route("/clear-history", methods=["POST"])
    @login_required
    def clear_history():
        History.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash("History cleared.", "success")
        return redirect(url_for("dashboard"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)