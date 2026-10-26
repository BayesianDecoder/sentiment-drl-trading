"""
train_baselines.py
──────────────────
Trains Models 1 and 2:
  Model 1: TF-IDF + Logistic Regression
  Model 2: TF-IDF + XGBoost

Both are tracked in MLflow with full metrics and artifacts.

Run:
  python src/nlp/train_baselines.py
"""

import os
import sys
import time
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

import mlflow
import mlflow.sklearn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    LABEL2ID, ID2LABEL, SEED, RESULTS_DIR, MODELS_DIR,
    MLFLOW_EXPERIMENT
)
from src.nlp.dataset import prepare_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    """Compute all metrics we care about."""
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "f1_positive": round(f1_score(y_true, y_pred, average=None,
                                       labels=[0, 1, 2])[0], 4),
        "f1_neutral":  round(f1_score(y_true, y_pred, average=None,
                                       labels=[0, 1, 2])[1], 4),
        "f1_negative": round(f1_score(y_true, y_pred, average=None,
                                       labels=[0, 1, 2])[2], 4),
    }


def save_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str) -> str:
    """
    Save confusion matrix as PNG.
    Returns path to saved file.
    """
    cm  = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=["positive", "neutral", "negative"],
        yticklabels=["positive", "neutral", "negative"],
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)

    path = os.path.join(RESULTS_DIR, f"cm_{model_name.lower().replace(' ','_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def print_results(model_name: str, metrics: dict,
                  train_time: float, n_params: int = 0) -> None:
    """Pretty print results to console."""
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  F1 Weighted : {metrics['f1_weighted']:.4f}")
    print(f"  F1 Positive : {metrics['f1_positive']:.4f}")
    print(f"  F1 Neutral  : {metrics['f1_neutral']:.4f}")
    print(f"  F1 Negative : {metrics['f1_negative']:.4f}")
    print(f"  Train time  : {train_time:.1f}s")
    if n_params:
        print(f"  Parameters  : {n_params:,}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════
# MODEL 1: TF-IDF + LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

def train_tfidf_logreg(train_df: pd.DataFrame,
                        val_df:   pd.DataFrame,
                        test_df:  pd.DataFrame) -> dict:
    """
    TF-IDF + Logistic Regression baseline.

    TF-IDF = Term Frequency × Inverse Document Frequency
    → Converts each headline into a sparse vector where:
      - common words (the, a, is) get LOW weight
      - rare but informative words (inflation, Sensex) get HIGH weight

    ngram_range=(1,2) means we capture single words AND pairs:
      "rate hike" as a bigram captures different meaning than
      "rate" and "hike" separately.

    Logistic Regression: simple linear classifier on these vectors.
    Fast to train (< 5 seconds), surprisingly competitive (F1 ≈ 0.71).
    This is your BASELINE — everything else must beat this.
    """
    model_name = "TF-IDF + LogReg"
    log.info(f"Training {model_name}...")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=model_name):

        # ── hyperparameters ──────────────────────────────────
        params = {
            "tfidf_max_features": 15_000,
            "tfidf_ngram_range":  "(1,2)",
            "tfidf_min_df":       2,
            "lr_C":               1.0,
            "lr_max_iter":        1000,
            "lr_solver":          "lbfgs",
        }
        mlflow.log_params(params)

        # ── build and train ───────────────────────────────────
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=15_000,
                ngram_range=(1, 2),
                min_df=2,
                strip_accents="unicode",
                analyzer="word",
                sublinear_tf=True    # use log(tf) instead of tf — helps
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=SEED,
            ))
        ])

        start = time.time()
        pipeline.fit(train_df["text"].tolist(), train_df["label_id"].tolist())
        train_time = time.time() - start

        # ── evaluate on val and test ──────────────────────────
        val_preds  = pipeline.predict(val_df["text"].tolist())
        test_preds = pipeline.predict(test_df["text"].tolist())

        val_metrics  = compute_metrics(val_df["label_id"].tolist(),  val_preds)
        test_metrics = compute_metrics(test_df["label_id"].tolist(), test_preds)

        # log to MLflow
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}",  v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("train_time_seconds", round(train_time, 2))

        # ── artifacts ────────────────────────────────────────
        # confusion matrix
        cm_path = save_confusion_matrix(
            test_df["label_id"].tolist(), test_preds, model_name
        )
        mlflow.log_artifact(cm_path)

        # classification report
        report = classification_report(
            test_df["label_id"].tolist(), test_preds,
            target_names=["positive", "neutral", "negative"]
        )
        report_path = os.path.join(RESULTS_DIR, "report_logreg.txt")
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        # log model
        mlflow.sklearn.log_model(pipeline, "model")

        # save locally
        model_path = os.path.join(MODELS_DIR, "tfidf_logreg.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        print_results(model_name, test_metrics, train_time)
        print(f"\n  Classification Report:\n{report}")

        run_id = mlflow.active_run().info.run_id
        log.info(f"✅ {model_name} logged. MLflow run_id: {run_id}")

        return {
            "model_name":   model_name,
            "pipeline":     pipeline,
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "train_time":   train_time,
            "run_id":       run_id,
        }


# ══════════════════════════════════════════════════════════════
# MODEL 2: TF-IDF + XGBOOST
# ══════════════════════════════════════════════════════════════

def train_tfidf_xgboost(train_df: pd.DataFrame,
                          val_df:   pd.DataFrame,
                          test_df:  pd.DataFrame) -> dict:
    """
    TF-IDF + XGBoost baseline.

    Same TF-IDF vectorizer as Model 1, but the classifier is now
    XGBoost (Extreme Gradient Boosting) — an ensemble of decision
    trees trained sequentially where each tree fixes the errors
    of the previous one.

    XGBoost generally handles:
    - Non-linear relationships between features better
    - Class imbalance better (via scale_pos_weight)
    - Overfitting better (via regularization)

    Expected: slightly better than LogReg (F1 ≈ 0.74)
    """
    if not XGB_AVAILABLE:
        log.warning("XGBoost not available. Skipping.")
        return {}

    model_name = "TF-IDF + XGBoost"
    log.info(f"Training {model_name}...")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=model_name):

        params = {
            "tfidf_max_features": 15_000,
            "tfidf_ngram_range":  "(1,2)",
            "xgb_n_estimators":   300,
            "xgb_max_depth":      6,
            "xgb_learning_rate":  0.1,
            "xgb_subsample":      0.8,
            "xgb_colsample":      0.8,
            "xgb_eval_metric":    "mlogloss",
        }
        mlflow.log_params(params)

        # ── vectorize (fit on train only) ─────────────────────
        vectorizer = TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
        X_train = vectorizer.fit_transform(train_df["text"].tolist())
        X_val   = vectorizer.transform(val_df["text"].tolist())
        X_test  = vectorizer.transform(test_df["text"].tolist())

        y_train = train_df["label_id"].tolist()
        y_val   = val_df["label_id"].tolist()
        y_test  = test_df["label_id"].tolist()

        # ── train XGBoost ─────────────────────────────────────
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )

        start = time.time()
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        train_time = time.time() - start

        log.info(f"  Best iteration: {clf.best_iteration}")
        mlflow.log_metric("best_iteration", clf.best_iteration)

        # ── evaluate ──────────────────────────────────────────
        val_preds  = clf.predict(X_val)
        test_preds = clf.predict(X_test)

        val_metrics  = compute_metrics(y_val,  val_preds)
        test_metrics = compute_metrics(y_test, test_preds)

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}",  v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("train_time_seconds", round(train_time, 2))

        # ── artifacts ────────────────────────────────────────
        cm_path = save_confusion_matrix(y_test, test_preds, model_name)
        mlflow.log_artifact(cm_path)

        # feature importance plot
        importance_path = _plot_feature_importance(
            clf, vectorizer, model_name
        )
        if importance_path:
            mlflow.log_artifact(importance_path)

        report = classification_report(
            y_test, test_preds,
            target_names=["positive", "neutral", "negative"]
        )
        report_path = os.path.join(RESULTS_DIR, "report_xgboost.txt")
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n\n{report}")
        mlflow.log_artifact(report_path)

        # save model
        model_path = os.path.join(MODELS_DIR, "tfidf_xgboost.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)

        print_results(model_name, test_metrics, train_time)
        print(f"\n  Classification Report:\n{report}")

        run_id = mlflow.active_run().info.run_id
        log.info(f"✅ {model_name} logged. MLflow run_id: {run_id}")

        return {
            "model_name":   model_name,
            "vectorizer":   vectorizer,
            "clf":          clf,
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "train_time":   train_time,
            "run_id":       run_id,
        }


def _plot_feature_importance(clf, vectorizer,
                               model_name: str,
                               top_n: int = 20) -> str:
    """Plot top N most important TF-IDF features for XGBoost."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        importances   = clf.feature_importances_

        top_idx   = np.argsort(importances)[::-1][:top_n]
        top_words = [feature_names[i] for i in top_idx]
        top_imps  = importances[top_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(top_n), top_imps[::-1], color="#6366f1")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_words[::-1], fontsize=9)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Features — {model_name}",
                     fontweight="bold")
        plt.tight_layout()

        path = os.path.join(RESULTS_DIR,
                            "feature_importance_xgboost.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        log.warning(f"Could not plot feature importance: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

def print_baseline_comparison(results: list) -> None:
    """Print a clean comparison table of all baseline models."""
    print("\n" + "="*65)
    print("BASELINE COMPARISON TABLE")
    print("="*65)
    print(f"{'Model':<25} {'F1 (weighted)':>14} {'Accuracy':>10} {'Time(s)':>9}")
    print("-"*65)
    for r in results:
        if r:
            m = r["test_metrics"]
            print(f"{r['model_name']:<25} {m['f1_weighted']:>14.4f} "
                  f"{m['accuracy']:>10.4f} {r['train_time']:>9.1f}")
    print("="*65)
    print("→ These are your baseline numbers.")
    print("  Every deep learning model (TextCNN, BiLSTM, etc.) must beat these.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS (Models 1 & 2)")
    print("="*60)

    # Step 1: Load data
    log.info("Loading data...")
    train_df, val_df, test_df, tokenizer = prepare_data()

    # Step 2: Train Model 1
    logreg_results = train_tfidf_logreg(train_df, val_df, test_df)

    # Step 3: Train Model 2
    xgb_results = train_tfidf_xgboost(train_df, val_df, test_df)

    # Step 4: Compare
    print_baseline_comparison([logreg_results, xgb_results])

    # Step 5: Instructions for next step
    print("\n✅ Baselines done! Next step:")
    print("   Run: python src/nlp/models/textcnn.py")
    print("   (This is Model 3 — your first deep learning model)")

    print("\n📊 View MLflow UI:")
    print("   mlflow ui --host 0.0.0.0 --port 5000")
    print("   Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
