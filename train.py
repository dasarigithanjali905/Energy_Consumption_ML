import logging
import os
from typing import Dict, Tuple

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.multiclass import type_of_target
from xgboost import XGBClassifier, XGBRegressor

from config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    TEXT_COLUMN,
    ensure_directories,
)
from utils import build_preprocessing_pipeline, load_dataset, save_json, save_preprocessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _train_models(
    X: np.ndarray, y: np.ndarray, task_type: str
) -> Tuple[str, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, object]]:
    models: Dict[str, object] = {}
    scores: Dict[str, float] = {}
    reports: Dict[str, Dict[str, float]] = {}
    conf_matrices: Dict[str, np.ndarray] = {}

    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, n_jobs=-1, solver="lbfgs"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
            ),
        }

        for name, model in models.items():
            logger.info("Training classification model: %s", name)
            model.fit(X, y)
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            scores[name] = acc
            reports[name] = classification_report(y, y_pred, output_dict=True)
            conf_matrices[name] = confusion_matrix(y, y_pred)
            logger.info("Model %s accuracy: %.4f", name, acc)

    else:
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
            ),
            "xgboost_regressor": XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
            ),
        }

        for name, model in models.items():
            logger.info("Training regression model: %s", name)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            scores[name] = r2
            logger.info("Model %s R2 score: %.4f", name, r2)

    best_model_name = max(scores, key=scores.get)
    logger.info("Best model selected: %s", best_model_name)
    return best_model_name, scores, reports, conf_matrices, models


def train_and_save(data_path: str = DATA_PATH) -> None:
    ensure_directories()

    if not os.path.exists(data_path):
        logger.error("Dataset not found at %s. Training aborted.", data_path)
        return

    df = load_dataset(data_path)

    X, y, preprocessor, metadata = build_preprocessing_pipeline(
        df, text_col=TEXT_COLUMN, target_col=TARGET_COLUMN
    )

    target_kind = type_of_target(y)
    task_type = "classification" if target_kind in {"binary", "multiclass"} else "regression"
    logger.info("Detected task type for target '%s': %s", TARGET_COLUMN, task_type)

    best_model_name, scores, reports, conf_matrices, models = _train_models(X, y, task_type)
    best_model = models[best_model_name]

    # Feature importance (if applicable)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = {
            "indices": list(range(len(importances))),
            "values": importances.tolist(),
        }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(best_model, MODEL_PATH)
    save_preprocessor(PREPROCESSOR_PATH, preprocessor)

    metrics_payload = {
        "task_type": task_type,
        "best_model": best_model_name,
        "accuracies": scores,  # for regression this holds R2 scores
        "classification_reports": reports if task_type == "classification" else {},
        "confusion_matrices": {
            name: matrix.tolist() for name, matrix in conf_matrices.items()
        },
        "feature_importance": feature_importance,
        "target_column": TARGET_COLUMN,
        "text_columns": metadata.text_columns,
        "numeric_columns": metadata.numeric_columns,
        "categorical_columns": metadata.categorical_columns,
        "dropped_columns": metadata.dropped_columns,
    }
    save_json(METRICS_PATH, metrics_payload)

    logger.info("Training complete. Model and preprocessor saved.")


if __name__ == "__main__":
    train_and_save()

