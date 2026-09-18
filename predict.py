import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator

from config import MODEL_PATH, PREPROCESSOR_PATH, TARGET_COLUMN, TEXT_COLUMN, ensure_directories


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_artifacts() -> Tuple[BaseEstimator, Any]:
    ensure_directories()

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        raise FileNotFoundError("Model or preprocessor not found. Please train first.")

    model = load(MODEL_PATH)
    preprocessor = load(PREPROCESSOR_PATH)
    logger.info("Loaded model and preprocessor from disk")
    return model, preprocessor


def predict_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    model, preprocessor = _load_artifacts()
    X = df.copy()
    X_transformed = preprocessor.transform(X)
    preds = model.predict(X_transformed)
    return preds


def predict_ticket(text: str) -> Dict[str, Any]:
    model, preprocessor = _load_artifacts()

    text_col_name = None
    if TEXT_COLUMN:
        for candidate in [c.strip() for c in TEXT_COLUMN.split(",")]:
            text_col_name = candidate
            break

    if not text_col_name:
        raise ValueError("TEXT_COLUMN is not configured properly.")

    df_input = pd.DataFrame([{text_col_name: text}])
    X_transformed = preprocessor.transform(df_input)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_transformed)
        preds = np.argmax(proba, axis=1)
        confidence = float(np.max(proba, axis=1)[0])
    else:
        preds = model.predict(X_transformed)
        confidence = 0.0

    prediction = int(preds[0])

    return {
        "prediction": prediction,
        "confidence": confidence,
    }

