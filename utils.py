import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import DATA_PATH, TARGET_COLUMN, TEXT_COLUMN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    text_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    dropped_columns: List[str]
    target_column: str
    id_like_columns: List[str]
    high_null_columns: List[str]
    constant_columns: List[str]


class HighCardinalityLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_: List[dict] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.classes_ = []
        for col in X.columns:
            uniques = pd.Series(X[col].astype(str).fillna("__MISSING__")).unique()
            mapping = {val: idx for idx, val in enumerate(uniques)}
            self.classes_.append(mapping)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        encoded_cols = []
        for i, col in enumerate(X.columns):
            mapping = self.classes_[i]
            encoded = X[col].astype(str).fillna("__MISSING__").map(mapping).fillna(-1)
            encoded_cols.append(encoded.to_numpy())
        return np.vstack(encoded_cols).T


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lowercase = False
        self.remove_punct = False
        self.remove_stopwords = False

    def fit(self, X, y=None):
        series = pd.Series(X).astype(str)
        if series.str.contains(r"[A-Z]", regex=True).any():
            self.lowercase = True
        if series.str.contains(r"[^\w\s]", regex=True).any():
            self.remove_punct = True
        if len(series) > 500:
            self.remove_stopwords = True
        logger.info(
            "TextPreprocessor config - lowercase=%s, remove_punct=%s, remove_stopwords=%s",
            self.lowercase,
            self.remove_punct,
            self.remove_stopwords,
        )
        return self

    def transform(self, X):
        import re
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        series = pd.Series(X).astype(str)
        if self.lowercase:
            series = series.str.lower()
        if self.remove_punct:
            series = series.apply(lambda s: re.sub(r"[^\w\s]", " ", s))
        if self.remove_stopwords:
            stopwords = ENGLISH_STOP_WORDS
            series = series.apply(
                lambda s: " ".join([w for w in s.split() if w not in stopwords])
            )
        return series.to_numpy()


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning("Dataset not found at %s", path)
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def detect_schema(df: pd.DataFrame, target_col: str, text_col_hint: Optional[str]) -> DatasetMetadata:
    logger.info("Detecting dataset schema using dtypes and heuristics")

    dropped_columns: List[str] = []
    id_like_columns: List[str] = []
    high_null_columns: List[str] = []
    constant_columns: List[str] = []

    n_rows = len(df)
    for col in df.columns:
        if col == target_col:
            continue

        nunique = df[col].nunique(dropna=True)
        null_ratio = df[col].isna().mean()

        if nunique == 1:
            constant_columns.append(col)
            continue
        if n_rows > 0 and nunique / max(n_rows, 1) > 0.95:
            id_like_columns.append(col)
            continue
        if null_ratio > 0.5:
            high_null_columns.append(col)
            continue

    df = df.drop(columns=id_like_columns + high_null_columns + constant_columns, errors="ignore")
    dropped_columns.extend(id_like_columns + high_null_columns + constant_columns)

    logger.info("Dropped columns: %s", dropped_columns)
    logger.info("ID-like columns: %s", id_like_columns)
    logger.info("High-null columns: %s", high_null_columns)
    logger.info("Constant columns: %s", constant_columns)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_candidates = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    text_columns: List[str] = []
    cat_columns: List[str] = []

    text_col_from_hint: Optional[str] = None
    if text_col_hint:
        # Support comma-separated config by taking the first valid column
        for candidate in [c.strip() for c in text_col_hint.split(",")]:
            if candidate in df.columns:
                text_col_from_hint = candidate
                break

    for col in categorical_candidates:
        if col == target_col:
            continue
        series = df[col].astype(str)
        avg_len = series.str.len().mean()
        if text_col_from_hint and col == text_col_from_hint:
            text_columns.append(col)
        elif avg_len > 30:
            text_columns.append(col)
        else:
            cat_columns.append(col)

    logger.info("Numeric columns: %s", numeric_columns)
    logger.info("Categorical columns: %s", cat_columns)
    logger.info("Text columns: %s", text_columns)

    return DatasetMetadata(
        text_columns=text_columns,
        numeric_columns=[c for c in numeric_columns if c != target_col],
        categorical_columns=cat_columns,
        dropped_columns=dropped_columns,
        target_column=target_col,
        id_like_columns=id_like_columns,
        high_null_columns=high_null_columns,
        constant_columns=constant_columns,
    )


def build_preprocessing_pipeline(
    df: pd.DataFrame, text_col: Optional[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray, Pipeline, DatasetMetadata]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    metadata = detect_schema(df, target_col=target_col, text_col_hint=text_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if not numeric
    if not pd.api.types.is_numeric_dtype(y):
        logger.info("Encoding non-numeric target column '%s' as categorical codes", target_col)
        y = y.astype("category").cat.codes

    text_columns = metadata.text_columns
    numeric_columns = metadata.numeric_columns
    categorical_columns = metadata.categorical_columns

    only_text_features = (
        len(text_columns) == 1 and not numeric_columns and not categorical_columns
    )

    if only_text_features:
        logger.info("Using NLP-only pipeline based on dataset schema")
        text_col_name = text_columns[0]
        text_series = X[text_col_name].fillna("")
        text_preprocessor = TextPreprocessor()
        text_preprocessor.fit(text_series)

        tfidf = TfidfVectorizer()
        tfidf.fit(text_preprocessor.transform(text_series))

        def transform_fn(df_input: pd.DataFrame):
            series = df_input[text_col_name].fillna("")
            return tfidf.transform(text_preprocessor.transform(series))

        class TextOnlyPipeline(Pipeline):
            def __init__(self, preproc, vectorizer, column):
                super().__init__([("text_preprocessor", preproc), ("tfidf", vectorizer)])
                self._column = column

            def transform(self, X_input):
                series = pd.Series(X_input[self._column]).fillna("")
                Xt = self.named_steps["text_preprocessor"].transform(series)
                return self.named_steps["tfidf"].transform(Xt)

        pipeline = TextOnlyPipeline(text_preprocessor, tfidf, text_col_name)
        X_transformed = pipeline.transform(X)
        pipeline.metadata_ = metadata  # type: ignore[attr-defined]
        return X_transformed, np.asarray(y), pipeline, metadata

    transformers = []

    if numeric_columns:
        numeric_pipeline_base = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipeline_base, numeric_columns))

    low_card_cats: List[str] = []
    high_card_cats: List[str] = []
    for col in categorical_columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 20:
            low_card_cats.append(col)
        else:
            high_card_cats.append(col)

    if low_card_cats:
        cat_low_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat_low", cat_low_pipeline, low_card_cats))

    if high_card_cats:
        cat_high_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("label", HighCardinalityLabelEncoder()),
            ]
        )
        transformers.append(("cat_high", cat_high_pipeline, high_card_cats))

    for txt_col in text_columns:
        text_pipeline = Pipeline(
            steps=[
                (
                    "selector",
                    FunctionTransformer(
                        lambda X_df, c=txt_col: pd.Series(X_df[c]).fillna(""),
                        validate=False,
                    ),
                ),
                ("text_preprocessor", TextPreprocessor()),
                ("tfidf", TfidfVectorizer()),
            ]
        )
        transformers.append((f"text_{txt_col}", text_pipeline, [txt_col]))

    preprocessor = ColumnTransformer(transformers=transformers)

    logger.info("Fitting preprocessing pipeline")
    X_transformed = preprocessor.fit_transform(X)

    # Attach metadata and info for downstream models
    preprocessor.metadata_ = metadata  # type: ignore[attr-defined]
    preprocessor.numeric_with_scaler_columns_ = numeric_columns  # type: ignore[attr-defined]

    return X_transformed, np.asarray(y), preprocessor, metadata


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_preprocessor(path: str, preprocessor: ColumnTransformer) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(preprocessor, path)
