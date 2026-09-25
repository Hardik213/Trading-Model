from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_signal_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model


def predict_probabilities(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=X.index, name="probability")
