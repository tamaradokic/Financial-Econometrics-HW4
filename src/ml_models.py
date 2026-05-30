"""
ml_models.py
------------
Machine learning models for one-step-ahead lnRV forecasting.

All models share the same fit(X_train, y_train) / predict(X_test) interface
so they can be plugged seamlessly into the rolling forecast engine.

Models:
  - LASSOModel   : L1-regularised linear regression (sklearn)
  - RidgeModel   : L2-regularised linear regression (sklearn)
  - XGBoostModel : Extreme Gradient Boosting (xgboost)
  - LightGBMModel: Light Gradient Boosting (lightgbm)

Feature set (same as HAR, for a fair comparison):
  [y_{t-1},  ȳ^(5)_{t-1},  ȳ^(22)_{t-1}]
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


# ── feature builder (shared) ─────────────────────────────────────────────────

def build_features(lnrv_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) feature matrix from a 1-D lnRV array.
    X columns: [y_{t-1}, mean_{t-5:t-1}, mean_{t-22:t-1}]
    Valid rows start at index 22.
    """
    T = len(lnrv_array)
    y_d = np.roll(lnrv_array, 1)
    y_w = np.array([lnrv_array[max(0, i-5):i].mean() if i >= 5 else np.nan
                    for i in range(T)])
    y_m = np.array([lnrv_array[max(0, i-22):i].mean() if i >= 22 else np.nan
                    for i in range(T)])

    X = np.column_stack([y_d, y_w, y_m])
    y = lnrv_array.copy()

    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


# ── LASSO ────────────────────────────────────────────────────────────────────

class LASSOModel:
    """LASSO with cross-validated regularisation strength."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = LassoCV(cv=5, max_iter=5000, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LASSOModel":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])


# ── Ridge ────────────────────────────────────────────────────────────────────

class RidgeModel:
    """Ridge regression with cross-validated regularisation strength."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])


# ── XGBoost ──────────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    Extreme Gradient Boosting.
    Hyperparameters fixed at sensible defaults; can be tuned via GridSearchCV
    if desired (omitted here to keep rolling OOS tractable across 30 stocks).
    """
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])


# ── LightGBM ─────────────────────────────────────────────────────────────────

class LightGBMModel:
    """
    LightGBM – faster alternative to XGBoost with similar accuracy.
    """
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])

