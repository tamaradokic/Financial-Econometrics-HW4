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
  - LSTMModel    : Long Short-Term Memory network (PyTorch)

Feature set (same as HAR, for a fair comparison):
  [y_{t-1},  ȳ^(5)_{t-1},  ȳ^(22)_{t-1}]
"""

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


# ── LSTM ─────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """Single-layer LSTM followed by a linear output layer."""
    def __init__(self, input_size: int = 3, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    """
    LSTM for one-step-ahead lnRV forecasting.

    Uses a sliding window of `seq_len` days as input to allow the network
    to capture temporal dependencies beyond the 22-day HAR horizon.

    We keep seq_len = 22 to maintain comparability with HAR features.
    Training uses the Adam optimizer with early stopping via a validation split.
    """

    def __init__(self, seq_len: int = 22, hidden_size: int = 32,
                 epochs: int = 50, batch_size: int = 64,
                 lr: float = 1e-3, device: Optional[str] = None):
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_x   = StandardScaler()
        self.scaler_y   = StandardScaler()
        self.net        = None

    def _make_sequences(self, X: np.ndarray, y: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Convert (X, y) into sliding-window sequences."""
        seqs, targets = [], []
        for i in range(self.seq_len, len(y)):
            seqs.append(X[i - self.seq_len: i])
            targets.append(y[i])
        return np.array(seqs), np.array(targets)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        # Standardise
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        seqs, tgts = self._make_sequences(Xs, ys)

        # 90/10 train/val split (time-ordered)
        n_val = max(1, int(0.1 * len(tgts)))
        X_tr, y_tr = seqs[:-n_val], tgts[:-n_val]
        X_val, y_val = seqs[-n_val:], tgts[-n_val:]

        ds_tr  = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        ds_val = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        dl_tr  = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=False)

        self.net = _LSTMNet(
            input_size=X.shape[1],
            hidden_size=self.hidden_size,
        ).to(self.device)

        opt  = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val, patience, counter = np.inf, 7, 0
        best_state = None
        self.net.train()
        for epoch in range(self.epochs):
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                opt.step()

            # Validation loss
            self.net.eval()
            with torch.no_grad():
                xv = torch.FloatTensor(X_val).to(self.device)
                yv = torch.FloatTensor(y_val).to(self.device)
                val_loss = loss_fn(self.net(xv), yv).item()
            self.net.train()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self.net.eval()
        # Keep last seq_len of Xs for prediction
        self._last_X = Xs[-self.seq_len:]
        return self

    def predict_one(self, x: np.ndarray) -> float:
        """
        One-step forecast.
        x : feature vector for the NEW observation (shape 3,)
        Appends x to the stored sequence window and predicts.
        """
        xs = self.scaler_x.transform(x.reshape(1, -1))
        window = np.vstack([self._last_X[1:], xs])  # slide window
        self._last_X = window

        t = torch.FloatTensor(window[np.newaxis]).to(self.device)
        with torch.no_grad():
            pred_s = self.net(t).item()
        return float(self.scaler_y.inverse_transform([[pred_s]])[0, 0])
