"""
LSTM classifier for directional prediction.
Uses PyTorch. Expects scaled sequences of shape (batch, seq_len, n_features).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.utils.config import get
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)   # logit


# ── Training wrapper ──────────────────────────────────────────────────────────

class LSTMTrainer:
    def __init__(self):
        cfg = get("models.lstm", {})
        self.seq_len     = cfg.get("seq_len", 20)
        self.hidden_size = cfg.get("hidden_size", 64)
        self.num_layers  = cfg.get("num_layers", 2)
        self.dropout     = cfg.get("dropout", 0.3)
        self.batch_size  = cfg.get("batch_size", 64)
        self.lr          = cfg.get("lr", 0.001)
        self.epochs      = cfg.get("epochs", 100)
        self.patience    = cfg.get("patience", 15)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler      = StandardScaler()
        self.model: LSTMClassifier | None = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len: i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def _loader(self, X_seq, y_seq, shuffle: bool):
        ds = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "LSTMTrainer":
        X_tr = self.scaler.fit_transform(X_train)
        X_va = self.scaler.transform(X_val)

        Xtr_seq, ytr_seq = self._make_sequences(X_tr, y_train)
        Xva_seq, yva_seq = self._make_sequences(X_va, y_val)

        n_features = X_tr.shape[1]
        self.model = LSTMClassifier(n_features, self.hidden_size,
                                    self.num_layers, self.dropout).to(self.device)

        # Class weights
        pos_ratio = y_train.mean()
        pos_weight = torch.tensor([(1 - pos_ratio) / (pos_ratio + 1e-8)],
                                   dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        tr_loader = self._loader(Xtr_seq, ytr_seq, shuffle=True)
        va_loader = self._loader(Xva_seq, yva_seq, shuffle=False)

        best_val_loss = float("inf")
        patience_ctr  = 0
        best_state    = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()

            self.model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    va_loss += criterion(self.model(xb), yb).item()

            scheduler.step(va_loss)

            if epoch % 10 == 0:
                log.info(f"Epoch {epoch:3d} | tr_loss={tr_loss/len(tr_loader):.4f} "
                         f"| va_loss={va_loss/len(va_loader):.4f}")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probabilities for class 1 (up). Handles seq_len offset."""
        assert self.model is not None, "Call fit() first"
        X_sc = self.scaler.transform(X)
        # Pad the first seq_len rows with zeros so output length == input length
        pad = np.zeros((self.seq_len, X_sc.shape[1]), dtype=np.float32)
        X_padded = np.vstack([pad, X_sc])
        Xseq, _ = self._make_sequences(X_padded, np.zeros(len(X_padded)))
        loader = self._loader(Xseq, np.zeros(len(Xseq)), shuffle=False)

        probs = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb.to(self.device))
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.array(probs, dtype=np.float32)
