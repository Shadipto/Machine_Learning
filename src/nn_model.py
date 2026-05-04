from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score


@dataclass
class TrainingHistory:
    """Container for per-epoch training diagnostics."""

    epoch: int
    train_loss: float
    val_loss: float
    train_f1: float
    val_f1: float


class SparseBinaryMLP:
    """Feed-forward MLP for binary text classification with sparse TF-IDF input.

    Architecture:
    - Input: sparse TF-IDF vector
    - Hidden 1: Dense(hidden_dim_1) + ReLU + Dropout(dropout_1)
    - Hidden 2: Dense(hidden_dim_2) + ReLU + Dropout(dropout_2)
    - Output: Dense(1) + Sigmoid

    Notes:
    - Uses manual forward/backward propagation with Adam optimizer.
    - Keeps sparse input until the first dense projection for memory efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 64,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
        learning_rate: float = 1e-3,
        l2_lambda: float = 1e-5,
        batch_size: int = 1024,
        max_epochs: int = 20,
        patience: int = 4,
        random_state: int = 42,
        positive_label: str = "suicide",
        negative_label: str = "non-suicide",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.positive_label = positive_label
        self.negative_label = negative_label

        self._rng = np.random.default_rng(random_state)
        self._eps = 1e-7
        self._t = 0

        self._params = self._initialize_parameters()
        self._adam_m = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self._params.items()}

    def _initialize_parameters(self) -> dict[str, np.ndarray]:
        scale_1 = np.sqrt(2.0 / self.input_dim)
        scale_2 = np.sqrt(2.0 / self.hidden_dim_1)
        scale_3 = np.sqrt(2.0 / self.hidden_dim_2)

        return {
            "W1": (self._rng.standard_normal((self.input_dim, self.hidden_dim_1)).astype(np.float32) * scale_1),
            "b1": np.zeros((1, self.hidden_dim_1), dtype=np.float32),
            "W2": (self._rng.standard_normal((self.hidden_dim_1, self.hidden_dim_2)).astype(np.float32) * scale_2),
            "b2": np.zeros((1, self.hidden_dim_2), dtype=np.float32),
            "W3": (self._rng.standard_normal((self.hidden_dim_2, 1)).astype(np.float32) * scale_3),
            "b3": np.zeros((1, 1), dtype=np.float32),
        }

    @staticmethod
    def _ensure_csr(X: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
        if sparse.issparse(X):
            return X.tocsr().astype(np.float32)
        return sparse.csr_matrix(X, dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _apply_dropout(self, activations: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
        if rate <= 0.0:
            mask = np.ones_like(activations, dtype=np.float32)
            return activations, mask
        keep_prob = 1.0 - rate
        mask = (self._rng.random(activations.shape) < keep_prob).astype(np.float32) / keep_prob
        return activations * mask, mask

    def _forward(self, X_batch: sparse.csr_matrix, training: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        params = self._params

        z1 = np.asarray(X_batch.dot(params["W1"]), dtype=np.float32) + params["b1"]
        a1 = self._relu(z1)
        if training:
            a1, drop1_mask = self._apply_dropout(a1, self.dropout_1)
        else:
            drop1_mask = np.ones_like(a1, dtype=np.float32)

        z2 = a1.dot(params["W2"]) + params["b2"]
        a2 = self._relu(z2)
        if training:
            a2, drop2_mask = self._apply_dropout(a2, self.dropout_2)
        else:
            drop2_mask = np.ones_like(a2, dtype=np.float32)

        z3 = a2.dot(params["W3"]) + params["b3"]
        y_hat = self._sigmoid(z3)

        cache = {
            "z1": z1,
            "a1": a1,
            "drop1_mask": drop1_mask,
            "z2": z2,
            "a2": a2,
            "drop2_mask": drop2_mask,
            "z3": z3,
            "y_hat": y_hat,
        }
        return y_hat, cache

    def _compute_loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        y_hat = np.clip(y_hat, self._eps, 1.0 - self._eps)
        data_loss = -np.mean(y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat))

        l2_penalty = (
            np.sum(self._params["W1"] ** 2)
            + np.sum(self._params["W2"] ** 2)
            + np.sum(self._params["W3"] ** 2)
        )
        reg_loss = 0.5 * self.l2_lambda * l2_penalty
        return float(data_loss + reg_loss)

    def _backward(
        self,
        X_batch: sparse.csr_matrix,
        y_true: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        params = self._params
        m = float(y_true.shape[0])

        y_hat = cache["y_hat"]
        d_z3 = (y_hat - y_true) / m
        d_W3 = cache["a2"].T.dot(d_z3) + self.l2_lambda * params["W3"]
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_a2 = d_z3.dot(params["W3"].T)
        d_a2 *= cache["drop2_mask"]
        d_z2 = d_a2 * (cache["z2"] > 0.0)
        d_W2 = cache["a1"].T.dot(d_z2) + self.l2_lambda * params["W2"]
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = d_z2.dot(params["W2"].T)
        d_a1 *= cache["drop1_mask"]
        d_z1 = d_a1 * (cache["z1"] > 0.0)
        d_W1 = np.asarray(X_batch.T.dot(d_z1), dtype=np.float32) + self.l2_lambda * params["W1"]
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        return {
            "W1": d_W1.astype(np.float32),
            "b1": d_b1.astype(np.float32),
            "W2": d_W2.astype(np.float32),
            "b2": d_b2.astype(np.float32),
            "W3": d_W3.astype(np.float32),
            "b3": d_b3.astype(np.float32),
        }

    def _adam_step(self, gradients: dict[str, np.ndarray]) -> None:
        beta1 = 0.9
        beta2 = 0.999
        self._t += 1

        for name, grad in gradients.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad ** 2)

            m_hat = self._adam_m[name] / (1.0 - (beta1 ** self._t))
            v_hat = self._adam_v[name] / (1.0 - (beta2 ** self._t))

            self._params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self._eps)

    def _prepare_binary_targets(self, y: np.ndarray | list[str]) -> np.ndarray:
        y_array = np.asarray(y)
        return (y_array == self.positive_label).astype(np.float32).reshape(-1, 1)

    def _epoch_metrics(self, X: sparse.csr_matrix, y_true: np.ndarray) -> tuple[float, float]:
        y_binary = self._prepare_binary_targets(y_true)
        y_hat, _ = self._forward(X, training=False)
        loss = self._compute_loss(y_binary, y_hat)
        y_pred = (y_hat.ravel() >= 0.5).astype(int)
        f1 = f1_score(y_binary.ravel().astype(int), y_pred, zero_division=0)
        return loss, float(f1)

    def fit(
        self,
        X_train: sparse.spmatrix | np.ndarray,
        y_train: np.ndarray | list[str],
        X_val: sparse.spmatrix | np.ndarray,
        y_val: np.ndarray | list[str],
        verbose: bool = True,
    ) -> list[TrainingHistory]:
        """Train the model with mini-batch Adam and early stopping on validation F1."""
        X_train_csr = self._ensure_csr(X_train)
        X_val_csr = self._ensure_csr(X_val)

        y_train_array = np.asarray(y_train)
        y_val_array = np.asarray(y_val)
        y_train_binary = self._prepare_binary_targets(y_train_array)

        n_samples = X_train_csr.shape[0]
        best_val_f1 = -np.inf
        best_params = {name: value.copy() for name, value in self._params.items()}
        wait = 0
        history: list[TrainingHistory] = []

        for epoch in range(1, self.max_epochs + 1):
            shuffled_indices = self._rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = shuffled_indices[start:end]

                X_batch = X_train_csr[batch_idx]
                y_batch = y_train_binary[batch_idx]

                _, cache = self._forward(X_batch, training=True)
                gradients = self._backward(X_batch, y_batch, cache)
                self._adam_step(gradients)

            train_loss, train_f1 = self._epoch_metrics(X_train_csr, y_train_array)
            val_loss, val_f1 = self._epoch_metrics(X_val_csr, y_val_array)

            history.append(
                TrainingHistory(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_f1=train_f1,
                    val_f1=val_f1,
                )
            )

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{self.max_epochs} "
                    f"| train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"| train_f1={train_f1:.4f} val_f1={val_f1:.4f}"
                )

            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1
                best_params = {name: value.copy() for name, value in self._params.items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break

        self._params = best_params
        return history

    def predict_positive_proba(self, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
        X_csr = self._ensure_csr(X)
        y_hat, _ = self._forward(X_csr, training=False)
        return y_hat.ravel().astype(np.float32)

    def predict_proba(self, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
        positive = self.predict_positive_proba(X)
        negative = 1.0 - positive
        return np.column_stack([negative, positive]).astype(np.float32)

    def predict(self, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
        positive = self.predict_positive_proba(X)
        return np.where(positive >= 0.5, self.positive_label, self.negative_label)

    def get_params(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
            "dropout_1": self.dropout_1,
            "dropout_2": self.dropout_2,
            "learning_rate": self.learning_rate,
            "l2_lambda": self.l2_lambda,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "random_state": self.random_state,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
        }


def save_sparse_mlp(model: SparseBinaryMLP, filepath: str | Path) -> None:
    """Persist a trained sparse MLP model using pickle."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file_handle:
        pickle.dump(model, file_handle)


def load_sparse_mlp(filepath: str | Path) -> SparseBinaryMLP:
    """Load a pickled sparse MLP model from disk."""
    with open(filepath, "rb") as file_handle:
        model = pickle.load(file_handle)
    return model
