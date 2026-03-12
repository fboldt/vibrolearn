
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset


class _CNN1DNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters, kernel_size, hidden_dim, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_filters=32,
        kernel_size=7,
        hidden_dim=64,
        dropout=0.2,
        epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=None,
        random_state=42,
        verbose=False,
    ):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _resolve_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_inputs(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        if X.ndim != 3:
            raise ValueError("X must be 2D or 3D with shape (n_samples, length, channels).")
        # Convert from (N, L, C) to (N, C, L) for Conv1d.
        X = np.transpose(X, (0, 2, 1))
        return X

    def fit(self, X, y):
        self._set_seed()
        X = self._prepare_inputs(X)
        y = np.asarray(y)

        classes, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError("CNN1D requires at least 2 classes in y.")

        self.device_ = self._resolve_device()
        self.model_ = _CNN1DNet(
            in_channels=X.shape[1],
            n_classes=self.n_classes_,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device_)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(loader.dataset)
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_loss:.5f}")

        return self

    def predict(self, X):
        check_is_fitted(self, ["model_", "classes_", "device_"])
        X = self._prepare_inputs(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)

        predictions = []
        self.model_.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device_)
                logits = self.model_(batch_x)
                batch_pred = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.append(batch_pred)

        y_pred_encoded = np.concatenate(predictions)
        return self.classes_[y_pred_encoded]