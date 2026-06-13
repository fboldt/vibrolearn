import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class CNNLSTMNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=100,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Sequential(
            nn.Linear(200, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        x = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)

        return x


class CNNLSTMClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        device="cuda",
        verbose=True,
        early_stopping_patience=10,
        min_delta=0.0,
        checkpoint_path="best_cnn_lstm.pt",
        use_rms_normalization=True
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.use_rms_normalization = use_rms_normalization

    def fit(self, X, y, X_val=None, y_val=None):
        X = self._prepare_input(X)
        y = np.asarray(y)

        has_validation = X_val is not None and y_val is not None

        if has_validation:
            X_val = self._prepare_input(X_val)
            y_val = np.asarray(y_val)

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device_ = "cpu"
            if self.verbose:
                print("CUDA is not available. Using CPU instead.")
        else:
            self.device_ = self.device

        self.classes_ = np.unique(y)

        self.class_to_index_ = {
            label: idx for idx, label in enumerate(self.classes_)
        }

        self.index_to_class_ = {
            idx: label for label, idx in self.class_to_index_.items()
        }

        y_encoded = np.array(
            [self.class_to_index_[label] for label in y],
            dtype=np.int64
        )

        if has_validation:
            unknown_labels = set(y_val) - set(self.classes_)

            if unknown_labels:
                raise ValueError(
                    f"Validation set contains labels not present in training set: "
                    f"{unknown_labels}"
                )

            y_val_encoded = np.array(
                [self.class_to_index_[label] for label in y_val],
                dtype=np.int64
            )

        self.model_ = CNNLSTMNet(
            num_classes=len(self.classes_)
        ).to(self.device_)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y_encoded, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = None

        if has_validation:
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val_encoded, dtype=torch.long)
                ),
                batch_size=self.batch_size,
                shuffle=False
            )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate
        )

        best_score = -float("inf")
        best_epoch = 0
        best_model_state = None
        best_metric_name = "val_accuracy" if has_validation else "train_accuracy"
        epochs_without_improvement = 0

        self.history_ = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(self.epochs):
            train_loss, train_accuracy = self._train_one_epoch(
                train_loader,
                criterion,
                optimizer
            )

            self.history_["train_loss"].append(train_loss)
            self.history_["train_accuracy"].append(train_accuracy)

            if has_validation:
                val_loss, val_accuracy = self._evaluate_loader(
                    val_loader,
                    criterion
                )

                self.history_["val_loss"].append(val_loss)
                self.history_["val_accuracy"].append(val_accuracy)

                checkpoint_score = val_accuracy
            else:
                val_loss = None
                val_accuracy = None

                checkpoint_score = train_accuracy

            improved = checkpoint_score > (best_score + self.min_delta)

            if improved:
                best_score = checkpoint_score
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                best_model_state = copy.deepcopy(
                    self.model_.state_dict()
                )

                if self.checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "model_state_dict": best_model_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_score": best_score,
                            "checkpoint_metric": best_metric_name,
                            "classes": self.classes_,
                            "class_to_index": self.class_to_index_,
                            "index_to_class": self.index_to_class_,
                            "used_validation": has_validation,
                            "use_rms_normalization": self.use_rms_normalization
                        },
                        self.checkpoint_path
                    )
            else:
                epochs_without_improvement += 1

            if self.verbose:
                if has_validation:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}] - "
                        f"Loss: {train_loss:.4f} - "
                        f"Train Acc: {train_accuracy:.4f} - "
                        f"Val Loss: {val_loss:.4f} - "
                        f"Val Acc: {val_accuracy:.4f} - "
                        f"Best Val Acc: {best_score:.4f} - "
                        f"Best Epoch: {best_epoch}"
                    )
                else:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}] - "
                        f"Loss: {train_loss:.4f} - "
                        f"Train Acc: {train_accuracy:.4f} - "
                        f"Best Train Acc: {best_score:.4f} - "
                        f"Best Epoch: {best_epoch}"
                    )

            if (
                self.early_stopping_patience is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch} - "
                        f"Best {best_metric_name}: {best_score:.4f}"
                    )
                break

        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)

        self.best_score_ = best_score
        self.best_epoch_ = best_epoch
        self.best_metric_name_ = best_metric_name
        self.used_validation_ = has_validation

        return self

    def predict(self, X):
        X = self._prepare_input(X)

        self.model_.eval()

        predictions = []

        loader = DataLoader(
            torch.tensor(X, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device_)

                outputs = self.model_(xb)
                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())

        return np.array(
            [self.index_to_class_[idx] for idx in predictions]
        )

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.asarray(y))

    def _prepare_input(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 2:
            X = X[..., np.newaxis]

        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (n_samples, seq_len) or "
                f"(n_samples, seq_len, channels), got shape {X.shape}"
            )

        if self.use_rms_normalization:
            X = self._rms_normalize(X)

        return X

    def _rms_normalize(self, X):
        rms = np.sqrt(
            np.mean(X ** 2, axis=1, keepdims=True)
        )

        return X / (rms + 1e-8)

    def _train_one_epoch(self, train_loader, criterion, optimizer):
        self.model_.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(self.device_)
            yb = yb.to(self.device_)

            optimizer.zero_grad()

            outputs = self.model_(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)

            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate_loader(self, loader, criterion):
        self.model_.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                outputs = self.model_(xb)
                loss = criterion(outputs, yb)

                batch_size = xb.size(0)

                total_loss += loss.item() * batch_size

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy