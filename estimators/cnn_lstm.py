import copy

import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

from estimators.pipeline import Pipeline


# ============================================================
# REDE
# ============================================================

class _CNNLSTMNet(nn.Module):
    """
    Arquitetura CNN-LSTM utilizada para classificação
    de sinais de vibração.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
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

        self.output = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):
        # (batch, sequence, channels)
        #         ↓
        # (batch, channels, sequence)

        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        # Entrada esperada pela LSTM:
        # (batch, sequence, features)

        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        # Concatena os últimos estados das duas
        # direções da BiLSTM.
        x = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        x = self.fc1(x)
        x = self.fc2(x)

        return self.output(x)


# ============================================================
# CLASSIFICADOR
# ============================================================

class CNNLSTMClassifier(
    ClassifierMixin,
    BaseEstimator
):
    """
    Classificador CNN-LSTM compatível com a interface
    de estimadores do Scikit-learn.
    """

    def __init__(
        self,
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        device="cuda",
        verbose=True,
        early_stopping_patience=10,
        min_delta=0.0,
        checkpoint_path=None,
        use_rms_normalization=True
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = device
        self.verbose = verbose

        self.early_stopping_patience = (
            early_stopping_patience
        )
        self.min_delta = min_delta

        self.checkpoint_path = checkpoint_path

        self.use_rms_normalization = (
            use_rms_normalization
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None
    ):
        X = self._prepare_input(X)
        y = np.asarray(y)

        has_validation = (
            X_val is not None
            and y_val is not None
        )

        self.device_ = self._resolve_device()

        # ----------------------------------------------------
        # Classes
        # ----------------------------------------------------

        self.classes_ = np.unique(y)

        self.class_to_index_ = {
            label: index
            for index, label
            in enumerate(self.classes_)
        }

        y_encoded = self._encode_labels(y)

        if has_validation:
            X_val = self._prepare_input(
                X_val
            )

            y_val_encoded = (
                self._encode_labels(
                    np.asarray(y_val),
                    validate=True
                )
            )

        # ----------------------------------------------------
        # Modelo
        # ----------------------------------------------------

        self.model_ = _CNNLSTMNet(
            num_classes=len(self.classes_)
        ).to(self.device_)

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        train_loader = self._create_loader(
            X,
            y_encoded,
            shuffle=True
        )

        val_loader = None

        if has_validation:
            val_loader = self._create_loader(
                X_val,
                y_val_encoded,
                shuffle=False
            )

        # ----------------------------------------------------
        # Otimização
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate
        )

        # ----------------------------------------------------
        # Histórico
        # ----------------------------------------------------

        self.history_ = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        best_score = -float("inf")
        best_epoch = 0
        best_model_state = None

        epochs_without_improvement = 0

        best_metric_name = (
            "val_accuracy"
            if has_validation
            else "train_accuracy"
        )

        # ====================================================
        # TREINAMENTO
        # ====================================================

        for epoch in range(self.epochs):

            train_loss, train_accuracy = (
                self._train_epoch(
                    train_loader,
                    criterion,
                    optimizer
                )
            )

            self.history_["train_loss"].append(
                train_loss
            )

            self.history_[
                "train_accuracy"
            ].append(
                train_accuracy
            )

            val_loss = None
            val_accuracy = None

            # ------------------------------------------------
            # Validação
            # ------------------------------------------------

            if has_validation:
                val_loss, val_accuracy = (
                    self._evaluate(
                        val_loader,
                        criterion
                    )
                )

                self.history_[
                    "val_loss"
                ].append(
                    val_loss
                )

                self.history_[
                    "val_accuracy"
                ].append(
                    val_accuracy
                )

                current_score = val_accuracy

            else:
                current_score = (
                    train_accuracy
                )

            # ------------------------------------------------
            # Melhor modelo
            # ------------------------------------------------

            improved = (
                current_score
                > best_score + self.min_delta
            )

            if improved:
                best_score = current_score
                best_epoch = epoch + 1

                epochs_without_improvement = 0

                best_model_state = (
                    copy.deepcopy(
                        self.model_.state_dict()
                    )
                )

                self._save_checkpoint(
                    epoch=best_epoch,
                    score=best_score,
                    metric_name=best_metric_name,
                    model_state=best_model_state,
                    optimizer=optimizer,
                    has_validation=has_validation
                )

            else:
                epochs_without_improvement += 1

            # ------------------------------------------------
            # Log
            # ------------------------------------------------

            if self.verbose:
                self._print_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    metric_name=best_metric_name
                )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                self.early_stopping_patience
                is not None
                and epochs_without_improvement
                >= self.early_stopping_patience
            ):
                if self.verbose:
                    print(
                        "Early stopping at epoch "
                        f"{epoch + 1}. "
                        f"Best epoch: {best_epoch} - "
                        f"Best {best_metric_name}: "
                        f"{best_score:.4f}"
                    )

                break

        # ----------------------------------------------------
        # Restaura melhor modelo
        # ----------------------------------------------------

        if best_model_state is not None:
            self.model_.load_state_dict(
                best_model_state
            )

        self.best_score_ = best_score
        self.best_epoch_ = best_epoch
        self.best_metric_name_ = (
            best_metric_name
        )
        self.used_validation_ = (
            has_validation
        )

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):
        X = self._prepare_input(X)

        loader = self._create_loader(
            X,
            shuffle=False
        )

        self.model_.eval()

        predictions = []

        with torch.inference_mode():

            for xb in loader:

                xb = xb.to(
                    self.device_,
                    non_blocking=True
                )

                outputs = self.model_(xb)

                indices = outputs.argmax(
                    dim=1
                )

                predictions.extend(
                    indices.cpu().numpy()
                )

        # O índice produzido pela rede corresponde
        # diretamente à posição em self.classes_.
        return self.classes_[
            np.asarray(predictions)
        ]

    # ========================================================
    # PREPARAÇÃO DOS DADOS
    # ========================================================

    def _prepare_input(self, X):
        X = np.asarray(
            X,
            dtype=np.float32
        )

        if X.ndim == 2:
            X = X[..., np.newaxis]

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                "(n_samples, seq_len) or "
                "(n_samples, seq_len, channels). "
                f"Received: {X.shape}"
            )

        if self.use_rms_normalization:
            X = self._rms_normalize(X)

        return X

    @staticmethod
    def _rms_normalize(X):
        rms = np.sqrt(
            np.mean(
                X ** 2,
                axis=1,
                keepdims=True
            )
        )

        return X / (rms + 1e-8)

    # ========================================================
    # DATA LOADER
    # ========================================================

    def _create_loader(
        self,
        X,
        y=None,
        shuffle=False
    ):
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32
        )

        if y is None:
            dataset = X_tensor

        else:
            y_tensor = torch.as_tensor(
                y,
                dtype=torch.long
            )

            dataset = TensorDataset(
                X_tensor,
                y_tensor
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=(
                self.device_ == "cuda"
            )
        )

    # ========================================================
    # TREINAMENTO DE UMA ÉPOCA
    # ========================================================

    def _train_epoch(
        self,
        loader,
        criterion,
        optimizer
    ):
        self.model_.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:

            xb = xb.to(
                self.device_,
                non_blocking=True
            )

            yb = yb.to(
                self.device_,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = self.model_(xb)

            loss = criterion(
                outputs,
                yb
            )

            loss.backward()
            optimizer.step()

            batch_size = yb.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            correct += (
                outputs.argmax(dim=1)
                == yb
            ).sum().item()

            total += batch_size

        return (
            total_loss / total,
            correct / total
        )

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    def _evaluate(
        self,
        loader,
        criterion
    ):
        self.model_.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():

            for xb, yb in loader:

                xb = xb.to(
                    self.device_,
                    non_blocking=True
                )

                yb = yb.to(
                    self.device_,
                    non_blocking=True
                )

                outputs = self.model_(xb)

                loss = criterion(
                    outputs,
                    yb
                )

                batch_size = yb.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                correct += (
                    outputs.argmax(dim=1)
                    == yb
                ).sum().item()

                total += batch_size

        return (
            total_loss / total,
            correct / total
        )

    # ========================================================
    # LABELS
    # ========================================================

    def _encode_labels(
        self,
        y,
        validate=False
    ):
        if validate:

            unknown_labels = (
                set(np.unique(y))
                - set(self.classes_)
            )

            if unknown_labels:
                raise ValueError(
                    "Validation set contains labels "
                    "not present in training set: "
                    f"{unknown_labels}"
                )

        return np.asarray(
            [
                self.class_to_index_[label]
                for label in y
            ],
            dtype=np.int64
        )

    # ========================================================
    # DEVICE
    # ========================================================

    def _resolve_device(self):
        requested_device = torch.device(
            self.device
        )

        if (
            requested_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            if self.verbose:
                print(
                    "CUDA is not available. "
                    "Using CPU instead."
                )

            return "cpu"

        if self.verbose:
            print(
                f"Using device: "
                f"{requested_device}"
            )

        return str(requested_device)

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def _save_checkpoint(
        self,
        epoch,
        score,
        metric_name,
        model_state,
        optimizer,
        has_validation
    ):
        if self.checkpoint_path is None:
            return

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model_state,
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_score": score,
                "checkpoint_metric":
                    metric_name,
                "classes":
                    self.classes_,
                "used_validation":
                    has_validation,
                "use_rms_normalization":
                    self.use_rms_normalization
            },
            self.checkpoint_path
        )

    # ========================================================
    # LOG
    # ========================================================

    def _print_epoch(
        self,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        best_score,
        best_epoch,
        metric_name
    ):
        message = (
            f"Epoch "
            f"[{epoch + 1}/{self.epochs}] - "
            f"Loss: {train_loss:.4f} - "
            f"Train Acc: "
            f"{train_accuracy:.4f}"
        )

        if val_loss is not None:
            message += (
                f" - Val Loss: "
                f"{val_loss:.4f}"
                f" - Val Acc: "
                f"{val_accuracy:.4f}"
            )

        message += (
            f" - Best {metric_name}: "
            f"{best_score:.4f}"
            f" - Best Epoch: "
            f"{best_epoch}"
        )

        print(message)


# ============================================================
# TRANSFORMADOR IDENTIDADE
# ============================================================

class _IdentityFeatures:
    """
    Mantém os segmentos de vibração sem extração explícita
    de características antes da CNN-LSTM.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# ============================================================
# MÉTODO
# ============================================================

class CNNLSTMMethod:
    """
    Apresenta a CNN-LSTM ao framework por meio da
    interface comum utilizada pelos métodos experimentais.
    """

    name = "cnn_lstm"

    def configurations(self):
        yield {}

    def build(self, configuration=None):
        steps = [
            (
                "feature_extraction",
                _IdentityFeatures()
            ),
            (
                "classifier",
                CNNLSTMClassifier()
            )
        ]

        return Pipeline(steps)

    def metadata(self, configuration=None):
        return {
            "method": self.name
        }