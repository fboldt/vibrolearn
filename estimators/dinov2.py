import copy
import random

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import Dinov2WithRegistersModel

from preprocessing.spectrogram_adapter import (
    SpectrogramAdapter
)
from estimators.pipeline import Pipeline


# ============================================================
# CONSTANTES
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DATASET
# ============================================================

class _SpectrogramDataset(Dataset):
    """
    Dataset interno responsável pelo carregamento e
    pré-processamento dos espectrogramas utilizados pelo DINOv2.
    """

    def __init__(self, X, y=None, image_size=224):
        self.X = np.asarray(X, dtype=object)
        self.y = (
            None
            if y is None
            else np.asarray(y, dtype=np.int64)
        )

        self.transform = T.Compose([
            T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image_path = str(self.X[index])

        with Image.open(image_path) as image:
            image = self.transform(
                image.convert("RGB")
            )

        if self.y is None:
            return image

        return image, self.y[index]


# ============================================================
# REDE
# ============================================================

class _DINOv2Net(nn.Module):
    """
    Backbone DINOv2 seguido por uma camada de classificação.
    """

    def __init__(
        self,
        num_classes,
        model_name,
        dropout,
        local_files_only=False
    ):
        super().__init__()

        self.backbone = (
            Dinov2WithRegistersModel.from_pretrained(
                model_name,
                attn_implementation="eager",
                local_files_only=local_files_only
            )
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.backbone.config.hidden_size,
                num_classes
            )
        )

    def forward(self, x):
        outputs = self.backbone(
            pixel_values=x,
            output_attentions=False
        )

        # Média dos embeddings dos tokens,
        # conforme implementação adotada no trabalho.
        embeddings = (
            outputs.last_hidden_state.mean(dim=1)
        )

        return self.classifier(embeddings)


# ============================================================
# CLASSIFICADOR
# ============================================================

class DINOv2Classifier(
    ClassifierMixin,
    BaseEstimator
):
    """
    Classificador DINOv2 compatível com a interface
    de estimadores do Scikit-learn.
    """

    def __init__(
        self,
        model_name="facebook/dinov2-with-registers-small",
        cv_epochs=30,
        final_epochs=15,
        batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        dropout=0.6,
        early_stopping_patience=3,
        min_delta=0.0,
        scheduler_factor=0.3,
        image_size=224,
        device="cuda",
        num_workers=2,
        random_state=42,
        checkpoint_path=None,
        local_files_only=False,
        verbose=True
    ):
        self.model_name = model_name

        self.cv_epochs = cv_epochs
        self.final_epochs = final_epochs

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.early_stopping_patience = (
            early_stopping_patience
        )
        self.min_delta = min_delta
        self.scheduler_factor = scheduler_factor

        self.image_size = image_size
        self.device = device
        self.num_workers = num_workers

        self.random_state = random_state
        self.checkpoint_path = checkpoint_path
        self.local_files_only = local_files_only

        self.verbose = verbose

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
        self._set_seed()
        self.device_ = self._resolve_device()

        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        has_validation = (
            X_val is not None
            and y_val is not None
        )

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
            X_val = np.asarray(
                X_val,
                dtype=object
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

        self.model_ = _DINOv2Net(
            num_classes=len(self.classes_),
            model_name=self.model_name,
            dropout=self.dropout,
            local_files_only=self.local_files_only
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

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = (
            torch.optim.lr_scheduler
            .ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor
            )
        )

        num_epochs = (
            self.cv_epochs
            if has_validation
            else self.final_epochs
        )

        self.history_ = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None
        epochs_without_improvement = 0

        # ====================================================
        # TREINAMENTO
        # ====================================================

        for epoch in range(num_epochs):

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
            self.history_["train_accuracy"].append(
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

                self.history_["val_loss"].append(
                    val_loss
                )
                self.history_["val_accuracy"].append(
                    val_accuracy
                )

                scheduler.step(val_loss)

                improved = (
                    val_loss
                    < best_val_loss - self.min_delta
                )

                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1

                    epochs_without_improvement = 0

                    best_model_state = copy.deepcopy(
                        self.model_.state_dict()
                    )

                    self._save_checkpoint(
                        best_epoch,
                        best_val_loss,
                        best_model_state,
                        optimizer
                    )

                else:
                    epochs_without_improvement += 1

            else:
                scheduler.step(train_loss)

            # ------------------------------------------------
            # Log
            # ------------------------------------------------

            if self.verbose:
                self._print_epoch(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    optimizer=optimizer,
                    best_epoch=best_epoch
                )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                has_validation
                and self.early_stopping_patience
                is not None
                and epochs_without_improvement
                >= self.early_stopping_patience
            ):
                if self.verbose:
                    print(
                        "Early stopping at epoch "
                        f"{epoch + 1}. "
                        f"Best epoch: {best_epoch} - "
                        "Best Val Loss: "
                        f"{best_val_loss:.4f}"
                    )

                break

        # ----------------------------------------------------
        # Recupera melhor modelo
        # ----------------------------------------------------

        if (
            has_validation
            and best_model_state is not None
        ):
            self.model_.load_state_dict(
                best_model_state
            )

        self.best_epoch_ = (
            best_epoch
            if has_validation
            else num_epochs
        )

        self.best_score_ = (
            best_val_loss
            if has_validation
            else train_loss
        )

        self.used_validation_ = has_validation

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):
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

                indices = torch.argmax(
                    outputs,
                    dim=1
                )

                predictions.extend(
                    indices.cpu().numpy()
                )

        # O índice produzido pela rede corresponde diretamente
        # à posição em self.classes_.
        return self.classes_[
            np.asarray(predictions)
        ]

    # ========================================================
    # DATA LOADER
    # ========================================================

    def _create_loader(
        self,
        X,
        y=None,
        shuffle=False
    ):
        dataset = _SpectrogramDataset(
            X,
            y,
            image_size=self.image_size
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
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
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            batch_size = yb.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            correct += (
                outputs.argmax(dim=1) == yb
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
                loss = criterion(outputs, yb)

                batch_size = yb.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                correct += (
                    outputs.argmax(dim=1) == yb
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
        val_loss,
        model_state,
        optimizer
    ):
        if self.checkpoint_path is None:
            return

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss": val_loss,
                "classes": self.classes_,
                "model_name": self.model_name,
            },
            self.checkpoint_path
        )

    # ========================================================
    # LOG
    # ========================================================

    def _print_epoch(
        self,
        epoch,
        num_epochs,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        optimizer,
        best_epoch
    ):
        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        message = (
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Loss: {train_loss:.4f} - "
            f"Train Acc: {train_accuracy:.4f}"
        )

        if val_loss is not None:
            message += (
                f" - Val Loss: {val_loss:.4f}"
                f" - Val Acc: {val_accuracy:.4f}"
                f" - Best Epoch: {best_epoch}"
            )

        message += (
            f" - LR: {current_lr:.2e}"
        )

        print(message)

    # ========================================================
    # REPRODUTIBILIDADE
    # ========================================================

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.random_state
            )


# ============================================================
# PIPELINE DO MÉTODO
# ============================================================

class DINOv2Method:

    name = "dinov2"

    def configurations(self):
        yield {}

    def build(
        self,
        configuration=None
    ):
        data_adapter = (
            SpectrogramAdapter(
                segment_length=12000,
                normalization="rms",
                nperseg=1024,
                noverlap=896,
                nfft=2048,
                db_min=-100.0,
                db_max=0.0,
                image_size=(224, 224),
                cache_dir=(
                    "cache/spectrograms"
                )
            )
        )

        steps = [
            (
                "classifier",
                DINOv2Classifier()
            )
        ]

        return Pipeline(
            steps=steps,
            data_adapter=data_adapter
        )

    def metadata(
        self,
        configuration=None
    ):
        return {
            "method": self.name
        }