import copy
import random

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import (
    Dinov2WithRegistersConfig,
    Dinov2WithRegistersModel,
)


# Normalização utilizada pelos checkpoints DINOv2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DATASET
# ============================================================

class SpectrogramDataset(Dataset):

    def __init__(
        self,
        X,
        y=None,
        image_size=224
    ):
        self.X = np.asarray(X, dtype=object)

        self.y = None
        if y is not None:
            self.y = np.asarray(y, dtype=np.int64)

        self.transform = T.Compose([
            T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        image_path = str(self.X[index])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        if self.y is None:
            return image

        return image, int(self.y[index])


# ============================================================
# REDE DINOv2
# ============================================================

class DINOv2Net(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name="facebook/dinov2-with-registers-small",
        dropout=0.6,
        local_files_only=False
    ):
        super().__init__()

        config = Dinov2WithRegistersConfig.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )

        self.dinov2 = Dinov2WithRegistersModel.from_pretrained(
            model_name,
            config=config,
            attn_implementation="eager",
            local_files_only=local_files_only
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                self.dinov2.config.hidden_size,
                num_classes
            )
        )

    def forward(self, x):

        outputs = self.dinov2(
            pixel_values=x,
            output_attentions=False
        )

        # Mesmo procedimento empregado na implementação
        # disponibilizada por Cardoso:
        # média dos embeddings dos tokens
        x = outputs.last_hidden_state.mean(dim=1)

        x = self.classifier(x)

        return x


# ============================================================
# CLASSIFICADOR COMPATÍVEL COM SCIKIT-LEARN
# ============================================================

class DINOv2Classifier(
    ClassifierMixin,
    BaseEstimator
):

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

        self.early_stopping_patience = early_stopping_patience
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

        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        has_validation = (
            X_val is not None
            and y_val is not None
        )

        if has_validation:
            X_val = np.asarray(
                X_val,
                dtype=object
            )

            y_val = np.asarray(y_val)

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------
        print("Device:", self.device)
        if (
            self.device == "cuda"
            and not torch.cuda.is_available()
        ):            
            self.device_ = "cpu"

            if self.verbose:
                print(
                    "CUDA is not available. "
                    "Using CPU instead."
                )

        else:
            self.device_ = self.device


        # ----------------------------------------------------
        # CLASSES
        # ----------------------------------------------------

        self.classes_ = np.unique(y)

        self.class_to_index_ = {
            label: index
            for index, label
            in enumerate(self.classes_)
        }

        self.index_to_class_ = {
            index: label
            for label, index
            in self.class_to_index_.items()
        }

        y_encoded = np.asarray(
            [
                self.class_to_index_[label]
                for label in y
            ],
            dtype=np.int64
        )

        if has_validation:

            unknown_labels = (
                set(y_val)
                - set(self.classes_)
            )

            if unknown_labels:
                raise ValueError(
                    "Validation set contains labels "
                    "not present in training set: "
                    f"{unknown_labels}"
                )

            y_val_encoded = np.asarray(
                [
                    self.class_to_index_[label]
                    for label in y_val
                ],
                dtype=np.int64
            )


        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        self.model_ = DINOv2Net(
            num_classes=len(self.classes_),
            model_name=self.model_name,
            dropout=self.dropout,
            local_files_only=self.local_files_only
        ).to(self.device_)


        # ----------------------------------------------------
        # DATA LOADERS
        # ----------------------------------------------------

        train_dataset = SpectrogramDataset(
            X,
            y_encoded,
            image_size=self.image_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(
                self.device_ == "cuda"
            )
        )

        val_loader = None

        if has_validation:

            val_dataset = SpectrogramDataset(
                X_val,
                y_val_encoded,
                image_size=self.image_size
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(
                    self.device_ == "cuda"
                )
            )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()


        # ----------------------------------------------------
        # ADAMW
        # ----------------------------------------------------

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor
            )
        )


        # ----------------------------------------------------
        # ÉPOCAS
        #
        # Com validação:
        #     30 épocas
        #
        # Treinamento final:
        #     20 épocas
        # ----------------------------------------------------

        if has_validation:
            number_of_epochs = self.cv_epochs
        else:
            number_of_epochs = self.final_epochs


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None

        epochs_without_improvement = 0


        self.history_ = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }


        # ====================================================
        # TREINAMENTO
        # ====================================================

        for epoch in range(number_of_epochs):

            train_loss, train_accuracy = (
                self._train_one_epoch(
                    train_loader,
                    criterion,
                    optimizer
                )
            )

            self.history_[
                "train_loss"
            ].append(train_loss)

            self.history_[
                "train_accuracy"
            ].append(train_accuracy)


            # ------------------------------------------------
            # VALIDAÇÃO
            # ------------------------------------------------

            if has_validation:

                val_loss, val_accuracy = (
                    self._evaluate_loader(
                        val_loader,
                        criterion
                    )
                )

                self.history_[
                    "val_loss"
                ].append(val_loss)

                self.history_[
                    "val_accuracy"
                ].append(val_accuracy)


                # Scheduler acompanha validation loss
                scheduler.step(val_loss)


                improved = (
                    val_loss
                    <
                    best_val_loss
                    - self.min_delta
                )

                if improved:

                    best_val_loss = val_loss

                    best_epoch = epoch + 1

                    epochs_without_improvement = 0

                    best_model_state = copy.deepcopy(
                        self.model_.state_dict()
                    )


                    if self.checkpoint_path is not None:

                        torch.save(
                            {
                                "epoch": best_epoch,
                                "model_state_dict":
                                    best_model_state,
                                "optimizer_state_dict":
                                    optimizer.state_dict(),
                                "val_loss":
                                    best_val_loss,
                                "classes":
                                    self.classes_,
                                "class_to_index":
                                    self.class_to_index_,
                                "model_name":
                                    self.model_name
                            },
                            self.checkpoint_path
                        )

                else:

                    epochs_without_improvement += 1


            else:

                val_loss = None
                val_accuracy = None

                scheduler.step(train_loss)


            # ------------------------------------------------
            # LOG
            # ------------------------------------------------

            if self.verbose:

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                if has_validation:

                    print(
                        f"Epoch "
                        f"[{epoch + 1}/{number_of_epochs}] - "
                        f"Loss: {train_loss:.4f} - "
                        f"Train Acc: {train_accuracy:.4f} - "
                        f"Val Loss: {val_loss:.4f} - "
                        f"Val Acc: {val_accuracy:.4f} - "
                        f"LR: {current_lr:.2e} - "
                        f"Best Epoch: {best_epoch}"
                    )

                else:

                    print(
                        f"Epoch "
                        f"[{epoch + 1}/{number_of_epochs}] - "
                        f"Loss: {train_loss:.4f} - "
                        f"Train Acc: {train_accuracy:.4f} - "
                        f"LR: {current_lr:.2e}"
                    )


            # ------------------------------------------------
            # EARLY STOPPING
            #
            # Apenas quando existe conjunto de validação.
            # ------------------------------------------------

            if (
                has_validation
                and self.early_stopping_patience is not None
                and epochs_without_improvement
                    >= self.early_stopping_patience
            ):

                if self.verbose:

                    print(
                        f"Early stopping at epoch "
                        f"{epoch + 1}. "
                        f"Best epoch: {best_epoch} - "
                        f"Best Val Loss: "
                        f"{best_val_loss:.4f}"
                    )

                break


        # ----------------------------------------------------
        # RESTAURA MELHOR CHECKPOINT
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
            else number_of_epochs
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

        X = np.asarray(
            X,
            dtype=object
        )

        dataset = SpectrogramDataset(
            X,
            image_size=self.image_size
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(
                self.device_ == "cuda"
            )
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

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                predictions.extend(
                    preds.cpu().numpy()
                )


        return np.asarray(
            [
                self.index_to_class_[index]
                for index in predictions
            ]
        )


    # ========================================================
    # SCORE
    # ========================================================

    def score(
        self,
        X,
        y
    ):

        y_pred = self.predict(X)

        return np.mean(
            y_pred
            ==
            np.asarray(y)
        )


    # ========================================================
    # TREINAMENTO DE UMA ÉPOCA
    # ========================================================

    def _train_one_epoch(
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


            batch_size = xb.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == yb
            ).sum().item()

            total += batch_size


        average_loss = (
            total_loss / total
        )

        accuracy = (
            correct / total
        )

        return (
            average_loss,
            accuracy
        )


    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    def _evaluate_loader(
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


                batch_size = xb.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    predictions == yb
                ).sum().item()

                total += batch_size


        average_loss = (
            total_loss / total
        )

        accuracy = (
            correct / total
        )

        return (
            average_loss,
            accuracy
        )


    # ========================================================
    # REPRODUTIBILIDADE
    # ========================================================

    def _set_seed(self):

        random.seed(
            self.random_state
        )

        np.random.seed(
            self.random_state
        )

        torch.manual_seed(
            self.random_state
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                self.random_state
            )