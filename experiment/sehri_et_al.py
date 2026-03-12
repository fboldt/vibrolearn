import itertools
import time
from pprint import pprint

from sklearn.metrics import accuracy_score, confusion_matrix

from dataset.cwru.sehri_et_al import (
    get_list_of_papers_inspired_cross_validation_folds,
    papers_inspired_cross_validation_folds,
    segment_length,
)
from dataset.utils import get_list_of_X_y, load_matlab_acquisition
from utils.metrics import f1_macro

list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]

default_cnn_tuning_grid = {
    "n_filters": [16, 32],
    "kernel_size": [3, 7],
    "hidden_dim": [32, 64],
    "dropout": [0.1, 0.3],
    "epochs": [5, 10],
    "batch_size": [64],
    "learning_rate": [1e-3, 5e-4],
    "weight_decay": [0.0, 1e-4],
    "device": [None],
    "random_state": [42],
    "verbose": [False],
}

def run_sehri_experiment(model):
    from dataset.cwru.sehri_et_al import run_papers_experiment

    scores = run_papers_experiment(model, list_of_metrics)
    print("Scores for papers experiment:")
    pprint(scores)


def run_inspired_experiment(model):
    from dataset.cwru.sehri_et_al import run_papers_inspired_experiment

    scores = run_papers_inspired_experiment(model, list_of_metrics)
    print("Scores for papers inspired experiment:")
    pprint(scores)


def run_proposed_experiment(model):
    from dataset.cwru.sehri_et_al import run_proposed_experiment as run_proposed_dataset_experiment

    scores = run_proposed_dataset_experiment(model, list_of_metrics)
    print("Scores for proposed experiment:")
    pprint(scores)


def _iter_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def _count_param_combinations(param_grid):
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def run_inspired_cnn_tuning_cv3(
    model_class,
    param_grid,
    list_of_metrics,
    tuning_metric,
    verbose=False,
    log_every=1,
):
    list_of_folds = get_list_of_papers_inspired_cross_validation_folds()
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition
    )

    if len(list_of_X_y) != 3:
        raise ValueError("Inspired CNN tuning experiment expects exactly 3 folds.")

    split_scores = []
    fold_names = papers_inspired_cross_validation_folds
    n_param_combinations = _count_param_combinations(param_grid)
    n_splits = 6

    if verbose:
        print("Starting inspired CNN tuning (CV3 train/val/test rotations).")
        print(f"- parameter combinations: {n_param_combinations}")
        print(f"- total split evaluations: {n_splits}")

    split_index = 0
    for test_idx in range(3):
        remaining = [idx for idx in range(3) if idx != test_idx]
        role_pairs = [(remaining[0], remaining[1]), (remaining[1], remaining[0])]
        for train_idx, val_idx in role_pairs:
            split_index += 1
            X_train, y_train = list_of_X_y[train_idx]
            X_val, y_val = list_of_X_y[val_idx]
            X_test, y_test = list_of_X_y[test_idx]

            best_params = None
            best_val_score = float("-inf")
            split_start = time.perf_counter()

            if verbose:
                print(
                    f"[split {split_index}/{n_splits}] "
                    f"train={fold_names[train_idx]} val={fold_names[val_idx]} test={fold_names[test_idx]}"
                )

            for combo_index, params in enumerate(_iter_param_combinations(param_grid), start=1):
                model = model_class(**params)
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                val_score = tuning_metric(y_val, y_val_pred)
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_params = params
                    if verbose:
                        print(
                            f"  new best @ combo {combo_index}/{n_param_combinations}: "
                            f"val={best_val_score:.6f} params={best_params}"
                        )
                elif verbose and (combo_index % max(1, log_every) == 0):
                    print(
                        f"  combo {combo_index}/{n_param_combinations}: "
                        f"val={val_score:.6f} best={best_val_score:.6f}"
                    )

            best_model = model_class(**best_params)
            best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)

            test_scores = {}
            for metric in list_of_metrics:
                test_scores[metric.__name__] = metric(y_test, y_test_pred)

            if verbose:
                split_duration = time.perf_counter() - split_start
                print(
                    f"[split {split_index}/{n_splits}] done in {split_duration:.2f}s "
                    f"best_val={best_val_score:.6f} test_f1_macro={test_scores.get('f1_macro')}"
                )

            split_scores.append(
                {
                    "train_fold": fold_names[train_idx],
                    "validation_fold": fold_names[val_idx],
                    "test_fold": fold_names[test_idx],
                    "best_validation_score": best_val_score,
                    "best_params": best_params,
                    "test_scores": test_scores,
                }
            )

    return split_scores


def run_inspired_cnn_tuning_experiment(param_grid=None, verbose=False, log_every=1):
    from estimator.CNN1D import CNN1D

    scores = run_inspired_cnn_tuning_cv3(
        model_class=CNN1D,
        param_grid=param_grid or default_cnn_tuning_grid,
        list_of_metrics=list_of_metrics,
        tuning_metric=f1_macro,
        verbose=verbose,
        log_every=log_every,
    )
    print("Scores for inspired CNN tuning experiment:")
    pprint(scores)
