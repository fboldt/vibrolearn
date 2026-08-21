import json
import os
from dataset.utils import get_folds
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def get_list_of_metrics():
    return [accuracy_score, f1_macro, confusion_matrix]


def train_test_split(filters, test_fold_key):
    test_filter = filters[test_fold_key]
    train_filters = []
    for key in filters:
        if key != test_fold_key:
            train_filters.extend(filters[key])
    return train_filters, test_filter


def holdout(model, experimental_setup, combination_key, test_fold_key, list_of_metrics):
    folds = get_folds(experimental_setup, combination_key, experimental_setup["config_file"])
    exp_setup = experimental_setup.copy()
    del exp_setup["setup"]
    train_filters, test_filter = train_test_split(folds, test_fold_key)
    model.train(train_filters, exp_setup)
    scores = model.evaluate(test_filter, list_of_metrics)
    print_dict_of_scores(scores)
    return scores


def multiple_holdout(model, experimental_setup, list_of_metrics):
    scores = {}
    for combination_key in experimental_setup["setup"]:
        print(f"Running holdout for combination: {combination_key}")
        scores[combination_key] = holdout(model, experimental_setup, combination_key,  test_fold_key="testing", list_of_metrics=list_of_metrics)
    return scores


def cross_validation(model, experimental_setup, list_of_metrics):
    scores = {}
    for combination_key in experimental_setup["setup"]:
        folds = get_folds(experimental_setup, combination_key, experimental_setup["config_file"])
        for test_fold_key in folds:
            print(f"Running cross-validation for combination: {combination_key}, fold: {test_fold_key}")
            fold_scores = holdout(model, experimental_setup, combination_key, test_fold_key, list_of_metrics)
            scores[test_fold_key] = fold_scores
    return scores


def run_experiment(model, experimental_setup):
    list_of_metrics = get_list_of_metrics()
    scores = {}
    if experimental_setup["type"] == "train_test_split":
        scores = multiple_holdout(model, experimental_setup, list_of_metrics)
    elif experimental_setup["type"] == "cross_validation":
        scores = scores | cross_validation(model, experimental_setup,  list_of_metrics)
    return scores


def print_dict_of_scores(scores):
    print(20 * "-")
    for metric_name, score in scores.items():
        print(f"-- {metric_name}:", end="\t")
        if metric_name == "confusion_matrix":
            print()
        print(score)
    print(10 * "# ")


def save_scores(scores, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print(f"Saved scores to: {output_file}")

