from dataset.utils import get_X_y, get_folds, load_matlab_acquisition


def train_test_split(filters, test_fold_key):
    test_filter = filters[test_fold_key]
    train_filters = []
    for key in filters:
        if key != test_fold_key:
            train_filters.extend(filters[key])
    return train_filters, test_filter


def holdout(model, filters, test_fold_key, list_of_metrics):
    train_filters, test_filter = train_test_split(filters, test_fold_key)
    model.train(train_filters)
    scores = model.evaluate(test_filter, list_of_metrics)
    return scores


def cross_validation(model, folds, list_of_metrics):
    scores = {}
    for test_fold_key in folds:
        fold_scores = holdout(model, folds, test_fold_key, list_of_metrics)
        scores[test_fold_key] = fold_scores
    return scores


def load_function(registers, experimental_setup):
    raw_dir_path=experimental_setup["raw_dir_path"]
    channels_columns=experimental_setup["channels_columns"]
    segment_length=experimental_setup["segment_length"]
    load_acquisition_func=eval(experimental_setup["load_acquisition_func"])
    X, y = get_X_y(registers, 
                   raw_dir_path=raw_dir_path, 
                   channels_columns=channels_columns, 
                   segment_length=segment_length, 
                   load_acquisition_func=load_acquisition_func)
    return X, y


def run_experiment(model, experimental_setup, list_of_metrics):
    model.set_load_function(lambda registers: load_function(registers, experimental_setup))
    scores = {}
    if experimental_setup["type"] == "train_test_split":
        scores["testing"] = perform_holdout_experiment(model, experimental_setup, list_of_metrics, scores)
    elif experimental_setup["type"] == "cross_validation":
        scores = perform_experimental_cross_validation(model, experimental_setup, list_of_metrics, scores)
    return scores


def perform_holdout_experiment(model, experimental_setup, list_of_metrics, scores):
    for combination_key in experimental_setup["setup"]:
        folds = get_folds(experimental_setup["setup"], combination_key, experimental_setup["config_file"])
        holdout_scores = holdout(model, folds, test_fold_key="testing", list_of_metrics=list_of_metrics)
        scores = scores | holdout_scores
    return scores


def perform_experimental_cross_validation(model, experimental_setup, list_of_metrics, scores):
    for combination_key in experimental_setup["setup"]:
        folds = get_folds(experimental_setup["setup"], combination_key, experimental_setup["config_file"])
        fold_scores = cross_validation(model, folds, list_of_metrics)
        scores = scores | fold_scores
    return scores


def print_dict_of_scores(scores):
    print(20 * "-")
    for metric_name, score in scores.items():
        print(f"-- {metric_name} --\n{score}\n")


def print_scores_list(scores):
    for key in scores:
        print(f"### {key}:")
        print_dict_of_scores(scores[key])
