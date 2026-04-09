from dataset.utils import get_folds


def train_test_split(list_of_filters, test_fold_key):
    test_filter = list_of_filters[test_fold_key]
    train_filters = []
    for key in list_of_filters:
        if key != test_fold_key:
            train_filters.extend(list_of_filters[key])
    return train_filters, test_filter


def holdout(model, list_of_filters, test_fold_key, list_of_metrics):
    train_filters, test_filter = train_test_split(list_of_filters, test_fold_key)
    model.train(train_filters)
    scores = model.evaluate(test_filter, list_of_metrics)
    return scores


def cross_validation(model, folds, list_of_metrics):
    scores = {}
    for test_fold_key in folds:
        fold_scores = holdout(model, folds, test_fold_key, list_of_metrics)
        scores[test_fold_key] = fold_scores
    return scores


def run_experiment(model, config_file, experimental_setup, load_function, list_of_metrics):
    scores = {}
    model.set_load_function(load_function)
    for combination_key in experimental_setup:
        folds = get_folds(experimental_setup, combination_key, config_file)
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
