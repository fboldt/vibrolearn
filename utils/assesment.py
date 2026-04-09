def train_test_split(list_of_filters, test_fold_index):
    test_filter = list_of_filters[test_fold_index]
    train_filters = []
    for i in range(len(list_of_filters)):
        if i != test_fold_index:
            train_filters.extend(list_of_filters[i])
    return train_filters, test_filter


def holdout(model, list_of_filters, test_fold_index, list_of_metrics):
    train_filters, test_filter = train_test_split(list_of_filters, test_fold_index)
    model.train(train_filters)
    scores = model.evaluate(test_filter, list_of_metrics)
    return scores


def print_dict_of_scores(scores):
    print(20 * "-")
    for metric_name, score in scores.items():
        print(f"-- {metric_name} --\n{score}\n")


def print_scores_list(scores):
    for i, fold in enumerate(scores):
        print(f"### Fold {i + 1}:")
        print_dict_of_scores(fold)
