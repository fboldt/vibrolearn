from dataset.utils import get_train_test_split
from assesment import traintest

def performance(model, list_of_X_y, list_metrics):
    n_folds = len(list_of_X_y)
    scores_per_fold = []
    for i in range(n_folds):
        X_train, y_train, X_test, y_test = get_train_test_split(list_of_X_y, test_fold_index=i)
        scores = traintest.performance(model, X_train, y_train, X_test, y_test, list_metrics)
        scores_per_fold.append(scores)
    return scores_per_fold

def print_scores_per_fold(scores_per_fold):
    for test_fold, scores in enumerate(scores_per_fold):
        print(f"Scores for fold {test_fold}:")
        traintest.print_scores(scores)
        print()
