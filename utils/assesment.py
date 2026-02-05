from dataset.utils import concatenate_data
from utils.metrics import f1_macro

def train_test_split(list_of_X_y, test_fold_index):
    X_test, y_test = list_of_X_y[test_fold_index]
    X_train, y_train = concatenate_data([list_of_X_y[i] for i in range(len(list_of_X_y)) if i != test_fold_index])
    return (X_train, y_train), (X_test, y_test)

def holdout(model, list_of_X_y, test_fold_index, list_of_metrics):
    (X_train, y_train), (X_test, y_test) = train_test_split(list_of_X_y, test_fold_index)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    scores = {}
    for metric in list_of_metrics:
        scores[metric.__name__] = metric(y_test, y_test_pred)
    return scores

def cross_validation(model, list_of_X_y, list_of_metrics):
    n_folds = len(list_of_X_y)
    scores_per_fold = []
    for i in range(n_folds):
        scores = holdout(model, list_of_X_y, test_fold_index=i, list_of_metrics=list_of_metrics)
        scores_per_fold.append(scores)
    return scores_per_fold

def get_best_params(model, params, list_of_X_y):
    best_score = 0
    best_params = None
    for param_combination in params:
        model.set_params(**param_combination)
        scores_per_fold = cross_validation(model, list_of_X_y, list_of_metrics=[f1_macro])
        avg_score = sum([scores['f1_macro'] for scores in scores_per_fold]) / len(scores_per_fold)
        if avg_score > best_score:
            best_score = avg_score
            best_params = param_combination
    return best_params

def nested_cross_validation(model, params, list_of_X_y, list_of_metrics):
    n_folds = len(list_of_X_y)
    scores_per_fold = []
    for test_fold_out_index in range(n_folds):
        X_test_out, y_test_out = list_of_X_y[test_fold_out_index]
        list_of_X_y_out =  [list_of_X_y[i] for i in range(len(list_of_X_y)) if i != test_fold_out_index]
        best_params = get_best_params(model, params, list_of_X_y_out)
        model.set_params(**best_params)
        model.fit(*concatenate_data(list_of_X_y_out))
        y_test_pred = model.predict(X_test_out)
        scores = {}
        for metric in list_of_metrics:
            scores[metric.__name__] = metric(y_test_out, y_test_pred)
        scores_per_fold.append(scores)
    return scores_per_fold
    