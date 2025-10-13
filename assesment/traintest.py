def performance(model, X_train, y_train, X_test, y_test, list_metrics):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    scores = {}
    for metric in list_metrics:
        scores[metric] = metric(y_test, y_test_pred)
    return scores

def print_scores(scores):
    for metric, score in scores.items():
        print(f"{metric.__name__}:\n{score}")
