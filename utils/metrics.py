from sklearn.metrics import f1_score


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')