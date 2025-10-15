from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from assesment.crossvalidation import  print_scores_per_fold
from dataset.cwru.experiment import rauber_loca_et_al


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def run(model, verbose=False):
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores_per_fold = rauber_loca_et_al(model, list_of_metrics, verbose=verbose)
    return scores_per_fold
    