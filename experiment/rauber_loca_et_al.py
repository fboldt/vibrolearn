from utils.metrics import f1_macro
from experiment.sehri_et_al import print_fold_scores


def run_rauber_loca_et_al_experiment(model):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.rauber_loca_et_al import run_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    list_of_scores = run_experiment(model, list_of_metrics=list_of_metrics)
    print_fold_scores(list_of_scores)

