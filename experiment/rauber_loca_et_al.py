from utils.metrics import f1_macro
from utils.assesment import print_scores_list


def run_rauber_loca_et_al_experiment(model):
    print("Running rauber_loca_et_al experiment")
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.rauber_loca_et_al import run_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    list_of_scores = run_experiment(model, list_of_metrics=list_of_metrics)
    print_scores_list(list_of_scores)

