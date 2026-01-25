from dataset.cwru.rauber_loca_et_al import run_experiment
from dataset.cwru.sehri_et_al import run_papers_experiment, run_proposed_experiment
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from estimator.WPRF import WPRF as Estimator
from pprint import pprint

model = Estimator()

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]


if __name__ == "__main__":
    scores = run_papers_experiment(model, list_of_metrics)
    # scores = run_proposed_experiment(model, list_of_metrics)
    print("\n\nFinal Scores:")
    pprint(scores)
