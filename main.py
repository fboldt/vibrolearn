from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from estimator.SFRF import SFRF as Estimator

model = Estimator()

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]


if __name__ == "__main__":
    #'''
    from dataset.cwru.sehri_et_al import run_papers_experiment
    scores = run_papers_experiment(model, list_of_metrics)
    print("Scores for papers experiment:")
    pprint(scores)
    ''' 
    from dataset.cwru.sehri_et_al import run_proposed_experiment
    run_proposed_experiment(model, list_of_metrics)
    #'''
