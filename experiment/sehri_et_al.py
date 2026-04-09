from utils.metrics import f1_macro
from utils.assesment import print_dict_of_scores, print_scores_list


def run_sehri_et_al_papers(model):
    print("Running sehri_et_al_papers experiment")
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.sehri_et_al import run_papers_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    scores = run_papers_experiment(model,list_of_metrics=list_of_metrics)
    print_dict_of_scores(scores)


def run_sehri_et_al_papers_inspired_experiment(model):
    print("Running sehri_et_al_papers_inspired_experiment")
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.sehri_et_al import run_papers_inspired_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    list_of_scores = run_papers_inspired_experiment(model,list_of_metrics=list_of_metrics)
    print_scores_list(list_of_scores)


def run_sehri_et_al_proposed_experiment(model):
    print("Running sehri_et_al_proposed_experiment")
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset.cwru.sehri_et_al import run_proposed_experiment
    list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
    list_of_scores = run_proposed_experiment(model,list_of_metrics=list_of_metrics)
    print_scores_list(list_of_scores)
