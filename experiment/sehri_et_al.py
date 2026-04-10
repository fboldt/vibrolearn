from utils.metrics import f1_macro
from utils.assesment import print_dict_of_scores, print_scores_list, run_experiment
from sklearn.metrics import accuracy_score, confusion_matrix
import json


list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]


def run_sehri_et_al_papers(model):
    print("Running sehri_et_al_papers experiment")
    experimental_setup = json.load(open("dataset/cwru/sehri_et_al_papers_setup.json", "r"))
    scores = run_experiment(model, experimental_setup, list_of_metrics)
    print_dict_of_scores(scores)


def run_sehri_et_al_papers_inspired_experiment(model):
    print("Running sehri_et_al_papers_inspired_experiment")
    experimental_setup = json.load(open("dataset/cwru/sehri_et_al_inspired_setup.json", "r"))
    list_of_scores = run_experiment(model, experimental_setup, list_of_metrics)
    print_scores_list(list_of_scores)


def run_sehri_et_al_proposed_experiment(model):
    print("Running sehri_et_al_proposed_experiment")
    experimental_setup = json.load(open("dataset/cwru/sehri_et_al_proposed_setup.json", "r"))
    list_of_scores = run_experiment(model, experimental_setup, list_of_metrics)
    print_scores_list(list_of_scores)
