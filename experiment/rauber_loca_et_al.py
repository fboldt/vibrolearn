from utils.metrics import f1_macro
from utils.assesment import print_scores_list, run_experiment
from sklearn.metrics import accuracy_score, confusion_matrix
import json


list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]

def run_rauber_loca_et_al_experiment(model):
    print("Running rauber_loca_et_al experiment")

    experimental_setup = json.load(open("dataset/cwru/rauber_loca_et_al_setup.json", "r"))

    list_of_scores = run_experiment(model, experimental_setup, list_of_metrics)
    print_scores_list(list_of_scores)

