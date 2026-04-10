import json
from sklearn.metrics import accuracy_score, confusion_matrix
from estimators.wavelet_random_forest import WaveletRandomForest
from utils.assesment import print_scores_list, run_experiment
from utils.metrics import f1_macro

list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("-c", "--classifier", type=str, help="The classifier to use for the experiments (mandatory) choices: WaveletRandomForest")
    parser.add_argument("-e", "--experimental_setup", type=str, help="The experimental setup file to run (mandatory)")
    
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
    
    model = None
    if args.debug:
        print("Running in debug mode")

    if args.classifier:
        model = eval(args.classifier)()
        print(f"Using classifier: {model.__class__.__name__}")

    if args.experimental_setup:
        print(f"Running experimental setup: {args.experimental_setup}")
        experimental_setup = json.load(open(args.experimental_setup, "r"))
        list_of_scores = run_experiment(model, experimental_setup, list_of_metrics)
        print_scores_list(list_of_scores)

