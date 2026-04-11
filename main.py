import json
from estimators.wavelet_random_forest import WaveletRandomForest
from experiment.assesment import print_scores_list, run_experiment

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("-c", "--classifier", type=str, help="The classifier to use for the experiments (choices: WaveletRandomForest)")
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
    elif not args.classifier:
        model = WaveletRandomForest(42)

    if args.experimental_setup:
        print(f"Running experimental setup: {args.experimental_setup}")
        experimental_setup = json.load(open(args.experimental_setup, "r"))
        list_of_scores = run_experiment(model, experimental_setup)
        print_scores_list(list_of_scores)

