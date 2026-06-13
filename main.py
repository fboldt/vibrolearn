import argparse
from datetime import datetime
import json
from os import path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from estimators.cnn_lstm import CNNLSTMClassifier

from estimators.pipeline import Pipeline
from experiment.assesment import run_experiment, save_scores
from feature.extraction import *
from experiment.compile_results import compile_results, compile_results_across_folds_and_domains, generate_paired_augmentation_boxplots

from feature.feature_selector import StableDomainFeatureSelector

ALPHA_VALUES = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
def save_experiment_results(args, pipeline, alpha):
    results = {}
    results["experiment_name"] = path.basename(args.experimental_setup).split('.')[0]
    results["feature_extraction"] = pipeline.pipe.named_steps["feature_extraction"].__class__.__name__
    results["classifier"] = pipeline.pipe.named_steps["classifier"].__class__.__name__
    results["start_time"] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experimental_setup = json.load(open(args.experimental_setup, "r"))
    list_of_scores = run_experiment(pipeline, experimental_setup)
    results["scores"] = list_of_scores
    results["end_time"] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f"results_paper/sdfs_k_16_alpha_{alpha}_{results['end_time']}.json"
    save_scores(results, output_file)


if __name__ == "__main__":
    for alpha in ALPHA_VALUES:
        print(f"Running experiments with alpha: {alpha}")
        parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
        parser.add_argument("-f", "--feature_extraction", type=str, help="The feature extraction method to use for the experiments (choices: FlattenFeatures, StatisticalFeatures, HeterogeneousFeatures, WaveletFeatures, RMSFeatures)")
        parser.add_argument("-c", "--classifier", type=str, help="The classifier to use for the experiments (choices: RandomForestClassifier, KNeighborsClassifier, CNNLSTMClassifier)")
        parser.add_argument("-e", "--experimental_setup", type=str, help="The experimental setup file to run (mandatory)")
        parser.add_argument("-r", "--results_directory", type=str, nargs="?", const="results/", help="The directory to compile the results from the experiments (default: results/)")

        args = parser.parse_args()
        
        if not any(vars(args).values()):
            parser.print_help()
        
        steps = []    

        featextraction_method = WaveletFeatures()
        if args.feature_extraction:
            featextraction_method = eval(args.feature_extraction)()
        print(f"Using feature extraction method: {featextraction_method.__class__.__name__}")
        steps.append(("feature_extraction", featextraction_method))

        # steps.append(("feature_selection", SelectKBest(score_func=mutual_info_classif, k=8)))    
        steps.append(("feature_selection", StableDomainFeatureSelector(k=16, score_func="mutual_info", alpha=alpha)))

        model = RandomForestClassifier()
        if args.classifier:
            if args.classifier == "CNNLSTMClassifier":
                model = CNNLSTMClassifier()
            else:
                model = eval(args.classifier)()

        print(f"Using classifier: {model.__class__.__name__}")
        steps.append(("classifier", model))

        pipe = Pipeline(steps)

        if args.experimental_setup:
            print(f"Running experimental setup: {args.experimental_setup}")
            save_experiment_results(args, pipe, str(int(alpha*10)))
        elif not args.experimental_setup and not args.results_directory:
            print("No experimental setup specified, please provide one using the -e or --experimental_setup argument")

        if args.results_directory:
            print(f"Compiling results from directory: {args.results_directory}")
            output_file = compile_results(args.results_directory)
            print(f"Compiled results saved to: {output_file}")
            output_file = compile_results_across_folds_and_domains(args.results_directory)
            print(f"Compiled results across folds and domains saved to: {output_file}")
            output_file = generate_paired_augmentation_boxplots(args.results_directory)
            print(f"Generated paired augmentation box plots across folds and domains saved to: {output_file}")
        