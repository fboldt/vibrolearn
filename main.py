import argparse
from datetime import datetime
import json
from os import path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from estimators.cnn_lstm import CNNLSTMClassifier

from estimators.pipeline import Pipeline
from dataset.loader import augmented
from experiment.assesment import run_experiment, save_scores
from feature.extraction import *
from experiment.compile_results import compile_results, compile_results_across_folds_and_domains, generate_paired_augmentation_boxplots


def save_experiment_results(args, pipeline):
    results = {}
    results["experiment_name"] = path.basename(args.experimental_setup).split('.')[0]
    results["feature_extraction"] = pipeline.pipe.named_steps["feature_extraction"].__class__.__name__
    results["classifier"] = pipeline.pipe.named_steps["classifier"].__class__.__name__
    results["augmentation"] = args.augmentation
    results["start_time"] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experimental_setup = json.load(open(args.experimental_setup, "r"))
    list_of_scores = run_experiment(pipeline, experimental_setup)
    results["scores"] = list_of_scores
    results["end_time"] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    augmented_str = "aug" if args.augmentation else ""
    output_file = f"results/{results['experiment_name']}_{augmented_str}_{results['end_time']}.json"
    save_scores(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("-a", "--augmentation", action="store_true", help="Whether to use data augmentation in the experiments")
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

    model = RandomForestClassifier()
    if args.classifier:
        if args.classifier == "CNNLSTMClassifier":
            model = CNNLSTMClassifier(
                epochs=100,
                batch_size=64,
                learning_rate=0.001,
                device="cuda",
                early_stopping_patience=20,
                min_delta=1e-4,
                checkpoint_path="best_cnn_lstm.pt",
                verbose=True
            )
        else:
            model = eval(args.classifier)()

    print(f"Using classifier: {model.__class__.__name__}")
    steps.append(("classifier", model))

    pipe = Pipeline(steps)

    if args.augmentation:
        print("Using data augmentation")
        pipe.train_loader = augmented

    if args.experimental_setup:
        print(f"Running experimental setup: {args.experimental_setup}")
        save_experiment_results(args, pipe)
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
        