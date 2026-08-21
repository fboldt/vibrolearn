import argparse
from datetime import datetime
import json
from os import path

# spectrograms
from sklearn.base import BaseEstimator, TransformerMixin

from estimators.dinov2 import DINOv2Classifier
from dataset.loader import spectrogram_vanilla


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from estimators.cnn_lstm import CNNLSTMClassifier

from estimators.pipeline import Pipeline
from experiment.assesment import run_experiment, save_scores
from feature.extraction import *
from experiment.compile_results import compile_results, compile_results_across_folds_and_domains, generate_paired_augmentation_boxplots

from feature.feature_selector import StableDomainFeatureSelector

ALPHA_VALUES = [0.7]
k = [8]
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
    # output_file = f"results_paper/validation_sdfs/k-{k_value}/alpha-{alpha}_{results['end_time']}.json"
    output_file = f"results_paper/dinov2/sehri_et_al_proprosed_setup_normal/{results['end_time']}.json"

    save_scores(results, output_file)



class IdentityFeatures(
    BaseEstimator,
    TransformerMixin
):

    def fit(
        self,
        X,
        y=None
    ):
        return self

    def transform(
        self,
        X
    ):
        return np.asarray(
            X,
            dtype=object
        )



if __name__ == "__main__":
    for _ in range(5):
        for k_value in k:
            for alpha in ALPHA_VALUES:
                print(f"Running experiments with alpha: {alpha}")
                parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
                parser.add_argument("-f", "--feature_extraction", type=str, help="The feature extraction method to use for the experiments (choices: FlattenFeatures, StatisticalFeatures, HeterogeneousFeatures, WaveletFeatures, RMSFeatures)")
                parser.add_argument(
                    "-c",
                    "--classifier",
                    type=str,
                    help=(
                        "Classifier: "
                        "RandomForestClassifier, "
                        "KNeighborsClassifier, "
                        "CNNLSTMClassifier, "
                        "DINOv2Classifier"
                    )
                )                
                parser.add_argument("-e", "--experimental_setup", type=str, help="The experimental setup file to run (mandatory)")
                parser.add_argument("-r", "--results_directory", type=str, nargs="?", const="results/", help="The directory to compile the results from the experiments (default: results/)")

                args = parser.parse_args()
                
                if not any(vars(args).values()):
                    parser.print_help()
                
                steps = []


                # ============================================================
                # DINOv2
                # ============================================================

                if args.classifier == "DINOv2Classifier":

                    print(
                        "Using pre-generated spectrograms "
                        "with DINOv2."
                    )

                    # Mantém compatibilidade com
                    # save_experiment_results()
                    featextraction_method = (
                        IdentityFeatures()
                    )

                    steps.append(
                        (
                            "feature_extraction",
                            featextraction_method
                        )
                    )


                    model = DINOv2Classifier(
                        model_name=(
                            "facebook/"
                            "dinov2-with-registers-small"
                        ),

                        # Cardoso
                        cv_epochs=30,
                        final_epochs=15,

                        batch_size=32,

                        learning_rate=5e-5,

                        weight_decay=0.01,

                        dropout=0.6,

                        early_stopping_patience=3,

                        scheduler_factor=0.3,

                        image_size=224,

                        device="cuda",

                        num_workers=2,

                        verbose=True
                    )


                    steps.append(
                        (
                            "classifier",
                            model
                        )
                    )


                    pipe = Pipeline(
                        steps
                    )


                    # --------------------------------------------
                    # Substitui loader de sinais pelo loader
                    # dos espectrogramas
                    # --------------------------------------------

                    pipe.train_loader = (
                        spectrogram_vanilla
                    )

                    pipe.validation_loader = (
                        spectrogram_vanilla
                    )

                    pipe.evaluate_loader = (
                        spectrogram_vanilla
                    )


                # ============================================================
                # MÉTODOS CONVENCIONAIS / CNN-LSTM
                # ============================================================

                else:

                    featextraction_method = (
                        WaveletFeatures()
                    )

                    if args.feature_extraction:

                        featextraction_method = (
                            eval(
                                args.feature_extraction
                            )()
                        )


                    print(
                        "Using feature extraction method: "
                        f"{featextraction_method.__class__.__name__}"
                    )


                    steps.append(
                        (
                            "feature_extraction",
                            featextraction_method
                        )
                    )


                    steps.append(
                        (
                            "feature_selection",
                            StableDomainFeatureSelector(
                                k=k_value,
                                score_func="mutual_info",
                                alpha=alpha
                            )
                        )
                    )


                    model = (
                        RandomForestClassifier()
                    )


                    if args.classifier:

                        if (
                            args.classifier
                            == "CNNLSTMClassifier"
                        ):

                            model = (
                                CNNLSTMClassifier()
                            )

                        else:

                            model = (
                                eval(
                                    args.classifier
                                )()
                            )


                    print(
                        "Using classifier: "
                        f"{model.__class__.__name__}"
                    )


                    steps.append(
                        (
                            "classifier",
                            model
                        )
                    )


                    pipe = Pipeline(
                        steps
                    )

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
            