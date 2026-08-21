import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from dataset.loader import spectrogram_vanilla
from estimators.cnn_lstm import CNNLSTMClassifier
from estimators.dinov2 import DINOv2Classifier
from estimators.pipeline import Pipeline

from experiment.assesment import run_experiment, save_scores
from experiment.compile_results import (
    compile_results,
    compile_results_across_folds_and_domains,
    generate_paired_augmentation_boxplots,
)

from feature.extraction import (
    FlattenFeatures,
    StatisticalFeatures,
    HeterogeneousFeatures,
    WaveletFeatures,
    RMSFeatures,
)

from feature.feature_selector import StableDomainFeatureSelector


# ============================================================
# HIPERPARÂMETROS DO MÉTODO WPD + SCED + RF
# ============================================================

ALPHA_VALUES = [0.7]
K_VALUES = [8]


# ============================================================
# REGISTROS DE CLASSES
# Evita o uso de eval()
# ============================================================

FEATURE_EXTRACTORS = {
    "FlattenFeatures": FlattenFeatures,
    "StatisticalFeatures": StatisticalFeatures,
    "HeterogeneousFeatures": HeterogeneousFeatures,
    "WaveletFeatures": WaveletFeatures,
    "RMSFeatures": RMSFeatures,
}


CLASSIFIERS = {
    "RandomForestClassifier": RandomForestClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "CNNLSTMClassifier": CNNLSTMClassifier,
    "DINOv2Classifier": DINOv2Classifier,
}


# ============================================================
# IDENTITY FEATURES
# Utilizado pelo DINOv2 porque os espectrogramas já foram
# previamente gerados.
# ============================================================

class IdentityFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=object)


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run VibroLearn experiments."
    )

    parser.add_argument(
        "-f",
        "--feature_extraction",
        type=str,
        choices=FEATURE_EXTRACTORS.keys(),
        help="Feature extraction method."
    )

    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        choices=CLASSIFIERS.keys(),
        help="Classifier to use."
    )

    parser.add_argument(
        "-e",
        "--experimental_setup",
        type=str,
        help="Experimental setup JSON file."
    )

    parser.add_argument(
        "-r",
        "--results_directory",
        type=str,
        nargs="?",
        const="results/",
        help="Directory containing experiment results to compile."
    )

    return parser.parse_args()


# ============================================================
# CARREGAMENTO DA CONFIGURAÇÃO
# ============================================================

def load_experimental_setup(config_path):

    with open(config_path, "r", encoding="utf-8") as file:
        experimental_setup = json.load(file)

    return experimental_setup


# ============================================================
# PIPELINE DINOv2
# ============================================================

def build_dinov2_pipeline():

    print("Using pre-generated spectrograms with DINOv2.")

    steps = [
        (
            "feature_extraction",
            IdentityFeatures()
        ),
        (
            "classifier",
            DINOv2Classifier(
                model_name="facebook/dinov2-with-registers-small",

                # Configuração utilizada no trabalho
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
        )
    ]

    pipe = Pipeline(steps)

    # DINOv2 utiliza os espectrogramas previamente gerados.
    pipe.train_loader = spectrogram_vanilla
    pipe.validation_loader = spectrogram_vanilla
    pipe.evaluate_loader = spectrogram_vanilla

    return pipe


# ============================================================
# PIPELINE CONVENCIONAL
# ============================================================

def build_conventional_pipeline(
    classifier_name,
    feature_extraction_name,
    k_value,
    alpha
):

    # --------------------------------------------------------
    # Extração de características
    # --------------------------------------------------------

    if feature_extraction_name:
        feature_extractor = (
            FEATURE_EXTRACTORS[feature_extraction_name]()
        )
    else:
        feature_extractor = WaveletFeatures()

    print(
        "Using feature extraction method: "
        f"{feature_extractor.__class__.__name__}"
    )

    # --------------------------------------------------------
    # Seleção de características
    # --------------------------------------------------------

    feature_selector = StableDomainFeatureSelector(
        k=k_value,
        alpha=alpha
    )

    # --------------------------------------------------------
    # Classificador
    # --------------------------------------------------------

    if classifier_name:
        classifier = CLASSIFIERS[classifier_name]()
    else:
        classifier = RandomForestClassifier()

    print(
        "Using classifier: "
        f"{classifier.__class__.__name__}"
    )

    steps = [
        (
            "feature_extraction",
            feature_extractor
        ),
        (
            "feature_selection",
            feature_selector
        ),
        (
            "classifier",
            classifier
        )
    ]

    return Pipeline(steps)


# ============================================================
# CONSTRUÇÃO DO PIPELINE
# ============================================================

def build_pipeline(
    args,
    k_value=None,
    alpha=None
):

    if args.classifier == "DINOv2Classifier":
        return build_dinov2_pipeline()

    return build_conventional_pipeline(
        classifier_name=args.classifier,
        feature_extraction_name=args.feature_extraction,
        k_value=k_value,
        alpha=alpha
    )


# ============================================================
# EXECUÇÃO E SALVAMENTO
# ============================================================

def run_and_save_experiment(
    pipeline,
    experimental_setup,
    experimental_setup_path,
    repetition,
    k_value=None,
    alpha=None
):

    experiment_name = experimental_setup.get(
        "experiment_name",
        Path(experimental_setup_path).stem
    )

    output_dir = Path(
        experimental_setup.get(
            "output_dir",
            "results"
        )
    )

    results = {
        "experiment_name": experiment_name,
        "feature_extraction": (
            pipeline.pipe
            .named_steps["feature_extraction"]
            .__class__.__name__
        ),
        "classifier": (
            pipeline.pipe
            .named_steps["classifier"]
            .__class__.__name__
        ),
        "repetition": repetition,
        "start_time": datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    }

    # Registra também os hiperparâmetros quando existirem.
    if k_value is not None:
        results["k"] = k_value

    if alpha is not None:
        results["alpha"] = alpha

    # --------------------------------------------------------
    # Executa o protocolo experimental
    # --------------------------------------------------------

    results["scores"] = run_experiment(
        pipeline,
        experimental_setup
    )

    results["end_time"] = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # --------------------------------------------------------
    # Nome do arquivo
    # --------------------------------------------------------

    output_file = output_dir / (
        f"run-{repetition:02d}_"
        f"{results['end_time']}.json"
    )

    save_scores(
        results,
        str(output_file)
    )


# ============================================================
# EXECUÇÃO DOS EXPERIMENTOS
# ============================================================

def execute_experiments(
    args,
    experimental_setup
):

    repetitions = experimental_setup.get(
        "repetitions",
        1
    )

    if repetitions < 1:
        raise ValueError(
            "'repetitions' must be greater than or equal to 1."
        )

    print(
        f"\nExperiment: "
        f"{experimental_setup.get('experiment_name', 'unnamed')}"
    )

    print(
        f"Repetitions: {repetitions}"
    )

    print(
        "Output directory: "
        f"{experimental_setup.get('output_dir', 'results')}\n"
    )

    # ========================================================
    # DINOv2
    #
    # Não precisa percorrer k e alpha porque esses parâmetros
    # pertencem à SCED.
    # ========================================================

    if args.classifier == "DINOv2Classifier":

        for repetition in range(
            1,
            repetitions + 1
        ):

            print(
                f"\n{'=' * 60}\n"
                f"Repetition {repetition}/{repetitions}\n"
                f"{'=' * 60}"
            )

            pipeline = build_pipeline(args)

            run_and_save_experiment(
                pipeline=pipeline,
                experimental_setup=experimental_setup,
                experimental_setup_path=args.experimental_setup,
                repetition=repetition
            )

        return

    # ========================================================
    # MÉTODOS COM SCED
    # ========================================================

    for k_value in K_VALUES:

        for alpha in ALPHA_VALUES:

            print(
                f"\nConfiguration: "
                f"k={k_value}, alpha={alpha}"
            )

            for repetition in range(
                1,
                repetitions + 1
            ):

                print(
                    f"\n{'=' * 60}\n"
                    f"Repetition {repetition}/{repetitions}\n"
                    f"k={k_value} | alpha={alpha}\n"
                    f"{'=' * 60}"
                )

                pipeline = build_pipeline(
                    args,
                    k_value=k_value,
                    alpha=alpha
                )

                run_and_save_experiment(
                    pipeline=pipeline,
                    experimental_setup=experimental_setup,
                    experimental_setup_path=args.experimental_setup,
                    repetition=repetition,
                    k_value=k_value,
                    alpha=alpha
                )


# ============================================================
# COMPILAÇÃO DOS RESULTADOS
# ============================================================

def compile_experiment_results(
    results_directory
):

    print(
        f"Compiling results from directory: "
        f"{results_directory}"
    )

    output_file = compile_results(
        results_directory
    )

    print(
        f"Compiled results saved to: "
        f"{output_file}"
    )

    output_file = (
        compile_results_across_folds_and_domains(
            results_directory
        )
    )

    print(
        "Compiled results across folds and domains "
        f"saved to: {output_file}"
    )

    output_files = (
        generate_paired_augmentation_boxplots(
            results_directory
        )
    )

    print(
        "Generated paired augmentation box plots "
        f"saved to: {output_files}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if (
        not args.experimental_setup
        and not args.results_directory
    ):
        print(
            "No experimental setup or results directory "
            "was specified."
        )
        return

    # --------------------------------------------------------
    # Executa experimento
    # --------------------------------------------------------

    if args.experimental_setup:

        print(
            f"Loading experimental setup: "
            f"{args.experimental_setup}"
        )

        experimental_setup = (
            load_experimental_setup(
                args.experimental_setup
            )
        )

        execute_experiments(
            args,
            experimental_setup
        )

    # --------------------------------------------------------
    # Compila resultados
    # --------------------------------------------------------

    if args.results_directory:

        compile_experiment_results(
            args.results_directory
        )


if __name__ == "__main__":
    main()