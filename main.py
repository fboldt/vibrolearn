import argparse

from tests.ptdataloader import check_pytorch_data_pipeline
from experiment.sehri_et_al import (
    run_inspired_cnn_tuning_experiment,
    run_inspired_experiment,
    run_proposed_experiment,
    run_sehri_experiment,
)
from estimator.CNN1D import CNN1D

model = CNN1D()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-pytorch-data",
        action="store_true",
        help="Run a smoke test for CWRU PyTorch Dataset and DataLoader.",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run Sehri, inspired, and proposed experiments.",
    )
    parser.add_argument(
        "--run-cnn-tuning",
        action="store_true",
        help="Run inspired-fold train/validation/test tuning experiment for CNN1D.",
    )
    parser.add_argument(
        "--run-inspired-cnn-tuning-cv3",
        action="store_true",
        help="Alias for --run-cnn-tuning: run the inspired 3-fold CNN tuning experiment.",
    )
    parser.add_argument(
        "--cnn-tuning-verbose",
        action="store_true",
        help="Print progress while running CNN tuning.",
    )
    parser.add_argument(
        "--cnn-tuning-log-every",
        type=int,
        default=1,
        help="Print one progress line every N evaluated parameter combinations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check_pytorch_data:
        check_pytorch_data_pipeline()

    should_run_cnn_tuning = args.run_cnn_tuning or args.run_inspired_cnn_tuning_cv3
    if should_run_cnn_tuning:
        run_inspired_cnn_tuning_experiment(
            verbose=args.cnn_tuning_verbose,
            log_every=max(1, args.cnn_tuning_log_every),
        )

    # Keep the previous default behavior unless a specific check-only run was requested.
    should_run_experiments = (
        (args.run_experiments or not args.check_pytorch_data)
        and not should_run_cnn_tuning
    )
    if should_run_experiments:
        run_sehri_experiment(model)
        run_inspired_experiment(model)
        run_proposed_experiment(model)
