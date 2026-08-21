import json
from pathlib import Path

import numpy as np


def compile_experiment_results(
    input_dir,
    output_filename="compiled_results.json"
):
    """
    Compile multiple experiment JSON files into a single summary JSON.

    Parameters
    ----------
    input_dir : str
        Directory containing experiment JSON files.

    output_filename : str
        Name of the output JSON file.

    Output
    ------
    Creates a JSON file containing:
    - Mean and std of accuracy_score per domain
    - Mean and std of f1_macro per domain
    - Mean load_data_time per domain
    - Mean training_time per domain
    - Mean prediction_time per domain
    - Global mean/std across domains
    """

    input_dir = Path(input_dir)

    json_files = sorted(input_dir.glob("*.json"))

    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in: {input_dir}")

    # Store metrics grouped by domain
    domain_metrics = {}

    for json_file in json_files:

        with open(json_file, "r") as f:
            data = json.load(f)

        scores = data.get("scores", {})

        for domain_name, metrics in scores.items():

            if domain_name not in domain_metrics:
                domain_metrics[domain_name] = {
                    "accuracy_score": [],
                    "f1_macro": [],
                    "load_data_time": [],
                    "training_time": [],
                    "prediction_time": []
                }

            domain_metrics[domain_name]["accuracy_score"].append(
                metrics["accuracy_score"]
            )

            domain_metrics[domain_name]["f1_macro"].append(
                metrics["f1_macro"]
            )

            domain_metrics[domain_name]["load_data_time"].append(
                metrics["load_data_time"]
            )

            domain_metrics[domain_name]["training_time"].append(
                metrics["training_time"]
            )

            domain_metrics[domain_name]["prediction_time"].append(
                metrics["prediction_time"]
            )

    compiled_results = {
        "num_experiments": len(json_files),
        "domains": {}
    }

    # Used later to compute global statistics
    domain_accuracy_means = []
    domain_f1_means = []

    # Compile statistics per domain
    for domain_name in sorted(domain_metrics.keys()):

        acc_values = domain_metrics[domain_name]["accuracy_score"]
        f1_values = domain_metrics[domain_name]["f1_macro"]

        load_values = domain_metrics[domain_name]["load_data_time"]
        train_values = domain_metrics[domain_name]["training_time"]
        pred_values = domain_metrics[domain_name]["prediction_time"]

        acc_mean = float(np.mean(acc_values))
        acc_std = float(np.std(acc_values))

        f1_mean = float(np.mean(f1_values))
        f1_std = float(np.std(f1_values))

        load_mean = float(np.mean(load_values))
        train_mean = float(np.mean(train_values))
        pred_mean = float(np.mean(pred_values))

        domain_accuracy_means.append(acc_mean)
        domain_f1_means.append(f1_mean)

        compiled_results["domains"][domain_name] = {
            "accuracy_score": {
                "mean": acc_mean,
                "std": acc_std
            },
            "f1_macro": {
                "mean": f1_mean,
                "std": f1_std
            },
            "load_data_time": {
                "mean": load_mean
            },
            "training_time": {
                "mean": train_mean
            },
            "prediction_time": {
                "mean": pred_mean
            }
        }

    # Global statistics across domains
    compiled_results["overall"] = {
        "accuracy_score": {
            "mean_across_domains": float(np.mean(domain_accuracy_means)),
            "std_across_domains": float(np.std(domain_accuracy_means))
        },
        "f1_macro": {
            "mean_across_domains": float(np.mean(domain_f1_means)),
            "std_across_domains": float(np.std(domain_f1_means))
        }
    }

    output_path = input_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(compiled_results, f, indent=4)

    print(f"Compiled results saved to: {output_path}")

    return compiled_results

if __name__ == "__main__":
    output_filename = "compiled_results.json"
    # input_directory = f"results_paper/rf_statistical"
    # compile_experiment_results(input_directory, output_filename)
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        input_directory = f"results_paper/rf_statistical_sdfs_k8/alpha_{i}"
        compile_experiment_results(input_directory, output_filename)