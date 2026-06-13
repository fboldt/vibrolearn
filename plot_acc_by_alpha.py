import matplotlib.pyplot as plt
import numpy as np
import os


def plot_alpha_sensitivity(results, save_path=None):
    alphas = np.array([r["alpha"] for r in results])

    accuracies = np.array([
        r["accuracy_score"]["mean_across_domains"] * 100
        for r in results
    ])

    best_idx = np.argmax(accuracies)

    # Good size for SBC paper column/half-page figures
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    ax.plot(
        alphas,
        accuracies,
        color="black",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="Mean accuracy"
    )

    for i, (x, y) in enumerate(zip(alphas, accuracies)):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            fontweight="bold" if i == best_idx else "normal"
        )

    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=10)

    ax.set_xticks(alphas)
    ax.tick_params(axis="both", labelsize=8)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="0.75"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=8,
        frameon=False,
        loc="lower right"
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight"
        )

    plt.show()




if __name__ == "__main__":
    for dirname in os.listdir("results_paper/rf_wpd_sdfs_alpha"):
        print(dirname)
        if dirname.startswith("compiled_results_alpha_"):
            with open(os.path.join("results_paper/rf_wpd_sdfs_alpha", dirname), "r") as f:
                data = json.load(f)
                print(data)
                # results.append({
                #     "alpha": int(dirname.split("_")[3]),
                #     "accuracy_score": data["overall"]["accuracy_score"],
                #     "f1_macro": data["overall"]["f1_macro"]
                # })
    
    # plot_alpha_sensitivity(
    #     results,
    #     save_path="alpha_sensitivity_validation.png"
    # )
