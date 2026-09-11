import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from experiments import run_experiment_suite

RESULTS_DIR = "results"

ALGORITHM_LABELS = {
    "deterministic_nn": "Deterministic NN",
    "top_k": "Top-k Random",
    "weighted": "Distance-Weighted",
    "randomized_start": "Randomized Start"
}


def run_and_save(runs=30, k=5, sizes=None, output_dir=RESULTS_DIR):
    """
    Run the full experiment suite and save the raw per-run results
    as a CSV so plots can be regenerated without rerunning everything.
    """

    os.makedirs(output_dir, exist_ok=True)

    records = run_experiment_suite(runs=runs, k=k, sizes=sizes)
    df = pd.DataFrame(records)

    df.to_csv(os.path.join(output_dir, "experiment_results.csv"), index=False)

    return df


def load_results(output_dir=RESULTS_DIR):
    return pd.read_csv(os.path.join(output_dir, "experiment_results.csv"))


def plot_approx_ratio_vs_n(df, structure="random", output_dir=RESULTS_DIR):
    """
    Mean approximation ratio vs instance size n, one line per algorithm.
    """

    subset = df[df["structure"] == structure]
    grouped = subset.groupby(["algorithm", "n"])["approx_ratio"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm, label in ALGORITHM_LABELS.items():
        algo_data = grouped[grouped["algorithm"] == algorithm].sort_values("n")

        if algo_data.empty:
            continue

        ax.plot(algo_data["n"], algo_data["approx_ratio"], marker="o", label=label)

    ax.set_xlabel("Number of cities (n)")
    ax.set_ylabel("Mean approximation ratio")
    ax.set_title(f"Approximation Ratio vs n ({structure} instances)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"approx_ratio_vs_n_{structure}.png"), dpi=150)
    plt.close(fig)


def plot_runtime_vs_n(df, structure="random", output_dir=RESULTS_DIR):
    """
    Mean runtime vs instance size n, one line per algorithm (log-log).
    """

    subset = df[df["structure"] == structure]
    grouped = subset.groupby(["algorithm", "n"])["time"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm, label in ALGORITHM_LABELS.items():
        algo_data = grouped[grouped["algorithm"] == algorithm].sort_values("n")

        if algo_data.empty:
            continue

        ax.plot(algo_data["n"], algo_data["time"], marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cities (n)")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title(f"Runtime vs n ({structure} instances)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"runtime_vs_n_{structure}.png"), dpi=150)
    plt.close(fig)


def plot_variance_boxplot(df, instance, output_dir=RESULTS_DIR):
    """
    Boxplot of tour length distribution per randomized algorithm,
    for one specific instance, with the deterministic NN length
    drawn as a reference line.
    """

    subset = df[(df["instance"] == instance) & (df["algorithm"] != "deterministic_nn")]

    algorithms = [
        a for a in ALGORITHM_LABELS
        if a != "deterministic_nn" and a in subset["algorithm"].unique()
    ]

    if not algorithms:
        return

    data = [subset[subset["algorithm"] == a]["length"].values for a in algorithms]
    labels = [ALGORITHM_LABELS[a] for a in algorithms]

    det_length = df[
        (df["instance"] == instance) & (df["algorithm"] == "deterministic_nn")
    ]["length"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=labels)

    if not det_length.empty:
        ax.axhline(det_length.iloc[0], color="red", linestyle="--", label="Deterministic NN")
        ax.legend()

    ax.set_ylabel("Tour length")
    ax.set_title(f"Randomized Variance — {instance}")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"variance_boxplot_{instance}.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(df, output_dir=RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    for structure in df["structure"].unique():
        plot_approx_ratio_vs_n(df, structure=structure, output_dir=output_dir)
        plot_runtime_vs_n(df, structure=structure, output_dir=output_dir)

    for instance in df["instance"].unique():
        plot_variance_boxplot(df, instance=instance, output_dir=output_dir)


if __name__ == "__main__":
    results_df = run_and_save()
    generate_all_plots(results_df)
    print(f"Saved {len(results_df)} records and plots to '{RESULTS_DIR}/'")
