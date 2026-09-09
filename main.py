import statistics

from datasets import load_tsplib

from algorithms import (
    randomized_nearest_neighbor_weighted,
    randomized_start_nearest_neighbor
)

from experiments import (
    run_deterministic,
    run_randomized_top_k,
    measure_algorithm
)



# Configuration
NUMBER_OF_TSP_FILES = 15
RUNS = 500
TOP_K = 3
FILE_PREFIX = "graph-"



def run_randomized_trials(algorithm, distance_matrix, runs=500):
    """
    Run a randomized algorithm multiple times and collect
    statistical information.
    """

    results = [
        measure_algorithm(algorithm, distance_matrix, seed=seed) for seed in range(runs)
    ]

    lengths = [result["length"] for result in results]
    times = [result["time"] for result in results]
    best_result = min(results, key=lambda result: result["length"])

    return {
        "best_length": min(lengths),
        "worst_length": max(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "std_length": (
            statistics.stdev(lengths)
            if len(lengths) > 1
            else 0
        ),
        "mean_time": statistics.mean(times),
        "best_tour": best_result["tour"]
    }



# Improvement calculation
def calculate_improvement(deterministic_length, randomized_length):
    """
    Calculate percentage improvement of randomized solution
    compared with deterministic NN.
    """

    return (
        (deterministic_length - randomized_length) / deterministic_length
    ) * 100



# Run one TSP instance
def run_graph(filename):
    """
    Run all algorithms on one TSP file.
    """

    data = load_tsplib(filename)
    distance_matrix = data["distance_matrix"]

    # Deterministic NN
    deterministic = run_deterministic(distance_matrix)

    # Random starting city
    randomized_start = run_randomized_trials(randomized_start_nearest_neighbor, distance_matrix, runs=RUNS)

    # Top-k randomized NN
    randomized_top_k = run_randomized_top_k(distance_matrix, k=TOP_K, runs=RUNS)

    # Distance weighted randomized NN
    randomized_weighted = run_randomized_trials(randomized_nearest_neighbor_weighted, distance_matrix, runs=RUNS)

    deterministic_length = deterministic["length"]

    return {
        "filename": filename,
        "dimension": len(distance_matrix),
        "deterministic_length": deterministic_length,
        "deterministic_time": deterministic["time"],

        # Random start
        "start_best": randomized_start["best_length"],
        "start_mean": randomized_start["mean_length"],
        "start_std": randomized_start["std_length"],
        "start_improvement": calculate_improvement(deterministic_length, randomized_start["best_length"]),

        # Top-k
        "topk_best": randomized_top_k["best_length"],
        "topk_mean": randomized_top_k["mean_length"],
        "topk_std": randomized_top_k["std_length"],
        "topk_improvement": calculate_improvement(deterministic_length, randomized_top_k["best_length"]),

        # Weighted
        "weighted_best": randomized_weighted["best_length"],
        "weighted_mean": randomized_weighted["mean_length"],
        "weighted_std": randomized_weighted["std_length"],
        "weighted_improvement": calculate_improvement(deterministic_length, randomized_weighted["best_length"])
    }



# Print main results table
def print_results_table(results):

    print("\n")
    print("=" * 138)

    print(
        f"{'Graph':<14}"
        f"{'N':>5}"
        f"{'NN':>10}"
        f"{'RandStart':>12}"
        f"{'Imp %':>9}"
        f"{'Top-k':>12}"
        f"{'Imp %':>9}"
        f"{'Weighted':>12}"
        f"{'Imp %':>9}"
    )

    print("=" * 138)

    for result in results:

        print(
            f"{result['filename']:<14}"
            f"{result['dimension']:>5}"
            f"{result['deterministic_length']:>10.2f}"

            f"{result['start_best']:>12.2f}"
            f"{result['start_improvement']:>9.2f}"

            f"{result['topk_best']:>12.2f}"
            f"{result['topk_improvement']:>9.2f}"

            f"{result['weighted_best']:>12.2f}"
            f"{result['weighted_improvement']:>9.2f}"
        )

    print("=" * 138)



# Print mean results table
def print_mean_table(results):

    print("\n")
    print("Mean randomized tour lengths")
    print("=" * 85)

    print(
        f"{'Graph':<14}"
        f"{'NN':>12}"
        f"{'RandStart Mean':>18}"
        f"{'Top-k Mean':>18}"
        f"{'Weighted Mean':>18}"
    )

    print("=" * 85)

    for result in results:

        print(
            f"{result['filename']:<14}"
            f"{result['deterministic_length']:>12.2f}"
            f"{result['start_mean']:>18.2f}"
            f"{result['topk_mean']:>18.2f}"
            f"{result['weighted_mean']:>18.2f}"
        )

    print("=" * 85)



# Overall analysis
def print_overall_analysis(results):

    print("\n")
    print("=" * 70)
    print("OVERALL ANALYSIS")
    print("=" * 70)

    number_of_graphs = len(results)


    # Count wins
    start_wins = sum(
        1
        for result in results
        if result["start_best"]
        < result["deterministic_length"]
    )

    topk_wins = sum(
        1
        for result in results
        if result["topk_best"]
        < result["deterministic_length"]
    )

    weighted_wins = sum(
        1
        for result in results
        if result["weighted_best"]
        < result["deterministic_length"]
    )


    # Average improvements
    average_start_improvement = statistics.mean(
        result["start_improvement"]
        for result in results
    )

    average_topk_improvement = statistics.mean(
        result["topk_improvement"]
        for result in results
    )

    average_weighted_improvement = statistics.mean(
        result["weighted_improvement"]
        for result in results
    )

    print(
        f"Number of TSP instances : "
        f"{number_of_graphs}"
    )

    print(
        f"Random-start wins       : "
        f"{start_wins}/{number_of_graphs}"
    )

    print(
        f"Top-k wins              : "
        f"{topk_wins}/{number_of_graphs}"
    )

    print(
        f"Weighted wins           : "
        f"{weighted_wins}/{number_of_graphs}"
    )

    print()

    print(
        f"Average Random-start improvement : "
        f"{average_start_improvement:.2f}%"
    )

    print(
        f"Average Top-k improvement        : "
        f"{average_topk_improvement:.2f}%"
    )

    print(
        f"Average Weighted improvement     : "
        f"{average_weighted_improvement:.2f}%"
    )

    # Best performing method overall
    variants = [
        (
            "Randomized starting city",
            average_start_improvement
        ),
        (
            "Top-k randomized",
            average_topk_improvement
        ),
        (
            "Distance-weighted randomized",
            average_weighted_improvement
        )
    ]

    best_variant = max(
        variants,
        key=lambda item: item[1]
    )

    print()

    print(
        f"Best randomized variant overall: "
        f"{best_variant[0]}"
    )

    print(
        f"Average improvement             : "
        f"{best_variant[1]:.2f}%"
    )



def main():

    all_results = []

    for graph_number in range(1, NUMBER_OF_TSP_FILES + 1):
        filename = (
            f"{FILE_PREFIX}"
            f"{graph_number:03d}.tsp"
        )
        print(f"Processing {filename}...")

        try:
            result = run_graph(filename)
            all_results.append(result)

        except FileNotFoundError:
            print(f"WARNING: {filename} was not found. Skipping.")

        except Exception as error:
            print(f"ERROR while processing {filename}: {error}")

    if not all_results:
        print("\nNo TSP files were successfully processed.")
        return

    print_results_table(all_results)
    print_mean_table(all_results)
    print_overall_analysis(all_results)


if __name__ == "__main__":
    main()