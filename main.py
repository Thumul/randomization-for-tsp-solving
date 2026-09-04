import statistics

from datasets import (
    load_tsplib
)

from algorithms import (
    randomized_nearest_neighbor_weighted,
    randomized_start_nearest_neighbor
)

from experiments import (
    run_deterministic,
    run_randomized_top_k,
    measure_algorithm
)


def print_results(name, deterministic, randomized):

    improvement = (deterministic["length"] - randomized["best_length"])
    improvement_percentage = ( improvement / deterministic["length"]) * 100

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Deterministic NN length : {deterministic['length']:.2f}")
    print(f"Deterministic NN time   : {deterministic['time']:.6f} s")

    print()

    print(f"Randomized best length  : {randomized['best_length']:.2f}")
    print(f"Randomized mean length  : {randomized['mean_length']:.2f}")
    print(f"Randomized worst length : {randomized['worst_length']:.2f}")
    print(f"Standard deviation      : {randomized['std_length']:.2f}")
    print(f"Average execution time  : {randomized['mean_time']:.6f} s")
    print(f"Best improvement        : {improvement_percentage:.2f}%")


def run_randomized_trials(algorithm, distance_matrix, runs=500):

    results = [
        measure_algorithm(
            algorithm,
            distance_matrix,
            seed=seed
        )
        for seed in range(runs)
    ]

    lengths = [result["length"] for result in results]
    times = [result["time"] for result in results]
    best_result = min(results, key=lambda result: result["length"])

    return {
        "best_length": min(lengths),
        "worst_length": max(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "std_length": statistics.stdev(lengths),
        "mean_time": statistics.mean(times),
        "best_tour": best_result["tour"]
    }


def print_analysis(deterministic, results):

    best_variant = min(
        results,
        key=lambda item: item[1]["best_length"]
    )
    deterministic_length = deterministic["length"]

    print("\n" + "=" * 60)
    print("Final analysis")
    print("=" * 60)
    print(f"Deterministic baseline       : {deterministic_length:.2f}")

    for name, result in results:
        improvement = (
            (deterministic_length - result["best_length"])
            / deterministic_length
            * 100
        )
        print(
            f"{name:<28}: best {result['best_length']:.2f}, "
            f"mean {result['mean_length']:.2f}, "
            f"improvement {improvement:.2f}%"
        )

    print(
        f"Best randomized variant     : {best_variant[0]} "
        f"({best_variant[1]['best_length']:.2f})"
    )


def main():

    data = load_tsplib("trap20.tsp")
    distance_matrix = data["distance_matrix"]
    runs = 500

    deterministic = run_deterministic(distance_matrix)
    randomized_start = run_randomized_trials(
        randomized_start_nearest_neighbor,
        distance_matrix,
        runs=runs
    )
    randomized_top_k = run_randomized_top_k(
        distance_matrix,
        k=3,
        runs=runs
    )
    randomized_weighted = run_randomized_trials(
        randomized_nearest_neighbor_weighted,
        distance_matrix,
        runs=runs
    )

    print_results(
        "Randomized Starting City NN",
        deterministic,
        randomized_start
    )
    print_results(
        "Top-k Randomized NN",
        deterministic,
        randomized_top_k
    )
    print_results(
        "Distance-weighted Randomized NN",
        deterministic,
        randomized_weighted
    )
    print_analysis(
        deterministic,
        [
            ("Randomized starting city", randomized_start),
            ("Top-k randomized", randomized_top_k),
            ("Distance-weighted randomized", randomized_weighted)
        ]
    )


if __name__ == "__main__":
    main()