import time
import statistics

from algorithms import (
    nearest_neighbor,
    randomized_nearest_neighbor_top_k
)


def measure_algorithm(algorithm, cities, **kwargs):
    """
    Measure execution time and tour length.
    """

    start = time.perf_counter()
    tour, length = algorithm(cities, **kwargs)
    end = time.perf_counter()
    elapsed_time = end - start

    return {
        "tour": tour,
        "length": length,
        "time": elapsed_time
    }


def run_deterministic(cities):
    return measure_algorithm(
        nearest_neighbor,
        cities,
        start_city=0
    )


def run_randomized_top_k(cities, k=3, runs=50):
    lengths = []
    times = []
    best_result = None

    for seed in range(runs):
        result = measure_algorithm(
            randomized_nearest_neighbor_top_k,
            cities,
            k=k,
            seed=seed,
            random_start=False
        )

        lengths.append(result["length"])
        times.append(result["time"])

        if (best_result is None or result["length"] < best_result["length"]):
            best_result = result

    return {
        "best_length": min(lengths),
        "worst_length": max(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "std_length": statistics.stdev(lengths),
        "mean_time": statistics.mean(times),
        "best_tour": best_result["tour"]
    }