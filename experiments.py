import time
import statistics

from algorithms import (
    nearest_neighbor,
    randomized_nearest_neighbor_top_k,
    randomized_nearest_neighbor_weighted,
    randomized_start_nearest_neighbor,
    held_karp
)

# Beyond this size, Held-Karp (O(n^2 * 2^n)) is no longer practical.
EXACT_SOLVABLE_MAX_N = 13

# Known optimal tour lengths for the TSPLIB benchmark instances used.
KNOWN_OPTIMAL_LENGTHS = {
    "eil51": 426,
    "berlin52": 7542
}

RANDOMIZED_VARIANTS = {
    "top_k": {
        "algorithm": randomized_nearest_neighbor_top_k,
        "kwargs": {"random_start": False}
    },
    "weighted": {
        "algorithm": randomized_nearest_neighbor_weighted,
        "kwargs": {"random_start": False}
    },
    "randomized_start": {
        "algorithm": randomized_start_nearest_neighbor,
        "kwargs": {}
    }
}


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


def run_randomized(algorithm, cities, runs=30, seed_offset=0, **kwargs):
    """
    Run a randomized algorithm `runs` times with independent seeds.

    Returns one record per run (not just summary stats) so the full
    distribution of outcomes is preserved for variance analysis and
    boxplots later.
    """

    records = []

    for i in range(runs):
        seed = seed_offset + i

        result = measure_algorithm(
            algorithm,
            cities,
            seed=seed,
            **kwargs
        )

        records.append({
            "seed": seed,
            "length": result["length"],
            "time": result["time"]
        })

    return records


def summarize(records):
    """
    Reduce a list of {length, time} records to summary statistics.
    """

    lengths = [r["length"] for r in records]
    times = [r["time"] for r in records]

    return {
        "best_length": min(lengths),
        "worst_length": max(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "mean_time": statistics.mean(times)
    }


def run_randomized_top_k(cities, k=3, runs=50):
    """
    Convenience wrapper kept for the existing top-k-only workflow.
    """

    records = run_randomized(
        randomized_nearest_neighbor_top_k,
        cities,
        runs=runs,
        k=k,
        random_start=False
    )

    stats = summarize(records)

    best_seed = min(records, key=lambda r: r["length"])["seed"]
    best_tour, _ = randomized_nearest_neighbor_top_k(
        cities, k=k, seed=best_seed, random_start=False
    )

    stats["best_tour"] = best_tour

    return stats


def reference_length(cities, instance_name, all_lengths):
    """
    Ground-truth / best-available reference length for computing
    approximation ratios.

    Preference order:
    1. Known TSPLIB optimum, if `instance_name` matches a benchmark.
    2. Exact optimum via Held-Karp, if the instance is small enough.
    3. Best tour length observed across every algorithm run on this
       instance (the strongest lower-bound estimate available).
    """

    if instance_name in KNOWN_OPTIMAL_LENGTHS:
        return KNOWN_OPTIMAL_LENGTHS[instance_name], "tsplib_optimal"

    if len(cities) <= EXACT_SOLVABLE_MAX_N:
        _, length = held_karp(cities)
        return length, "exact"

    return min(all_lengths), "best_found"


def run_experiment(cities, instance_name, structure, runs=30, k=5):
    """
    Run deterministic NN and every randomized variant on one instance.

    Returns a flat list of per-run records (one per algorithm run),
    each annotated with the instance's approximation ratio reference.
    """

    all_records = []

    deterministic = run_deterministic(cities)

    all_records.append({
        "instance": instance_name,
        "structure": structure,
        "n": len(cities),
        "algorithm": "deterministic_nn",
        "seed": None,
        "length": deterministic["length"],
        "time": deterministic["time"]
    })

    for variant_name, config in RANDOMIZED_VARIANTS.items():
        kwargs = dict(config["kwargs"])

        if variant_name == "top_k":
            kwargs["k"] = k

        records = run_randomized(
            config["algorithm"],
            cities,
            runs=runs,
            **kwargs
        )

        for record in records:
            all_records.append({
                "instance": instance_name,
                "structure": structure,
                "n": len(cities),
                "algorithm": variant_name,
                "seed": record["seed"],
                "length": record["length"],
                "time": record["time"]
            })

    all_lengths = [r["length"] for r in all_records]
    ref_length, ref_type = reference_length(cities, instance_name, all_lengths)

    for record in all_records:
        record["reference_length"] = ref_length
        record["reference_type"] = ref_type
        record["approx_ratio"] = record["length"] / ref_length

    return all_records


def run_experiment_suite(runs=30, k=5, sizes=None):
    """
    Run the full experiment suite described in the proposal:
    the synthetic random sweep, a matching clustered sweep, one
    adversarial instance, and the TSPLIB benchmark instances.

    Returns a flat list of per-run records suitable for loading
    directly into a pandas DataFrame.
    """

    from datasets import (
        generate_instance_suite,
        generate_clustered,
        generate_adversarial,
        load_tsplib,
        INSTANCE_SIZES
    )

    if sizes is None:
        sizes = INSTANCE_SIZES

    all_records = []

    random_suite = generate_instance_suite(sizes=sizes)

    for n, cities in random_suite.items():
        all_records += run_experiment(
            cities,
            instance_name=f"random_n{n}",
            structure="random",
            runs=runs,
            k=k
        )

    for n in sizes:
        cities = generate_clustered(n=n, seed=2000 + n)

        all_records += run_experiment(
            cities,
            instance_name=f"clustered_n{n}",
            structure="clustered",
            runs=runs,
            k=k
        )

    adversarial_cities = generate_adversarial(
        seed=3000,
        num_clusters=20,
        cluster_size=6,
        cluster_std=8,
        spread=1000
    )

    all_records += run_experiment(
        adversarial_cities,
        instance_name="adversarial",
        structure="adversarial",
        runs=runs,
        k=k
    )

    for name in KNOWN_OPTIMAL_LENGTHS:
        cities = load_tsplib(f"data/{name}.tsp")

        all_records += run_experiment(
            cities,
            instance_name=name,
            structure="tsplib",
            runs=runs,
            k=k
        )

    return all_records
