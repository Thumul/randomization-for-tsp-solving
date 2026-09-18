import math
import time
import statistics

from algorithms import (
    nearest_neighbor,
    randomized_nearest_neighbor_top_k,
    randomized_nearest_neighbor_weighted,
    randomized_start_nearest_neighbor,
    held_karp
)
from datasets import load_tsplib_solutions, load_euclidean_instance

# Beyond this size, Held-Karp (O(n^2 * 2^n)) is no longer practical.
EXACT_SOLVABLE_MAX_N = 13

# Best-known tour lengths for every TSPLIB instance in data/tsplib/,
# keyed by instance name (e.g. "berlin52"). Loaded once at import time.
KNOWN_OPTIMAL_LENGTHS = load_tsplib_solutions()

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


def run_deterministic(distance_matrix):
    return measure_algorithm(
        nearest_neighbor,
        distance_matrix,
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


def euclidean_reference(row):
    """
    Reference length for one Euclidean-category manifest row (as produced
    by `datasets.load_euclidean_manifest`): the exact Held-Karp optimum,
    if `n` is small enough to compute it -- otherwise None.

    Unlike TSPLIB, generated Euclidean instances have no externally
    published optimum, so "no reference" is the honest answer once n
    exceeds what Held-Karp can solve; it isn't filled in with a
    single-algorithm's own tour length; a "best observed" reference
    becomes meaningful once randomized variants contribute multiple
    tours per instance.
    """

    if row["n"] > EXACT_SOLVABLE_MAX_N:
        return None

    cities = load_euclidean_instance(row["file"])
    _, length = held_karp(cities)

    return length, "exact"


def tsplib_reference(row):
    """
    Reference length for one TSPLIB manifest row: its published
    best-known/optimal tour length, if this instance has one recorded
    in `data/tsplib/solutions.txt`.
    """

    if row["name"] not in KNOWN_OPTIMAL_LENGTHS:
        return None

    return KNOWN_OPTIMAL_LENGTHS[row["name"]], "tsplib_optimal"


def run_nn_over_manifest(manifest, load_matrix, reference_fn=None, label="", progress_every=50):
    """
    Run deterministic NN once per instance described in `manifest` (a
    list of dicts as produced by the dataset manifest loaders/builders
    in `datasets.py`).

    `load_matrix(row)` loads that row's dense distance matrix -- how to
    do so differs per dataset category (Euclidean coordinates vs. a
    sparse directional edge list), so it's supplied by the caller rather
    than hardcoded here, keeping this loop reusable across categories.

    `reference_fn(row)`, if given, returns (reference_length,
    reference_type) or None when no reference is available for that
    row; used to compute an approximation ratio where one is meaningful.

    A run is marked infeasible when NN was forced to close the tour
    using a pair with no edge in the original sparse graph (an infinite
    entry in the dense matrix) -- this can only happen on the
    distance-matrix category, and is itself one of the things the
    dataset was built to expose.
    """

    records = []

    for i, row in enumerate(manifest):
        distance_matrix = load_matrix(row)
        result = run_deterministic(distance_matrix)
        length = result["length"]
        feasible = math.isfinite(length)

        record = {
            "instance": row["name"],
            "category": row["category"],
            "structure": row["structure"],
            "n": row["n"],
            "algorithm": "deterministic_nn",
            "length": length,
            "time": result["time"],
            "feasible": feasible,
            "reference_length": None,
            "reference_type": None,
            "approx_ratio": None,
        }

        reference = reference_fn(row) if (reference_fn and feasible) else None

        if reference is not None:
            reference_length, reference_type = reference
            record["reference_length"] = reference_length
            record["reference_type"] = reference_type
            record["approx_ratio"] = length / reference_length

        records.append(record)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{label}] {i + 1}/{len(manifest)} done")

    return records


def run_randomized_over_manifest(
    manifest,
    algorithm,
    load_matrix,
    algorithm_name,
    runs=30,
    reference_fn=None,
    label="",
    progress_every=50,
    **algorithm_kwargs,
):
    """
    Run a randomized NN variant `runs` times (independent seeds) per
    instance in `manifest`, mirroring `run_nn_over_manifest` but keeping
    every seeded run instead of collapsing to one result -- needed to
    characterize the distribution of outcomes (mean/std/best/worst), per
    the proposal's R=30-seed methodology, rather than just a single
    value.

    The reference length (if `reference_fn` is given) is computed once
    per instance -- it doesn't depend on the seed -- and reused across
    all of that instance's runs, rather than recomputed `runs` times.
    """

    records = []

    for i, row in enumerate(manifest):
        distance_matrix = load_matrix(row)
        reference = reference_fn(row) if reference_fn else None

        seed_records = run_randomized(
            algorithm,
            distance_matrix,
            runs=runs,
            **algorithm_kwargs,
        )

        for seed_record in seed_records:
            length = seed_record["length"]
            feasible = math.isfinite(length)

            record = {
                "instance": row["name"],
                "category": row["category"],
                "structure": row["structure"],
                "n": row["n"],
                "algorithm": algorithm_name,
                "seed": seed_record["seed"],
                "length": length,
                "time": seed_record["time"],
                "feasible": feasible,
                "reference_length": None,
                "reference_type": None,
                "approx_ratio": None,
            }

            if reference is not None and feasible:
                reference_length, reference_type = reference
                record["reference_length"] = reference_length
                record["reference_type"] = reference_type
                record["approx_ratio"] = length / reference_length

            records.append(record)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{label}] {i + 1}/{len(manifest)} instances done")

    return records
