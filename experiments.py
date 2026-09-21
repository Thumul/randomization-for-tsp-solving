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

# Largest instance size for which the exact Held-Karp optimum is computed
EXACT_SOLVABLE_MAX_N = 13

# Best-known tour lengths for the TSPLIB instances in data/tsplib/, keyed by
# instance name (e.g. "berlin52")
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
    Run an algorithm once and record its wall-clock time.

    Args:
        algorithm: Callable taking a distance matrix (or cities) and
            returning a (tour, length) pair.
        cities: Input passed as the first argument to `algorithm`.
        **kwargs: Additional keyword arguments forwarded to `algorithm`.

    Returns:
        A dict with the keys "tour", "length" and "time" (seconds).
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
    """
    Run deterministic Nearest Neighbor from city 0.

    Args:
        distance_matrix: n x n matrix of pairwise distances.

    Returns:
        A dict with the keys "tour", "length" and "time" (seconds).
    """

    return measure_algorithm(
        nearest_neighbor,
        distance_matrix,
        start_city=0
    )


def run_randomized(algorithm, cities, runs=30, seed_offset=0, **kwargs):
    """
    Run a randomized algorithm repeatedly with consecutive seeds.

    Args:
        algorithm: Randomized algorithm accepting a `seed` keyword.
        cities: Input passed as the first argument to `algorithm`.
        runs: Number of independent runs.
        seed_offset: Seed of the first run; run i uses seed_offset + i.
        **kwargs: Additional keyword arguments forwarded to `algorithm`.

    Returns:
        A list with one dict per run, holding "seed", "length" and
        "time" (seconds).
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
    Summarize a list of run records.

    Args:
        records: List of dicts with "length" and "time" keys, as
            returned by `run_randomized`.

    Returns:
        A dict with best, worst, mean, median and standard deviation of
        the tour length (standard deviation is 0.0 for a single record),
        and the mean run time.
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
    Run top-k randomized Nearest Neighbor repeatedly and summarize it.

    The starting city is fixed at city 0, so the variation between runs
    comes only from the randomized candidate choice.

    Args:
        cities: n x n matrix of pairwise distances.
        k: Size of the candidate pool.
        runs: Number of independent runs.

    Returns:
        The dict produced by `summarize`, extended with "best_tour", the
        tour of the shortest run.
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
    Reference tour length for a generated Euclidean instance.

    The reference is the exact Held-Karp optimum, computed only when
    `n <= EXACT_SOLVABLE_MAX_N`. Generated instances have no published
    optimum, so larger instances have no reference.

    Args:
        row: Manifest row, as produced by `datasets.load_euclidean_manifest`.

    Returns:
        A (reference_length, "exact") pair, or None if the instance is
        too large for the exact solver.
    """

    if row["n"] > EXACT_SOLVABLE_MAX_N:
        return None

    cities = load_euclidean_instance(row["file"])
    _, length = held_karp(cities)

    return length, "exact"


def tsplib_reference(row):
    """
    Reference tour length for a TSPLIB instance.

    The reference is the published best-known or optimal length recorded
    in `data/tsplib/solutions.txt`.

    Args:
        row: TSPLIB manifest row with a "name" key.

    Returns:
        A (reference_length, "tsplib_optimal") pair, or None if the
        instance has no recorded length.
    """

    if row["name"] not in KNOWN_OPTIMAL_LENGTHS:
        return None

    return KNOWN_OPTIMAL_LENGTHS[row["name"]], "tsplib_optimal"


def run_nn_over_manifest(manifest, load_matrix, reference_fn=None, label="", progress_every=50):
    """
    Run deterministic Nearest Neighbor once on every instance in a
    manifest.

    A run is marked infeasible when its tour uses a pair with no edge in
    the original sparse graph, i.e. when its length is infinite. This can
    only occur in the distance-matrix category.

    Args:
        manifest: List of instance descriptors, as produced by the
            manifest loaders in `datasets.py`. Each row needs the keys
            "name", "category", "structure" and "n".
        load_matrix: Callable taking a manifest row and returning its
            dense n x n distance matrix.
        reference_fn: Optional callable taking a manifest row and
            returning a (reference_length, reference_type) pair, or None
            when the row has no reference. It is used to compute the
            approximation ratio of feasible runs.
        label: Name shown in progress messages.
        progress_every: Print progress after every this many instances;
            0 or None disables progress output.

    Returns:
        A list with one record per instance, holding the instance
        metadata, "algorithm", "length", "time", "feasible",
        "reference_length", "reference_type" and "approx_ratio".
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
    Run a randomized Nearest Neighbor variant repeatedly on every
    instance in a manifest.

    Every seeded run is kept as its own record, so the distribution of
    outcomes (mean, standard deviation, best, worst) can be analyzed. The
    reference length does not depend on the seed and is computed once per
    instance.

    Args:
        manifest: List of instance descriptors, as produced by the
            manifest loaders in `datasets.py`. Each row needs the keys
            "name", "category", "structure" and "n".
        algorithm: Randomized algorithm accepting a `seed` keyword.
        load_matrix: Callable taking a manifest row and returning its
            dense n x n distance matrix.
        algorithm_name: Label stored in the "algorithm" field of every
            record.
        runs: Number of independent seeds per instance.
        reference_fn: Optional callable taking a manifest row and
            returning a (reference_length, reference_type) pair, or None
            when the row has no reference.
        label: Name shown in progress messages.
        progress_every: Print progress after every this many instances;
            0 or None disables progress output.
        **algorithm_kwargs: Additional keyword arguments forwarded to
            `algorithm` (for example `k` or `random_start`).

    Returns:
        A list with one record per instance and seed, holding the
        instance metadata, "algorithm", "seed", "length", "time",
        "feasible", "reference_length", "reference_type" and
        "approx_ratio".
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
