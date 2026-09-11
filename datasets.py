import csv
import json
import os
import random
import re

import numpy as np

INSTANCE_SIZES = [10, 20, 50, 100, 200, 500, 1000]

TSPLIB_DIR = "data/tsplib"

# Sizes above this are excluded from the automated experiment sweep:
# at O(n^2) per run with R=30 trials, the largest TSPLIB instances
# (up to n=85900) would take hours per instance.
TSPLIB_SWEEP_MAX_N = 5000


def generate_random_euclidean(
    n,
    width=1000,
    height=1000,
    seed=None
):
    """
    Generate n uniformly distributed cities.
    """

    rng = random.Random(seed)
    cities = []

    for _ in range(n):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        cities.append((x, y))

    return cities


def generate_clustered(
    n,
    num_clusters=5,
    width=1000,
    height=1000,
    cluster_std=50,
    seed=None
):
    """
    Generate clustered Euclidean TSP instances.
    """

    rng = random.Random(seed)
    cluster_centers = []

    for _ in range(num_clusters):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        cluster_centers.append((x, y))

    cities = []

    for _ in range(n):
        center_x, center_y = rng.choice(cluster_centers)
        x = rng.gauss(center_x, cluster_std)
        y = rng.gauss(center_y, cluster_std)
        cities.append((x, y))

    return cities


def generate_instance_suite(sizes=INSTANCE_SIZES, base_seed=1000):
    """
    Generate one fixed-seed random Euclidean instance per size in `sizes`.

    The seed is derived deterministically from `base_seed` and n, so
    every algorithm compared later sees the exact same coordinates
    for a given instance size.
    """

    return {
        n: generate_random_euclidean(n, seed=base_seed + n)
        for n in sizes
    }


def generate_adversarial(
    seed=None,
    num_clusters=4,
    cluster_size=15,
    cluster_std=8,
    spread=1000
):
    """
    Structured instance designed to expose Nearest Neighbor's greedy
    weakness.

    Cities form tight clusters spread far apart, and each cluster gets
    one "straggler" placed off to the side, roughly midway toward
    another cluster. NN tends to consume a whole cluster greedily
    before noticing the straggler was left behind, so it ends up far
    from everything else and forces an expensive detour later in the
    tour.
    """

    rng = random.Random(seed)

    cluster_centers = [
        (rng.uniform(0, spread), rng.uniform(0, spread))
        for _ in range(num_clusters)
    ]

    cities = []

    for center_x, center_y in cluster_centers:

        for _ in range(cluster_size):
            x = rng.gauss(center_x, cluster_std)
            y = rng.gauss(center_y, cluster_std)
            cities.append((x, y))

        # Straggler: placed partway toward another random cluster,
        # so it is isolated from its own cluster but not yet close
        # to the next one either.
        other_x, other_y = rng.choice(cluster_centers)
        t = rng.uniform(0.3, 0.5)

        straggler_x = center_x + t * (other_x - center_x)
        straggler_y = center_y + t * (other_y - center_y)

        cities.append((straggler_x, straggler_y))

    return cities


def get_tsplib_dimension(filename):
    """
    Read just the header of a TSPLIB file to get its DIMENSION,
    without parsing the full coordinate list.
    """

    with open(filename, "r") as file:

        for line in file:

            line = line.strip()

            if line.upper().startswith("DIMENSION"):
                return int(line.split(":")[1].strip())

            if line == "NODE_COORD_SECTION":
                break

    return None


def load_tsplib_solutions(directory=TSPLIB_DIR):
    """
    Parse the `solutions` reference file (one "name : length" pair per
    line, some with a trailing annotation) into {name: best_known_length}.
    """

    solutions = {}
    path = os.path.join(directory, "solutions.txt")

    with open(path, "r") as file:

        for line in file:

            if ":" not in line:
                continue

            name, value = line.split(":", 1)
            name = name.strip()

            # Take only the leading number (some lines have a trailing
            # annotation like "(CEIL_2D)" whose digits must be ignored).
            match = re.match(r"\s*(\d+)", value)

            if match:
                solutions[name] = int(match.group(1))

    return solutions


def list_tsplib_instances(directory=TSPLIB_DIR, max_n=None):
    """
    List available TSPLIB instance names (without the .tsp extension)
    in `directory`, optionally filtered to n <= max_n.
    """

    names = []

    for entry in sorted(os.listdir(directory)):

        if not entry.endswith(".tsp"):
            continue

        name = entry[:-4]

        if max_n is not None:
            n = get_tsplib_dimension(os.path.join(directory, entry))

            if n is None or n > max_n:
                continue

        names.append(name)

    return names


def load_tsplib(filename):
    """
    Load a simple TSPLIB EUC_2D instance.

    Expected format contains:

        NODE_COORD_SECTION
        1 x y
        2 x y
        ...
        EOF
    """

    cities = []

    reading_coordinates = False

    with open(filename, "r") as file:

        for line in file:

            line = line.strip()

            if line == "NODE_COORD_SECTION":
                reading_coordinates = True
                continue

            if line == "EOF":
                break

            if reading_coordinates:

                parts = line.split()

                if len(parts) >= 3:

                    x = float(parts[1])
                    y = float(parts[2])

                    cities.append((x, y))

    return cities


# =========================================================
# BUCKETED INSTANCE SIZE SAMPLING
#
# Instance sizes are drawn from 8 fixed ranges ("buckets") spanning
# 3..5000 cities, rather than a small set of exact fixed sizes. Within
# each bucket, n is normally distributed around the bucket's median, so
# every scale is guaranteed coverage (unlike a single normal distribution
# over the whole 3..5000 range, which would starve the small and large
# ends) while still producing a natural, non-repeating spread of sizes.
# =========================================================

# Suggested default bucket scheme -- not used automatically by any generator
# below. Every generator takes `buckets` as a required argument; this is
# just a convenient, documented starting point to pass in explicitly (see
# the notebook's configuration cells).
DEFAULT_SIZE_BUCKETS = [
    (3, 10),
    (10, 20),
    (20, 50),
    (50, 100),
    (100, 200),
    (200, 500),
    (500, 1000),
    (1000, 5000),
]

# Suggested default safety ceiling -- also passed in explicitly (`max_total`)
# rather than read implicitly, so a generation call can never silently
# balloon into an unreasonably large dataset without the caller having
# chosen that ceiling themselves.
DEFAULT_MAX_TOTAL_INSTANCES = 1200


def sample_bucket_size(low, high, rng, sigma_divisor=6):
    """
    Draw one instance size from within [low, high], normally distributed
    around the bucket's median with standard deviation
    (high-low)/sigma_divisor -- sigma_divisor=6 means about 99.7% of draws
    land inside the bucket without needing to clip. Any rare draw outside
    the bucket is clamped back to its edge.
    """

    median = (low + high) / 2
    sigma = (high - low) / sigma_divisor

    n = rng.gauss(median, sigma)
    n = max(low, min(high, n))

    return round(n)


def _bucket_instance_counts(total, num_buckets):
    """
    Split `total` instances as evenly as possible across `num_buckets`
    buckets, handing the remainder (total % num_buckets) to the first
    few buckets one each.
    """

    base = total // num_buckets
    remainder = total % num_buckets

    return [base + 1 if i < remainder else base for i in range(num_buckets)]


# =========================================================
# EUCLIDEAN DATASET
# Complete graphs, straight-line distances, symmetric.
# =========================================================

def generate_euclidean_dataset(
    total,
    buckets,
    seed,
    out_dir,
    max_total,
    structure_ratio=0.5,
    num_clusters=5,
    cluster_std=50,
    width=1000,
    height=1000,
    sigma_divisor=6,
):
    """
    Generate `total` Euclidean TSP instances spread across `buckets`,
    drawing each instance as either uniform-random or clustered spatial
    structure. Each instance is written as one JSON file (coordinates +
    metadata) to `out_dir`.

    `structure_ratio` is the fraction of instances generated as
    "clustered" (cities drawn from `num_clusters` Gaussian blobs of
    spread `cluster_std`, instead of uniformly) rather than "random"
    (uniformly placed) -- e.g. 0.5 means roughly an even mix of both.

    Returns a manifest: a list of dicts, one per generated instance,
    describing its category/structure/size/seed/file path.
    """

    if total > max_total:
        raise ValueError(
            f"requested {total} instances exceeds the safety cap of "
            f"{max_total} -- pass a higher max_total if this is really "
            f"intended"
        )

    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)
    counts = _bucket_instance_counts(total, len(buckets))

    manifest = []
    instance_index = 0

    for bucket_index, ((low, high), count) in enumerate(zip(buckets, counts)):

        for _ in range(count):

            n = sample_bucket_size(low, high, rng, sigma_divisor=sigma_divisor)
            structure = "clustered" if rng.random() < structure_ratio else "random"
            instance_seed = seed + instance_index

            if structure == "random":
                cities = generate_random_euclidean(
                    n, width=width, height=height, seed=instance_seed
                )
            else:
                cities = generate_clustered(
                    n,
                    num_clusters=num_clusters,
                    width=width,
                    height=height,
                    cluster_std=cluster_std,
                    seed=instance_seed,
                )

            name = f"euclidean_b{bucket_index}_{structure}_n{n}_s{instance_seed}"
            path = os.path.join(out_dir, f"{name}.json")

            with open(path, "w") as f:
                json.dump({
                    "name": name,
                    "category": "euclidean",
                    "structure": structure,
                    "bucket_index": bucket_index,
                    "bucket_low": low,
                    "bucket_high": high,
                    "n": n,
                    "seed": instance_seed,
                    "coordinates": cities,
                }, f)

            manifest.append({
                "name": name,
                "category": "euclidean",
                "structure": structure,
                "bucket_index": bucket_index,
                "bucket_low": low,
                "bucket_high": high,
                "n": n,
                "seed": instance_seed,
                "file": path,
            })

            instance_index += 1

    return manifest


def load_euclidean_instance(path):
    """
    Load one Euclidean instance JSON file back into a list of (x, y)
    coordinate tuples.
    """

    with open(path, "r") as f:
        data = json.load(f)

    return [tuple(point) for point in data["coordinates"]]


# =========================================================
# DISTANCE-MATRIX DATASET
# Incomplete, directional (possibly asymmetric) reachability graphs --
# e.g. a real road network, where not every pair of cities has a direct
# route and A->B need not cost the same as B->A.
# =========================================================

def generate_sparse_matrix(n, k_fraction, seed, width=1000, height=1000, k_min=2):
    """
    Build a directional k-nearest-neighbor reachability graph over `n`
    randomly placed cities: each city only has outgoing edges to its
    k = max(k_min, round(k_fraction * n)) nearest neighbors -- every other
    pair is unreachable (no entry in the returned edge list, i.e. an
    implicit float("inf") in the full n x n matrix).

    A random Hamiltonian cycle's directed edges are always force-kept on
    top of the k-NN edges, guaranteeing at least one valid tour exists
    regardless of how the k-NN sparsification falls (verifying Hamiltonian
    -cycle existence directly would be infeasible at n in the thousands).

    Returns (edges, n), where `edges` is a list of (from, to, distance)
    tuples.
    """

    rng = np.random.default_rng(seed)

    coords = np.column_stack([
        rng.uniform(0, width, size=n),
        rng.uniform(0, height, size=n),
    ])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    k = max(k_min, round(k_fraction * n))

    edge_map = {}

    for i in range(n):
        order = np.argsort(dist[i])
        order = order[order != i][:k]

        for j in order:
            edge_map[(i, int(j))] = float(dist[i, j])

    # Guaranteed-feasible backbone: a random Hamiltonian cycle whose
    # directed edges are always kept finite, regardless of k-NN
    # membership, so a valid tour always exists by construction.
    cycle = list(range(n))
    py_rng = random.Random(seed)
    py_rng.shuffle(cycle)

    for idx in range(n):
        a = cycle[idx]
        b = cycle[(idx + 1) % n]
        edge_map[(a, b)] = float(dist[a, b])

    edges = [(a, b, d) for (a, b), d in edge_map.items()]

    return edges, n


def generate_barrier_matrix(n, k_fraction, seed, num_bridges, width=1000, height=1000, k_min=2):
    """
    Build a directional reachability graph with a geographic barrier:
    cities are split into two groups by a vertical line through the
    middle of the map (like a river or mountain range). Within each
    side, every city keeps outgoing edges to its k = max(k_min,
    round(k_fraction * n)) nearest neighbours *on the same side* (same
    sparsification rule as `generate_sparse_matrix`, so storage stays
    comparably bounded). Crossing the barrier is only possible through
    `num_bridges` specific city pairs, each connected bidirectionally at
    their real distance -- every other cross-side pair is unreachable.

    A random Hamiltonian cycle's directed edges are always force-kept on
    top, guaranteeing at least one valid tour exists regardless of how
    the barrier and sparsification fall.

    Returns (edges, n).
    """

    rng = np.random.default_rng(seed)

    coords = np.column_stack([
        rng.uniform(0, width, size=n),
        rng.uniform(0, height, size=n),
    ])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    side = coords[:, 0] < (width / 2)  # True = left side, False = right side

    k = max(k_min, round(k_fraction * n))

    edge_map = {}

    for i in range(n):
        same_side = np.where((side == side[i]) & (np.arange(n) != i))[0]

        if len(same_side) == 0:
            continue

        order = same_side[np.argsort(dist[i, same_side])][:k]

        for j in order:
            edge_map[(i, int(j))] = float(dist[i, j])

    left = np.where(side)[0]
    right = np.where(~side)[0]

    if len(left) > 0 and len(right) > 0:
        py_rng = random.Random(seed)
        num_bridges_actual = min(num_bridges, len(left) * len(right))
        bridge_pairs = set()

        while len(bridge_pairs) < num_bridges_actual:
            a = int(py_rng.choice(left))
            b = int(py_rng.choice(right))
            bridge_pairs.add((a, b))

        for a, b in bridge_pairs:
            edge_map[(a, b)] = float(dist[a, b])
            edge_map[(b, a)] = float(dist[b, a])

    # Guaranteed-feasible backbone, same safety net as generate_sparse_matrix.
    cycle = list(range(n))
    py_rng = random.Random(seed)
    py_rng.shuffle(cycle)

    for idx in range(n):
        a = cycle[idx]
        b = cycle[(idx + 1) % n]
        edge_map[(a, b)] = float(dist[a, b])

    edges = [(a, b, d) for (a, b), d in edge_map.items()]

    return edges, n


def generate_random_drop_matrix(n, keep_probability, seed, width=1000, height=1000):
    """
    Build a directional reachability graph by independently keeping each
    ordered city pair (i, j) with probability `keep_probability` --
    dropped otherwise, regardless of distance. Unlike
    `generate_sparse_matrix` (keep the nearest) or `generate_barrier_matrix`
    (keep the nearest, plus a geographic bottleneck), which edges survive
    here has no relationship to distance at all -- the least physically
    realistic of the three, but a useful contrast: does *structured*
    sparsity (nearest-neighbour, barrier) matter to algorithm performance,
    or does equally-sparse *unstructured* randomness behave the same way?

    A random Hamiltonian cycle's directed edges are always force-kept on
    top, guaranteeing at least one valid tour exists.

    Returns (edges, n).
    """

    rng = np.random.default_rng(seed)

    coords = np.column_stack([
        rng.uniform(0, width, size=n),
        rng.uniform(0, height, size=n),
    ])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    keep_mask = rng.random((n, n)) < keep_probability
    np.fill_diagonal(keep_mask, False)

    edge_map = {}
    rows, cols = np.where(keep_mask)

    for i, j in zip(rows.tolist(), cols.tolist()):
        edge_map[(i, j)] = float(dist[i, j])

    # Guaranteed-feasible backbone, same safety net as the other two.
    cycle = list(range(n))
    py_rng = random.Random(seed)
    py_rng.shuffle(cycle)

    for idx in range(n):
        a = cycle[idx]
        b = cycle[(idx + 1) % n]
        edge_map[(a, b)] = float(dist[a, b])

    edges = [(a, b, d) for (a, b), d in edge_map.items()]

    return edges, n


def generate_distance_matrix_dataset(
    total,
    buckets,
    seed,
    out_dir,
    max_total,
    method_weights,
    k_fraction,
    k_min,
    num_bridges,
    keep_probability,
    width=1000,
    height=1000,
    sigma_divisor=6,
):
    """
    Generate `total` sparse, directional distance-matrix instances spread
    across `buckets` (same bucketed-normal size sampling as the Euclidean
    dataset). Each instance is independently built using one of three
    sparsification methods, chosen per-instance by `method_weights` (a
    dict like {"knn": 1, "barrier": 1, "random_drop": 1}, weights need not
    sum to 1):

    - "knn": `generate_sparse_matrix` -- keep each city's k nearest
      neighbours (k from `k_fraction`/`k_min`).
    - "barrier": `generate_barrier_matrix` -- k-nearest-neighbour within
      each side of a geographic split, plus `num_bridges` crossing points.
    - "random_drop": `generate_random_drop_matrix` -- keep each directed
      edge independently with probability `keep_probability`, unrelated
      to distance.

    Each instance is written as a CSV edge list (from,to,distance) plus a
    JSON metadata sidecar to `out_dir`.

    Returns a manifest: a list of dicts, one per generated instance.
    """

    if total > max_total:
        raise ValueError(
            f"requested {total} instances exceeds the safety cap of "
            f"{max_total} -- pass a higher max_total if this is really "
            f"intended"
        )

    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)
    counts = _bucket_instance_counts(total, len(buckets))

    methods = list(method_weights.keys())
    weights = list(method_weights.values())

    manifest = []
    instance_index = 0

    for bucket_index, ((low, high), count) in enumerate(zip(buckets, counts)):

        for _ in range(count):

            n = sample_bucket_size(low, high, rng, sigma_divisor=sigma_divisor)
            instance_seed = seed + instance_index
            method = rng.choices(methods, weights=weights, k=1)[0]

            if method == "knn":
                edges, n = generate_sparse_matrix(
                    n, k_fraction=k_fraction, seed=instance_seed,
                    width=width, height=height, k_min=k_min,
                )
            elif method == "barrier":
                edges, n = generate_barrier_matrix(
                    n, k_fraction=k_fraction, seed=instance_seed,
                    num_bridges=num_bridges, width=width, height=height, k_min=k_min,
                )
            elif method == "random_drop":
                edges, n = generate_random_drop_matrix(
                    n, keep_probability=keep_probability, seed=instance_seed,
                    width=width, height=height,
                )
            else:
                raise ValueError(f"unknown method {method!r} in method_weights")

            name = f"{method}_b{bucket_index}_n{n}_s{instance_seed}"
            csv_path = os.path.join(out_dir, f"{name}.csv")
            meta_path = os.path.join(out_dir, f"{name}.json")

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["from", "to", "distance"])
                writer.writerows(edges)

            with open(meta_path, "w") as f:
                json.dump({
                    "name": name,
                    "category": "distance_matrix",
                    "structure": method,
                    "bucket_index": bucket_index,
                    "bucket_low": low,
                    "bucket_high": high,
                    "n": n,
                    "seed": instance_seed,
                    "k_fraction": k_fraction,
                    "edge_count": len(edges),
                }, f)

            manifest.append({
                "name": name,
                "category": "distance_matrix",
                "structure": method,
                "bucket_index": bucket_index,
                "bucket_low": low,
                "bucket_high": high,
                "n": n,
                "seed": instance_seed,
                "edge_count": len(edges),
                "file": csv_path,
            })

            instance_index += 1

    return manifest


def load_sparse_matrix(csv_path):
    """
    Load a sparse distance-matrix instance saved by
    `generate_distance_matrix_dataset` back into an (edges, n) pair,
    where `n` is inferred as the highest city index seen, plus one.
    """

    edges = []
    max_index = -1

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header row

        for row in reader:
            a, b, d = int(row[0]), int(row[1]), float(row[2])
            edges.append((a, b, d))
            max_index = max(max_index, a, b)

    return edges, max_index + 1
