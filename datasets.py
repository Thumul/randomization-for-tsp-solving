import csv
import json
import os
import random
import re

import numpy as np

INSTANCE_SIZES = [10, 20, 50, 100, 200, 500, 1000]

TSPLIB_DIR = "data/tsplib"

# TSPLIB instances with more cities than this are excluded from the
# experiment sweep, since each run costs O(n^2) time and memory.
TSPLIB_SWEEP_MAX_N = 5000


def generate_random_euclidean(
    n,
    width=1000,
    height=1000,
    seed=None
):
    """
    Generate cities placed uniformly at random in a rectangle.

    Args:
        n: Number of cities.
        width: Width of the rectangle.
        height: Height of the rectangle.
        seed: Seed for the random number generator.

    Returns:
        A list of n (x, y) coordinate tuples.
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
    Generate cities drawn from Gaussian clusters.

    Cluster centers are placed uniformly at random in the rectangle. Each
    city picks a center uniformly at random and is offset from it by
    independent Gaussian noise in x and y.

    Args:
        n: Number of cities.
        num_clusters: Number of cluster centers.
        width: Width of the rectangle that contains the centers.
        height: Height of the rectangle that contains the centers.
        cluster_std: Standard deviation of the Gaussian offset around each
            center.
        seed: Seed for the random number generator.

    Returns:
        A list of n (x, y) coordinate tuples.
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
    Generate one uniform-random Euclidean instance per size.

    The seed of each instance is base_seed + n, so a given size always
    yields the same coordinates.

    Args:
        sizes: Instance sizes (numbers of cities).
        base_seed: Base value from which each instance seed is derived.

    Returns:
        A dict mapping each size n to its list of (x, y) coordinates.
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
    Generate an instance on which Nearest Neighbor performs poorly.

    Cities form tight, well-separated clusters. Each cluster also has one
    straggler city placed 30-50% of the way toward another cluster's
    center. Nearest Neighbor tends to visit a whole cluster before the
    straggler, which leaves the straggler to be reached later by an
    expensive detour.

    Args:
        seed: Seed for the random number generator.
        num_clusters: Number of clusters.
        cluster_size: Number of cities in each cluster, excluding the
            straggler.
        cluster_std: Standard deviation of the Gaussian offset around each
            cluster center.
        spread: Side length of the square that contains the cluster
            centers.

    Returns:
        A list of num_clusters * (cluster_size + 1) (x, y) tuples.
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

        # Straggler: placed partway toward another randomly chosen cluster
        other_x, other_y = rng.choice(cluster_centers)
        t = rng.uniform(0.3, 0.5)

        straggler_x = center_x + t * (other_x - center_x)
        straggler_y = center_y + t * (other_y - center_y)

        cities.append((straggler_x, straggler_y))

    return cities


def get_tsplib_dimension(filename):
    """
    Read the DIMENSION field from the header of a TSPLIB file.

    Parsing stops at the coordinate section, so the coordinates are not
    read.

    Args:
        filename: Path to a .tsp file.

    Returns:
        The number of cities as an int, or None if the header has no
        DIMENSION field.
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
    Parse the TSPLIB reference file `solutions.txt`.

    Each line has the form "name : length", optionally followed by an
    annotation such as "(CEIL_2D)".

    Args:
        directory: Directory containing `solutions.txt`.

    Returns:
        A dict mapping instance name to its best-known tour length.
    """

    solutions = {}
    path = os.path.join(directory, "solutions.txt")

    with open(path, "r") as file:

        for line in file:

            if ":" not in line:
                continue

            name, value = line.split(":", 1)
            name = name.strip()

            # Keep only the leading number; digits in a trailing
            # annotation are ignored.
            match = re.match(r"\s*(\d+)", value)

            if match:
                solutions[name] = int(match.group(1))

    return solutions


def list_tsplib_instances(directory=TSPLIB_DIR, max_n=None):
    """
    List the TSPLIB instances available in a directory.

    Args:
        directory: Directory containing .tsp files.
        max_n: If given, only instances with at most this many cities
            are listed.

    Returns:
        A sorted list of instance names, without the .tsp extension.
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
    Load the city coordinates of a TSPLIB EUC_2D instance.

    The file is expected to contain:

        NODE_COORD_SECTION
        1 x y
        2 x y
        ...
        EOF

    Args:
        filename: Path to a .tsp file.

    Returns:
        A list of (x, y) coordinate tuples in file order.
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
# Instance sizes are drawn from fixed ranges ("buckets"). Within each
# bucket, n is normally distributed around the bucket's median. Every
# size range is therefore covered, and sizes within a bucket are spread
# rather than repeated.
# =========================================================

# Default bucket scheme, as (low, high) numbers of cities. The generators
# do not read it implicitly; callers pass `buckets` explicitly.
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

# Default upper limit on the number of instances a generator call accepts.
# The generators do not read it implicitly; callers pass `max_total`
# explicitly.
DEFAULT_MAX_TOTAL_INSTANCES = 1200


def sample_bucket_size(low, high, rng, sigma_divisor=6):
    """
    Draw one instance size from a bucket.

    The size is normally distributed around the bucket's median with
    standard deviation (high - low) / sigma_divisor, clamped to
    [low, high] and rounded to an integer.

    Args:
        low: Lower bound of the bucket.
        high: Upper bound of the bucket.
        rng: A `random.Random` instance.
        sigma_divisor: Divisor of the bucket width that gives the
            standard deviation. With 6, about 99.7% of draws fall inside
            the bucket before clamping.

    Returns:
        The sampled number of cities as an int.
    """

    median = (low + high) / 2
    sigma = (high - low) / sigma_divisor

    n = rng.gauss(median, sigma)
    n = max(low, min(high, n))

    return round(n)


def _bucket_instance_counts(total, num_buckets):
    """
    Split `total` instances across `num_buckets` buckets as evenly as
    possible.

    The remainder (total % num_buckets) is distributed one instance each
    to the first buckets.

    Returns:
        A list of per-bucket instance counts.
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
    Generate a dataset of Euclidean TSP instances.

    Instances are split evenly across the size buckets. Each instance is
    either "random" (uniformly placed cities) or "clustered" (cities drawn
    from Gaussian clusters) and is written to `out_dir` as a JSON file
    holding its metadata and coordinates.

    Args:
        total: Number of instances to generate.
        buckets: List of (low, high) size ranges.
        seed: Seed for the dataset. Instance i uses seed + i.
        out_dir: Output directory, created if missing.
        max_total: Upper limit on `total`; a larger value raises
            ValueError.
        structure_ratio: Probability that an instance is "clustered"
            rather than "random".
        num_clusters: Number of clusters in clustered instances.
        cluster_std: Standard deviation of each cluster.
        width: Width of the map.
        height: Height of the map.
        sigma_divisor: Controls the spread of sizes within a bucket; see
            `sample_bucket_size`.

    Returns:
        A manifest: a list of dicts, one per instance, with the keys
        "name", "category", "structure", "bucket_index", "bucket_low",
        "bucket_high", "n", "seed" and "file".
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
    Load the coordinates of one Euclidean instance.

    Args:
        path: Path to an instance JSON file written by
            `generate_euclidean_dataset`.

    Returns:
        A list of (x, y) coordinate tuples.
    """

    with open(path, "r") as f:
        data = json.load(f)

    return [tuple(point) for point in data["coordinates"]]


def load_euclidean_manifest(out_dir):
    """
    Rebuild the manifest of a previously generated Euclidean dataset.

    The manifest is read from the metadata stored in each instance file.
    Coordinates are not loaded; use `load_euclidean_instance` with the
    "file" entry to read them.

    Args:
        out_dir: Directory containing the instance JSON files.

    Returns:
        A list of manifest dicts, in the same format as
        `generate_euclidean_dataset`.
    """

    manifest = []

    for entry in sorted(os.listdir(out_dir)):

        if not entry.endswith(".json"):
            continue

        path = os.path.join(out_dir, entry)

        with open(path, "r") as f:
            data = json.load(f)

        manifest.append({
            key: value for key, value in data.items() if key != "coordinates"
        })
        manifest[-1]["file"] = path

    return manifest


# =========================================================
# DISTANCE-MATRIX DATASET
# Incomplete, directional (possibly asymmetric) reachability graphs --
# e.g. a real road network, where not every pair of cities has a direct
# route and A->B need not cost the same as B->A.
# =========================================================

def generate_sparse_matrix(n, k_fraction, seed, width=1000, height=1000, k_min=2):
    """
    Build a directional k-nearest-neighbor reachability graph.

    Cities are placed uniformly at random. Each city keeps outgoing edges
    only to its k = max(k_min, round(k_fraction * n)) nearest neighbors;
    every other pair is unreachable and has no entry in the edge list.
    The directed edges of a random Hamiltonian cycle are always added, so
    at least one valid tour exists.

    Args:
        n: Number of cities.
        k_fraction: Fraction of n kept as outgoing neighbors per city.
        seed: Seed for the random number generators.
        width: Width of the map.
        height: Height of the map.
        k_min: Minimum number of outgoing neighbors per city.

    Returns:
        An (edges, n) pair, where `edges` is a list of
        (from, to, distance) tuples.
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

    # Add the edges of a random Hamiltonian cycle so that a valid tour
    # always exists
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
    Build a directional reachability graph with a geographic barrier.

    Cities are placed uniformly at random and split into two sides by a
    vertical line through the middle of the map. Within each side, every
    city keeps outgoing edges to its k = max(k_min, round(k_fraction * n))
    nearest neighbors on the same side. The sides are connected only by
    `num_bridges` city pairs, each joined in both directions at its real
    distance; every other cross-side pair is unreachable. The directed
    edges of a random Hamiltonian cycle are always added, so at least one
    valid tour exists.

    Args:
        n: Number of cities.
        k_fraction: Fraction of n kept as same-side outgoing neighbors per
            city.
        seed: Seed for the random number generators.
        num_bridges: Number of city pairs that cross the barrier.
        width: Width of the map.
        height: Height of the map.
        k_min: Minimum number of outgoing neighbors per city.

    Returns:
        An (edges, n) pair, where `edges` is a list of
        (from, to, distance) tuples.
    """

    rng = np.random.default_rng(seed)

    coords = np.column_stack([
        rng.uniform(0, width, size=n),
        rng.uniform(0, height, size=n),
    ])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    side = coords[:, 0] < (width / 2)  # True: left side, False: right side

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

    # Add the edges of a random Hamiltonian cycle so that a valid tour
    # always exists
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
    Build a directional reachability graph with randomly dropped edges.

    Cities are placed uniformly at random. Each ordered pair (i, j) is
    kept independently with probability `keep_probability` and dropped
    otherwise, regardless of distance. In contrast to
    `generate_sparse_matrix` and `generate_barrier_matrix`, which edges
    survive is unrelated to geography. The directed edges of a random
    Hamiltonian cycle are always added, so at least one valid tour exists.

    Args:
        n: Number of cities.
        keep_probability: Probability that a directed edge is kept.
        seed: Seed for the random number generators.
        width: Width of the map.
        height: Height of the map.

    Returns:
        An (edges, n) pair, where `edges` is a list of
        (from, to, distance) tuples.
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

    # Add the edges of a random Hamiltonian cycle so that a valid tour
    # always exists
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
    Generate a dataset of sparse, directional distance-matrix instances.

    Instances are split evenly across the size buckets, with sizes drawn
    as in `generate_euclidean_dataset`. Each instance is built with one
    of three sparsification methods, chosen per instance according to
    `method_weights`:

    - "knn": `generate_sparse_matrix`, each city keeps its k nearest
      neighbors.
    - "barrier": `generate_barrier_matrix`, k-nearest-neighbor edges
      within each side of a geographic split plus `num_bridges` crossings.
    - "random_drop": `generate_random_drop_matrix`, each directed edge is
      kept independently with probability `keep_probability`.

    Each instance is written to `out_dir` as a CSV edge list
    (from,to,distance) and a JSON metadata file with the same name.

    Args:
        total: Number of instances to generate.
        buckets: List of (low, high) size ranges.
        seed: Seed for the dataset. Instance i uses seed + i.
        out_dir: Output directory, created if missing.
        max_total: Upper limit on `total`; a larger value raises
            ValueError.
        method_weights: Dict mapping method name to its relative weight,
            e.g. {"knn": 1, "barrier": 1, "random_drop": 1}. The weights
            do not need to sum to 1.
        k_fraction: Fraction of n kept as outgoing neighbors per city
            ("knn" and "barrier").
        k_min: Minimum number of outgoing neighbors per city.
        num_bridges: Number of barrier crossings ("barrier").
        keep_probability: Probability that a directed edge is kept
            ("random_drop").
        width: Width of the map.
        height: Height of the map.
        sigma_divisor: Controls the spread of sizes within a bucket; see
            `sample_bucket_size`.

    Returns:
        A manifest: a list of dicts, one per instance, with the keys
        "name", "category", "structure", "bucket_index", "bucket_low",
        "bucket_high", "n", "seed", "edge_count" and "file" (the CSV
        path).
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
    Load a sparse distance-matrix instance from its CSV edge list.

    Args:
        csv_path: Path to a CSV file written by
            `generate_distance_matrix_dataset`.

    Returns:
        An (edges, n) pair, where `edges` is a list of
        (from, to, distance) tuples and `n` is the highest city index
        that appears, plus one.
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


def load_distance_matrix_manifest(out_dir):
    """
    Rebuild the manifest of a previously generated distance-matrix
    dataset.

    The manifest is read from the JSON metadata file of each instance.

    Args:
        out_dir: Directory containing the instance CSV and JSON files.

    Returns:
        A list of manifest dicts, in the same format as
        `generate_distance_matrix_dataset`.
    """

    manifest = []

    for entry in sorted(os.listdir(out_dir)):

        if not entry.endswith(".json"):
            continue

        meta_path = os.path.join(out_dir, entry)

        with open(meta_path, "r") as f:
            data = json.load(f)

        data["file"] = meta_path[:-len(".json")] + ".csv"
        manifest.append(data)

    return manifest
