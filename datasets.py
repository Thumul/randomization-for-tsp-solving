import os
import random
import re

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
