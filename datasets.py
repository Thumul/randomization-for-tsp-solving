import os
import math
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
    Load a TSPLIB instance.

    Supported:
    - EDGE_WEIGHT_TYPE: EUC_2D
    - EDGE_WEIGHT_TYPE: EXPLICIT

    Supported explicit formats:
    - FULL_MATRIX
    - LOWER_DIAG_ROW
    - UPPER_DIAG_ROW
    - LOWER_ROW
    - UPPER_ROW

    Returns:
        {
            "name": str,
            "dimension": int,
            "edge_weight_type": str,
            "cities": list | None,
            "distance_matrix": list[list[float]]
        }
    """

    path = "./data/tsp files/" + filename

    headers = {}
    coordinates = []
    edge_weights = []

    section = None

    with open(path, "r") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line == "EOF":
                break

            if line == "NODE_COORD_SECTION":
                section = "NODE_COORD_SECTION"
                continue

            if line == "EDGE_WEIGHT_SECTION":
                section = "EDGE_WEIGHT_SECTION"
                continue

            # Read header fields
            if section is None:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        headers[parts[0]] = parts[1]

                continue

            # Read coordinates
            if section == "NODE_COORD_SECTION":
                parts = line.split()

                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])

                    coordinates.append((x, y))

            # Read explicit edge weights
            elif section == "EDGE_WEIGHT_SECTION":
                edge_weights.extend(
                    float(value)
                    for value in line.split()
                )

    name = headers.get("NAME", filename)
    dimension = int(headers["DIMENSION"])
    edge_weight_type = headers.get("EDGE_WEIGHT_TYPE", "")
    edge_weight_format = headers.get("EDGE_WEIGHT_FORMAT", "")

    # =====================================================
    # EUC_2D
    # =====================================================

    if edge_weight_type == "EUC_2D":

        if len(coordinates) != dimension:
            raise ValueError(
                f"Expected {dimension} coordinates, "
                f"but found {len(coordinates)}"
            )

        distance_matrix = [
            [0] * dimension
            for _ in range(dimension)
        ]

        for i in range(dimension):
            for j in range(i + 1, dimension):

                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]

                # TSPLIB EUC_2D rounding
                distance = int(math.sqrt(dx * dx + dy * dy) + 0.5)

                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        return {
            "name": name,
            "dimension": dimension,
            "edge_weight_type": edge_weight_type,
            "cities": coordinates,
            "distance_matrix": distance_matrix
        }

    # =====================================================
    # EXPLICIT
    # =====================================================

    elif edge_weight_type == "EXPLICIT":

        matrix = [
            [0] * dimension
            for _ in range(dimension)
        ]

        index = 0

        # -------------------------------------------------
        # FULL_MATRIX
        # -------------------------------------------------

        if edge_weight_format == "FULL_MATRIX":

            expected = dimension * dimension

            if len(edge_weights) != expected:
                raise ValueError(
                    f"Expected {expected} edge weights, "
                    f"but found {len(edge_weights)}"
                )

            for i in range(dimension):
                for j in range(dimension):

                    matrix[i][j] = edge_weights[index]
                    index += 1

        # -------------------------------------------------
        # LOWER_DIAG_ROW
        # -------------------------------------------------

        elif edge_weight_format == "LOWER_DIAG_ROW":

            expected = dimension * (dimension + 1) // 2

            if len(edge_weights) != expected:
                raise ValueError(
                    f"Expected {expected} edge weights, "
                    f"but found {len(edge_weights)}"
                )

            for i in range(dimension):
                for j in range(i + 1):

                    value = edge_weights[index]
                    index += 1

                    matrix[i][j] = value
                    matrix[j][i] = value

        # -------------------------------------------------
        # UPPER_DIAG_ROW
        # -------------------------------------------------

        elif edge_weight_format == "UPPER_DIAG_ROW":

            expected = dimension * (dimension + 1) // 2

            if len(edge_weights) != expected:
                raise ValueError(
                    f"Expected {expected} edge weights, "
                    f"but found {len(edge_weights)}"
                )

            for i in range(dimension):
                for j in range(i, dimension):

                    value = edge_weights[index]
                    index += 1

                    matrix[i][j] = value
                    matrix[j][i] = value

        # -------------------------------------------------
        # LOWER_ROW
        # -------------------------------------------------

        elif edge_weight_format == "LOWER_ROW":

            expected = dimension * (dimension - 1) // 2

            if len(edge_weights) != expected:
                raise ValueError(
                    f"Expected {expected} edge weights, "
                    f"but found {len(edge_weights)}"
                )

            for i in range(dimension):
                for j in range(i):

                    value = edge_weights[index]
                    index += 1

                    matrix[i][j] = value
                    matrix[j][i] = value

        # -------------------------------------------------
        # UPPER_ROW
        # -------------------------------------------------

        elif edge_weight_format == "UPPER_ROW":

            expected = dimension * (dimension - 1) // 2

            if len(edge_weights) != expected:
                raise ValueError(
                    f"Expected {expected} edge weights, "
                    f"but found {len(edge_weights)}"
                )

            for i in range(dimension):
                for j in range(i + 1, dimension):

                    value = edge_weights[index]
                    index += 1

                    matrix[i][j] = value
                    matrix[j][i] = value

        else:
            raise NotImplementedError(
                f"Unsupported EDGE_WEIGHT_FORMAT: "
                f"{edge_weight_format}"
            )

        return {
            "name": name,
            "dimension": dimension,
            "edge_weight_type": edge_weight_type,
            "edge_weight_format": edge_weight_format,
            "cities": None,
            "distance_matrix": matrix
        }

    else:
        raise NotImplementedError(
            f"Unsupported EDGE_WEIGHT_TYPE: "
            f"{edge_weight_type}"
        )
