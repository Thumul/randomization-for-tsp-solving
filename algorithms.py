import itertools
import math
import random

import numpy as np


def euclidean_distance(city1, city2):
    """
    Calculate Euclidean distance between two cities

    city = (x, y)
    """
    return math.sqrt(
        (city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2
    )


def create_distance_matrix(cities):
    """
    Precompute distances between every pair of cities as a dense,
    symmetric n x n matrix.

    Vectorized with numpy: a pure-Python nested loop is O(n^2) Python-level
    iterations, which becomes the bottleneck at n in the thousands (the
    dataset's largest bucket goes up to n=5000).
    """
    coords = np.asarray(cities, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]

    return np.sqrt((diff ** 2).sum(axis=-1))


def build_dense_matrix_from_edges(edges, n, missing_value=float("inf")):
    """
    Expand a sparse, possibly directional (from, to, distance) edge list
    -- as produced for the distance-matrix dataset category -- into a
    dense n x n matrix, so it can be fed to the same NN/RNN implementations
    used for Euclidean instances.

    Every pair with no edge gets `missing_value` (default: infinity, i.e.
    "no route exists"). The diagonal is always 0. Since the source graph
    can be directional, matrix[i][j] and matrix[j][i] are populated
    independently and need not be equal.
    """
    matrix = np.full((n, n), missing_value, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    for a, b, distance in edges:
        matrix[a][b] = distance

    return matrix


def calculate_tour_length(tour, distance_matrix):
    """
    Calculate total tour distance.
    """
    total_distance = 0.0

    for i in range(len(tour)):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]

        total_distance += distance_matrix[current_city][next_city]

    return total_distance


# =========================================================
# EXACT SOLVER
# Held-Karp Dynamic Programming
# =========================================================

def held_karp(cities):
    """
    Exact TSP solver using Held-Karp dynamic programming.

    Runs in O(n^2 * 2^n) time, practical for n up to ~15.
    Tour always starts at city 0.
    """

    n = len(cities)

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    distance_matrix = create_distance_matrix(cities)

    # cost[(subset, last)] = (min cost to start at 0, visit exactly
    # `subset`, and end at `last`; predecessor of `last` on that path)
    cost = {}

    for k in range(1, n):
        cost[(frozenset([k]), k)] = (distance_matrix[0][k], 0)

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            subset = frozenset(subset)

            for k in subset:
                prev_subset = subset - {k}

                best_cost, best_prev = min(
                    (
                        cost[(prev_subset, m)][0] + distance_matrix[m][k],
                        m
                    )
                    for m in prev_subset
                )

                cost[(subset, k)] = (best_cost, best_prev)

    full_set = frozenset(range(1, n))

    best_cost, best_last = min(
        (cost[(full_set, k)][0] + distance_matrix[k][0], k)
        for k in range(1, n)
    )

    # Reconstruct the tour by walking predecessors backward
    tour = []
    subset = full_set
    last = best_last

    while last != 0:
        tour.append(last)
        _, prev = cost[(subset, last)]
        subset = subset - {last}
        last = prev

    tour.append(0)
    tour.reverse()

    return tour, best_cost


# =========================================================
# VERSION 1
# Deterministic Nearest Neighbor
# =========================================================

def nearest_neighbor(distance_matrix, start_city=0):
    """
    Deterministic Nearest Neighbor TSP heuristic.
    At each step, select the closest unvisited city.

    On a complete graph this always finds one. On a sparse/directional
    graph (missing pairs represented as float("inf")), the current city
    can have no reachable unvisited city left -- in that case the tour
    is forced to continue to the first unvisited city in index order so
    it still completes as a valid permutation. The resulting tour length
    then includes an infinite edge, which is how callers detect that the
    tour is infeasible in the original sparse graph.
    """

    n = len(distance_matrix)

    if n == 0:
        return [], 0.0

    visited = [False] * n

    tour = [start_city]
    visited[start_city] = True

    current_city = start_city

    for _ in range(n - 1):

        nearest_city = None
        nearest_distance = float("inf")

        for city in range(n):

            if not visited[city]:

                distance = distance_matrix[current_city][city]

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_city = city

        if nearest_city is None:

            for city in range(n):

                if not visited[city]:
                    nearest_city = city
                    break

        tour.append(nearest_city)

        visited[nearest_city] = True
        current_city = nearest_city

    tour_length = calculate_tour_length(
        tour,
        distance_matrix
    )

    return tour, tour_length


# =========================================================
# VERSION 2B
# Top-k Randomized Nearest Neighbor
# =========================================================

def randomized_nearest_neighbor_top_k(
    distance_matrix,
    k=3,
    seed=None,
    random_start=True
):
    """
    Randomized Nearest Neighbor.

    Instead of always selecting the closest city,
    randomly select one city from the k nearest
    unvisited cities.
    """

    rng = random.Random(seed)
    n = len(distance_matrix)

    if n == 0:
        return [], 0.0

    if random_start:
        start_city = rng.randrange(n)
    else:
        start_city = 0

    visited = [False] * n
    visited[start_city] = True
    tour = [start_city]
    current_city = start_city

    for _ in range(n - 1):
        candidates = []

        for city in range(n):

            if not visited[city]:
                candidates.append(
                    (distance_matrix[current_city][city], city)
                )

        # Sort cities according to distance
        candidates.sort(key=lambda x: x[0])

        # Select the k closest candidates
        top_k = candidates[:min(k, len(candidates))]

        # Randomly choose from them
        _, next_city = rng.choice(top_k)

        tour.append(next_city)

        visited[next_city] = True
        current_city = next_city

    tour_length = calculate_tour_length(
        tour,
        distance_matrix
    )

    return tour, tour_length

# =========================================================
# VERSION 2A
# Random starting city
# =========================================================


def randomized_start_nearest_neighbor(distance_matrix, seed=None):
    """
    Standard nearest neighbor but with a randomly selected
    starting city.
    """

    rng = random.Random(seed)

    start_city = rng.randrange(len(distance_matrix))

    return nearest_neighbor(
        distance_matrix,
        start_city=start_city
    )


# =========================================================
# VERSION 2C
# Distance-weighted Randomized NN
# =========================================================

def randomized_nearest_neighbor_weighted(
    distance_matrix,
    seed=None,
    random_start=False
):
    """
    Distance-Weighted Randomized Nearest Neighbor.

    At each step, an unvisited reachable city is selected randomly.
    Closer cities receive higher probability.

    Weight:
        w_i = d_min / d_i

    where:
        d_i   = distance to candidate city i
        d_min = minimum distance among the current candidates

    This relative weighting is numerically more stable than directly
    using 1 / d_i.

    Infinite distances represent unavailable edges and are excluded.
    If the algorithm reaches a city from which no unvisited city is
    reachable, the run is considered infeasible and returns infinity.
    """

    rng = random.Random(seed)

    n = len(distance_matrix)

    # ---------------------------------------------------------
    # Handle empty input
    # ---------------------------------------------------------
    if n == 0:
        return [], 0.0

    # ---------------------------------------------------------
    # Choose starting city
    # ---------------------------------------------------------
    if random_start:
        current = rng.randrange(n)
    else:
        current = 0

    tour = [current]

    visited = [False] * n
    visited[current] = True

    total_length = 0.0

    # ---------------------------------------------------------
    # Construct the tour
    # ---------------------------------------------------------
    while len(tour) < n:

        # Find all unvisited cities that are reachable from
        # the current city.
        candidates = [
            city
            for city in range(n)
            if not visited[city]
            and math.isfinite(distance_matrix[current][city])
            and distance_matrix[current][city] >= 0
        ]

        # No reachable unvisited city.
        # Therefore this run cannot construct a complete tour.
        if not candidates:
            return tour, float("inf")

        # -----------------------------------------------------
        # Calculate distance-weighted probabilities
        # -----------------------------------------------------

        distances = [
            distance_matrix[current][city]
            for city in candidates
        ]

        min_distance = min(distances)

        # Construct weights.
        #
        # Closest city:
        #     weight = min_distance / min_distance = 1
        #
        # Further cities:
        #     weight < 1
        #
        # Therefore closer cities have higher probability.
        weights = []

        for distance in distances:

            # Handle zero distance safely.
            if distance == 0:
                weight = 1.0
            else:
                weight = min_distance / distance

            weights.append(weight)

        # -----------------------------------------------------
        # Validate weights
        # -----------------------------------------------------

        total_weight = sum(weights)

        if (
            not weights
            or not math.isfinite(total_weight)
            or total_weight <= 0
        ):
            return tour, float("inf")

        # -----------------------------------------------------
        # Randomly select next city according to weights
        # -----------------------------------------------------

        next_city = rng.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

        # Add edge length
        total_length += distance_matrix[current][next_city]

        # Update tour
        tour.append(next_city)
        visited[next_city] = True

        current = next_city

    # ---------------------------------------------------------
    # Return to starting city
    # ---------------------------------------------------------

    return_edge = distance_matrix[current][tour[0]]

    # Cannot complete the TSP cycle
    if not math.isfinite(return_edge):
        return tour, float("inf")

    total_length += return_edge

    return tour, total_length
