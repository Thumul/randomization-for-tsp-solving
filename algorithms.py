import itertools
import math
import random

import numpy as np


def euclidean_distance(city1, city2):
    """
    Compute the Euclidean distance between two cities.

    Args:
        city1: (x, y) coordinates of the first city.
        city2: (x, y) coordinates of the second city.

    Returns:
        The straight-line distance as a float.
    """
    return math.sqrt(
        (city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2
    )


def create_distance_matrix(cities):
    """
    Build the dense, symmetric matrix of pairwise Euclidean distances.

    Distances are computed with vectorized numpy operations in O(n^2)
    time and memory.

    Args:
        cities: Sequence of n (x, y) coordinate pairs.

    Returns:
        An n x n numpy array where entry [i][j] is the distance between
        city i and city j.
    """
    coords = np.asarray(cities, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]

    return np.sqrt((diff ** 2).sum(axis=-1))


def build_dense_matrix_from_edges(edges, n, missing_value=float("inf")):
    """
    Expand a sparse, possibly directional edge list into a dense matrix.

    The result can be passed to the same nearest-neighbor implementations
    used for Euclidean instances. Entries [i][j] and [j][i] are populated
    independently, so the matrix need not be symmetric. The diagonal is 0.

    Args:
        edges: Iterable of (from, to, distance) tuples, one per directed
            edge that exists.
        n: Number of cities.
        missing_value: Value assigned to every pair with no edge. The
            default, infinity, marks the pair as unreachable.

    Returns:
        An n x n numpy array of edge distances.
    """
    matrix = np.full((n, n), missing_value, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    for a, b, distance in edges:
        matrix[a][b] = distance

    return matrix


def calculate_tour_length(tour, distance_matrix):
    """
    Compute the total length of a closed tour.

    The tour returns from its last city to its first, so the closing edge
    is included.

    Args:
        tour: Ordered list of city indices visiting each city once.
        distance_matrix: n x n matrix of pairwise distances.

    Returns:
        The sum of the edge distances along the closed tour. The result
        is infinite if the tour uses an edge whose distance is infinite.
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
    Solve TSP exactly with the Held-Karp dynamic program.

    The dynamic program stores, for every subset of cities and every
    possible last city, the cheapest path from city 0 that visits exactly
    that subset. Runs in O(n^2 * 2^n) time and O(n * 2^n) memory, which
    limits it to roughly n <= 15.

    Args:
        cities: Sequence of n (x, y) coordinate pairs.

    Returns:
        A (tour, length) pair: the optimal tour as a list of city indices
        starting at city 0, and its total length.
    """

    n = len(cities)

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    distance_matrix = create_distance_matrix(cities)

    # cost[(subset, last)] = (minimum cost of a path that starts at 0,
    # visits exactly `subset` and ends at `last`, predecessor of `last`)
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

    # Reconstruct the tour by following predecessors backward
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
    Deterministic Nearest Neighbor heuristic.

    Starting from `start_city`, repeatedly moves to the closest unvisited
    city until every city has been visited, then closes the tour. Runs in
    O(n^2) time.

    On a sparse or directional graph, where unreachable pairs are stored
    as float("inf"), the current city can have no reachable unvisited
    city. The tour then continues to the first unvisited city in index
    order so that it remains a valid permutation. Its length includes an
    infinite edge, which marks the tour as infeasible in the original
    graph.

    Args:
        distance_matrix: n x n matrix of pairwise distances.
        start_city: Index of the city where the tour begins.

    Returns:
        A (tour, length) pair: the visiting order as a list of city
        indices, and the closed tour length.
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
    Top-k Randomized Nearest Neighbor heuristic.

    At each step, chooses uniformly at random among the k closest
    unvisited cities instead of always taking the closest one. With k=1
    and random_start=False the behavior matches deterministic Nearest
    Neighbor.

    Args:
        distance_matrix: n x n matrix of pairwise distances.
        k: Size of the candidate pool. If fewer than k unvisited cities
            remain, all of them are candidates.
        seed: Seed for the random number generator, for reproducibility.
        random_start: If True, the starting city is drawn uniformly at
            random; otherwise the tour starts at city 0.

    Returns:
        A (tour, length) pair: the visiting order as a list of city
        indices, and the closed tour length.
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

        candidates.sort(key=lambda x: x[0])

        top_k = candidates[:min(k, len(candidates))]

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
    Randomized-start Nearest Neighbor heuristic.

    Runs deterministic Nearest Neighbor from a starting city drawn
    uniformly at random. Every subsequent step is greedy.

    Args:
        distance_matrix: n x n matrix of pairwise distances.
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        A (tour, length) pair: the visiting order as a list of city
        indices, and the closed tour length.
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
    Distance-weighted Randomized Nearest Neighbor heuristic.

    At each step, selects the next city at random from the reachable
    unvisited cities, with closer cities more likely to be chosen. The
    weight of candidate i is

        w_i = d_min / d_i

    where d_i is the distance to candidate i and d_min is the smallest
    distance among the current candidates, so the closest candidate has
    weight 1. The selection probability is w_i divided by the sum of the
    weights. Normalizing by d_min avoids the scale dependence of 1 / d_i.

    Edges with infinite distance are treated as unavailable and are
    excluded from the candidates. If the walk reaches a city with no
    reachable unvisited city, or cannot return to the start, the run is
    infeasible and the returned length is infinity.

    Args:
        distance_matrix: n x n matrix of pairwise distances.
        seed: Seed for the random number generator, for reproducibility.
        random_start: If True, the starting city is drawn uniformly at
            random; otherwise the tour starts at city 0.

    Returns:
        A (tour, length) pair: the visiting order as a list of city
        indices (partial if the run is infeasible), and the closed tour
        length, which is infinity for an infeasible run.
    """

    rng = random.Random(seed)

    n = len(distance_matrix)

    if n == 0:
        return [], 0.0

    if random_start:
        current = rng.randrange(n)
    else:
        current = 0

    tour = [current]

    visited = [False] * n
    visited[current] = True

    total_length = 0.0

    while len(tour) < n:

        # Unvisited cities reachable from the current city
        candidates = [
            city
            for city in range(n)
            if not visited[city]
            and math.isfinite(distance_matrix[current][city])
            and distance_matrix[current][city] >= 0
        ]

        # Dead end: no complete tour can be built from here
        if not candidates:
            return tour, float("inf")

        distances = [
            distance_matrix[current][city]
            for city in candidates
        ]

        min_distance = min(distances)

        weights = []

        for distance in distances:

            # A zero distance would divide by zero; treat it as the
            # highest-priority candidate.
            if distance == 0:
                weight = 1.0
            else:
                weight = min_distance / distance

            weights.append(weight)

        total_weight = sum(weights)

        if (
            not weights
            or not math.isfinite(total_weight)
            or total_weight <= 0
        ):
            return tour, float("inf")

        next_city = rng.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

        total_length += distance_matrix[current][next_city]

        tour.append(next_city)
        visited[next_city] = True

        current = next_city

    # Close the cycle by returning to the starting city
    return_edge = distance_matrix[current][tour[0]]

    if not math.isfinite(return_edge):
        return tour, float("inf")

    total_length += return_edge

    return tour, total_length
