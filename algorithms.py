import itertools
import math
import random


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
    Precompute distances between every pair of cities.
    """
    n = len(cities)

    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(cities[i], cities[j])

            matrix[i][j] = distance
            matrix[j][i] = distance

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

def nearest_neighbor(cities, start_city=0):
    """
    Deterministic Nearest Neighbor TSP heuristic.
    At each step, select the closest unvisited city.
    """

    n = len(cities)

    if n == 0:
        return [], 0.0

    distance_matrix = create_distance_matrix(cities)

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
    cities,
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
    n = len(cities)

    if n == 0:
        return [], 0.0

    distance_matrix = create_distance_matrix(cities)

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


def randomized_start_nearest_neighbor(cities, seed=None):
    """
    Standard nearest neighbor but with a randomly selected
    starting city.
    """

    rng = random.Random(seed)

    start_city = rng.randrange(len(cities))

    return nearest_neighbor(
        cities,
        start_city=start_city
    )


# =========================================================
# VERSION 2C
# Distance-weighted Randomized NN
# =========================================================

def randomized_nearest_neighbor_weighted(
    cities,
    seed=None,
    random_start=True
):
    """
    Distance-weighted randomized nearest neighbor.

    Closer cities receive higher probability but the
    closest city is not always selected.

    Probability is proportional to:

        1 / distance
    """

    rng = random.Random(seed)

    n = len(cities)

    if n == 0:
        return [], 0.0

    distance_matrix = create_distance_matrix(cities)

    if random_start:
        start_city = rng.randrange(n)
    else:
        start_city = 0

    visited = [False] * n

    visited[start_city] = True

    tour = [start_city]

    current_city = start_city

    epsilon = 1e-12

    for _ in range(n - 1):

        candidates = []
        weights = []

        for city in range(n):

            if not visited[city]:

                distance = distance_matrix[current_city][city]

                candidates.append(city)

                # Closer cities get larger probability
                weights.append(
                    1.0 / (distance + epsilon)
                )

        next_city = rng.choices(
            candidates,
            weights=weights,
            k=1
        )[0]

        tour.append(next_city)

        visited[next_city] = True
        current_city = next_city

    tour_length = calculate_tour_length(
        tour,
        distance_matrix
    )

    return tour, tour_length
