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
# VERSION 1
# Deterministic Nearest Neighbor
# =========================================================

def nearest_neighbor(distance_matrix, start_city=0):
    """
    Deterministic Nearest Neighbor TSP heuristic.
    At each step, select the closest unvisited city.
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
