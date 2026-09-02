import random


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