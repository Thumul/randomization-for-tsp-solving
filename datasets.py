"""TSP instance loading and generation.

Provides a single `Instance` type used by every experiment, built either from a
TSPLIB benchmark file or from a synthetic generator. Nodes are stored 0-based
(TSPLIB's 1-based ids are converted on load) so tours are plain Python indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import tsplib95

DATA_DIR = Path(__file__).parent / "data"

BEST_KNOWN = {
    "berlin52": 7542,
    "eil51": 426,
}


@dataclass
class Instance:
    """A Euclidean TSP instance: named coordinates plus a distance convention."""

    name: str
    coords: np.ndarray  # shape (n, 2), float
    rounded: bool = False  # TSPLIB EUC_2D rounds distances to nearest integer
    best_known: float | None = None
    _matrix: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def n(self) -> int:
        return len(self.coords)

    def distance(self, i: int, j: int) -> float:
        """Distance between two cities, honouring the instance's rounding rule."""
        d = math.dist(self.coords[i], self.coords[j])
        # TSPLIB EUC_2D defines nint(x) as round-half-up, not Python's
        # round-half-to-even, so floor(d + 0.5) is required to match it.
        return math.floor(d + 0.5) if self.rounded else d

    def distance_matrix(self) -> np.ndarray:
        """Full n x n distance matrix, computed once and cached."""
        if self._matrix is None:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            d = np.sqrt((diff**2).sum(axis=-1))
            if self.rounded:
                d = np.floor(d + 0.5)
            self._matrix = d
        return self._matrix

    def tour_length(self, tour: Sequence[int]) -> float:
        """Closed-tour length: visits every city in order and returns to the start."""
        if len(tour) != self.n:
            raise ValueError(f"tour visits {len(tour)} cities, instance has {self.n}")
        if len(set(tour)) != self.n:
            raise ValueError("tour repeats or omits cities")
        m = self.distance_matrix()
        nxt = np.roll(tour, -1)
        return float(m[tour, nxt].sum())

    def __repr__(self) -> str:
        ref = f", best_known={self.best_known}" if self.best_known else ""
        return f"Instance({self.name!r}, n={self.n}, rounded={self.rounded}{ref})"


def load_tsplib(name_or_path: str | Path) -> Instance:
    """Load a TSPLIB instance, by bare name (looked up in data/) or by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA_DIR / f"{path.name}.tsp"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the .tsp file in {DATA_DIR} "
            f"(see README for where to download TSPLIB instances)."
        )

    problem = tsplib95.load(str(path))
    if problem.edge_weight_type != "EUC_2D":
        raise NotImplementedError(
            f"{problem.name}: only EUC_2D instances are supported, "
            f"got {problem.edge_weight_type}"
        )

    # tsplib95 nodes are 1-based; sort by id so row i is city i.
    nodes = sorted(problem.get_nodes())
    coords = np.array([problem.node_coords[i] for i in nodes], dtype=float)

    return Instance(
        name=problem.name,
        coords=coords,
        rounded=True,  # EUC_2D
        best_known=BEST_KNOWN.get(problem.name),
    )


def random_uniform(n: int, seed: int, width: float = 1000.0, height: float = 1000.0) -> Instance:
    """`n` cities drawn uniformly at random from a width x height rectangle."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(0, width, n), rng.uniform(0, height, n)])
    return Instance(name=f"uniform_n{n}_s{seed}", coords=coords)


def clustered(
    n: int,
    seed: int,
    n_clusters: int = 5,
    spread: float = 50.0,
    width: float = 1000.0,
    height: float = 1000.0,
) -> Instance:
    """`n` cities drawn from `n_clusters` Gaussian blobs, to test robustness to
    non-uniform spatial structure. Points are clipped back into the rectangle."""
    rng = np.random.default_rng(seed)
    centers = np.column_stack(
        [rng.uniform(0, width, n_clusters), rng.uniform(0, height, n_clusters)]
    )
    assignment = rng.integers(0, n_clusters, n)
    coords = centers[assignment] + rng.normal(0, spread, (n, 2))
    coords[:, 0] = np.clip(coords[:, 0], 0, width)
    coords[:, 1] = np.clip(coords[:, 1], 0, height)
    return Instance(name=f"clustered_n{n}_k{n_clusters}_s{seed}", coords=coords)


if __name__ == "__main__":
    # Smoke test: verifies the TSPLIB loader against a known-optimal tour and
    # checks that the synthetic generators are reproducible.
    print("=== TSPLIB instances ===")
    for name in ("berlin52", "eil51"):
        inst = load_tsplib(name)
        print(f"  {inst}")

    print("\n=== Distance convention check ===")
    berlin = load_tsplib("berlin52")
    # The published optimal tour for berlin52 (1-based, from berlin52.opt.tour).
    opt_1based = [
        1, 49, 32, 45, 19, 41, 8, 9, 10, 43, 33, 51, 11, 52, 14, 13, 47, 26,
        27, 28, 12, 25, 4, 6, 15, 5, 24, 48, 38, 37, 40, 39, 36, 35, 34, 44,
        46, 16, 29, 50, 20, 23, 30, 2, 7, 42, 21, 17, 3, 18, 31, 22,
    ]
    opt_tour = [i - 1 for i in opt_1based]
    length = berlin.tour_length(opt_tour)
    print(f"  berlin52 optimal tour length = {length:.0f} (expected {BEST_KNOWN['berlin52']})")
    assert length == BEST_KNOWN["berlin52"], "EUC_2D rounding does not match TSPLIB"
    print("  OK - distance function matches TSPLIB EUC_2D")

    print("\n=== Synthetic instances ===")
    for n in (10, 50, 200):
        inst = random_uniform(n, seed=42)
        print(f"  {inst}  first city = {inst.coords[0].round(2)}")
    for n in (50, 200):
        inst = clustered(n, seed=42, n_clusters=5)
        print(f"  {inst}  first city = {inst.coords[0].round(2)}")

    print("\n=== Reproducibility check ===")
    a, b = random_uniform(100, seed=7), random_uniform(100, seed=7)
    c = random_uniform(100, seed=8)
    print(f"  same seed  -> identical coords: {np.array_equal(a.coords, b.coords)}")
    print(f"  diff seed  -> identical coords: {np.array_equal(a.coords, c.coords)}")
    assert np.array_equal(a.coords, b.coords) and not np.array_equal(a.coords, c.coords)
    print("  OK - generators are seed-reproducible")
