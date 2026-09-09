from datasets import (
    generate_random_euclidean
)

from experiments import (
    run_deterministic,
    run_randomized_top_k
)


def print_results(name, deterministic, randomized):

    improvement = (deterministic["length"] - randomized["best_length"])
    improvement_percentage = ( improvement / deterministic["length"]) * 100

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Deterministic NN length : {deterministic['length']:.2f}")
    print(f"Deterministic NN time   : {deterministic['time']:.6f} s")

    print()

    print(f"Randomized best length  : {randomized['best_length']:.2f}")
    print(f"Randomized mean length  : {randomized['mean_length']:.2f}")
    print(f"Randomized worst length : {randomized['worst_length']:.2f}")
    print(f"Standard deviation      : {randomized['std_length']:.2f}")
    print(f"Average execution time  : {randomized['mean_time']:.6f} s")
    print(f"Best improvement        : {improvement_percentage:.2f}%")


def main():

    cities = generate_random_euclidean(n=500, seed=94)

    deterministic = run_deterministic(cities)
    randomized_top_k = run_randomized_top_k(cities, k=5, runs=500)

    print_results(
        "Top-k Randomized NN",
        deterministic,
        randomized_top_k
    )


if __name__ == "__main__":
    main()