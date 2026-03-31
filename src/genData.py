import numpy as np
import random
import math
import os

SIZES = [100, 1_000, 10_000, 100_000]
random.seed(42)

def generate_random(n):
    return [random.randint(1, n * 10) for _ in range(n)]

def generate_nearly_sorted(n):
    arr = list(range(1, n + 1))
    swaps = max(1, n // 50)
    for _ in range(swaps):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def generate_many_duplicates(n):
    distinct = max(2, int(math.sqrt(n)))
    return [random.randint(1, distinct) for _ in range(n)]

def generate_reverse_sorted(n):
    return list(range(n, 0, -1))

GENERATORS = {
    "random":          generate_random,
    "nearly_sorted":   generate_nearly_sorted,
    "many_duplicates": generate_many_duplicates,
    "reverse_sorted":  generate_reverse_sorted,
}

for n in SIZES:
    folder = os.path.join("data", f"n_{n}")
    os.makedirs(folder, exist_ok=True)
    for dtype, gen_fn in GENERATORS.items():
        arr = np.array(gen_fn(n), dtype=np.int64)
        filepath = os.path.join(folder, f"{dtype}_n{n}.npz")
        np.savez_compressed(filepath, data=arr)
        print(f"data/n_{n}/{dtype}_n{n}.npz")