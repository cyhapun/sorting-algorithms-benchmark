import numpy as np
import math

SIZES = [100, 1_000, 10_000, 100_000]
TYPES = ["random", "nearly_sorted", "many_duplicates", "reverse_sorted"]

def check(dtype, n):
    filepath = f"data/n_{n}/{dtype}_n{n}.npz"
    arr = np.load(filepath)["data"]

    size_ok = len(arr) == n

    inversions = sum(1 for i in range(n - 1) if arr[i] > arr[i + 1])
    inversion_rate = inversions / (n - 1) * 100
    nearly_ok = inversion_rate < 10

    distinct = len(set(arr.tolist()))
    expected = int(math.sqrt(n))
    duplicates_ok = distinct <= expected * 2

    is_reverse = all(arr[i] >= arr[i + 1] for i in range(n - 1))

    print(f"{filepath}")
    print(f"  size           : {len(arr)} {'OK' if size_ok else 'FAIL'}")
    print(f"  nearly_sorted  : {inversion_rate:.2f}% nghich the {'OK' if nearly_ok else 'FAIL'}")
    print(f"  many_duplicates: {distinct} gia tri phan biet (expected ~{expected}) {'OK' if duplicates_ok else 'FAIL'}")
    print(f"  reverse_sorted : {'OK' if is_reverse else 'FAIL'}")
    print()

for n in SIZES:
    for dtype in TYPES:
        check(dtype, n)