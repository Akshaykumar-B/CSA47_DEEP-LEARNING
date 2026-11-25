"""Demonstrate floating-point precision and cancellation with NumPy types."""

import numpy as np


print("Representing 1 + a very small increment:")
for dtype in (np.float16, np.float32, np.float64):
    one = dtype(1.0)
    increment = dtype(1e-4)
    print(
        f"{dtype.__name__:8}: stored increment={increment!r}, "
        f"(1 + increment) - 1={dtype(one + increment - one)!r}"
    )

print("\nCatastrophic cancellation (subtracting nearly equal large values):")
for dtype in (np.float32, np.float64):
    large = dtype(1e8)
    small = dtype(1.0)
    print(
        f"{dtype.__name__:8}: (large + small) - large="
        f"{dtype(large + small - large)!r} (expected 1.0)"
    )

print("\nOverflow/underflow range:")
with np.errstate(over="ignore", under="ignore"):
    for dtype in (np.float16, np.float32, np.float64):
        print(f"{dtype.__name__:8}: 1e4 * 1e4={dtype(dtype(1e4) * dtype(1e4))!r}")

print("Lower precision saves memory but rounds more aggressively and has a narrower range.")
print("Use stable algorithms and an appropriate dtype when accuracy matters.")
