"""Generate temperatures and compare their histogram with a normal model."""

from math import erf, sqrt
from random import Random


mean, standard_deviation, sample_count = 20.0, 5.0, 10_000
rng = Random(42)
observations = [rng.gauss(mean, standard_deviation) for _ in range(sample_count)]
sample_mean = sum(observations) / sample_count
sample_variance = sum((value - sample_mean) ** 2 for value in observations) / sample_count

boundaries = [float("-inf"), 15.0, 20.0, 25.0, float("inf")]
observed_counts = [
    sum(lower <= value < upper for value in observations)
    for lower, upper in zip(boundaries, boundaries[1:])
]


def normal_cdf(value):
    return 0.5 * (1 + erf((value - mean) / (standard_deviation * sqrt(2))))


expected_probabilities = [
    normal_cdf(15),
    normal_cdf(20) - normal_cdf(15),
    normal_cdf(25) - normal_cdf(20),
    1 - normal_cdf(25),
]
expected_counts = [probability * sample_count for probability in expected_probabilities]
chi_square = sum(
    (observed - expected) ** 2 / expected
    for observed, expected in zip(observed_counts, expected_counts)
)

print(f"Sample mean: {sample_mean:.2f} (expected {mean:.2f})")
print(f"Sample standard deviation: {sqrt(sample_variance):.2f} (expected {standard_deviation:.2f})")
print("Bins: (-inf, 15), [15, 20), [20, 25), [25, inf)")
print("Observed counts:", observed_counts)
print("Expected counts:", [round(value, 1) for value in expected_counts])
print(f"Chi-square statistic: {chi_square:.2f} (3 degrees of freedom)")
print(
    "At the 5% level, the binned observations are consistent with the model."
    if chi_square < 7.815
    else "At the 5% level, the binned observations differ from the model."
)
