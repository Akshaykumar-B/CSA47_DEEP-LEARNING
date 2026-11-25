"""Mean, population variance, and standard deviation of examination marks."""

from math import sqrt


marks = [72, 75, 68, 80, 77, 74, 71, 83]
mean = sum(marks) / len(marks)
variance = sum((mark - mean) ** 2 for mark in marks) / len(marks)
standard_deviation = sqrt(variance)

print("Marks:", marks)
print(f"Mean: {mean:.2f}")
print(f"Population variance: {variance:.2f}")
print(f"Population standard deviation: {standard_deviation:.2f}")
print("The standard deviation is small relative to the mean, so these marks are fairly consistent.")
