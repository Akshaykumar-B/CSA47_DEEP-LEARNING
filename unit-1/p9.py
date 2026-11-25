"""Bayes' theorem for a positive medical diagnostic test."""

prevalence = 0.01
sensitivity = 0.95
specificity = 0.90

true_positive = sensitivity * prevalence
false_positive = (1 - specificity) * (1 - prevalence)
positive_probability = true_positive + false_positive
posterior_probability = true_positive / positive_probability

print(f"P(positive test): {positive_probability:.4f}")
print(f"P(disease | positive): {posterior_probability:.4f} ({posterior_probability:.1%})")
print("Even with a sensitive test, low prevalence means some positive results are false positives.")
