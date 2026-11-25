"""Use factor analysis to explore latent factors in simulated student records."""

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(75)
count = 500
effort = rng.normal(size=count)
engagement = rng.normal(size=count)
measurements = np.column_stack([
    0.85 * effort + 0.15 * engagement,
    0.75 * effort + rng.normal(0, 0.3, count),
    0.65 * engagement + 0.25 * effort,
    0.8 * engagement + rng.normal(0, 0.35, count),
    0.7 * effort + 0.3 * engagement,
    0.6 * effort + rng.normal(0, 0.4, count),
])
feature_names = [
    "exam_marks", "assignment_score", "attendance", "participation",
    "homework_completion", "study_hours",
]
standardized = StandardScaler().fit_transform(measurements)
model = FactorAnalysis(n_components=2, random_state=75)
factor_scores = model.fit_transform(standardized)
print("Factor-score shape:", factor_scores.shape)
for factor_index, loadings in enumerate(model.components_, start=1):
    strongest = np.argsort(np.abs(loadings))[-3:][::-1]
    summary = ", ".join(
        f"{feature_names[index]} ({loadings[index]:+.2f})" for index in strongest
    )
    print(f"Factor {factor_index} strongest measures: {summary}")
print("Factors are exploratory latent dimensions, not causal conclusions.")
