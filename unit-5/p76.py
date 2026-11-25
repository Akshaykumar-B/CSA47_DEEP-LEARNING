"""Estimate latent purchasing-behaviour factors from simulated customer data."""

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(76)
count = 700
price_sensitive = rng.normal(size=count)
brand_loyal = rng.normal(size=count)
impulsive = rng.normal(size=count)
measurements = np.column_stack([
    0.85 * price_sensitive + rng.normal(0, 0.25, count),
    0.75 * price_sensitive + rng.normal(0, 0.3, count),
    0.7 * price_sensitive - 0.2 * brand_loyal + rng.normal(0, 0.3, count),
    0.85 * brand_loyal + rng.normal(0, 0.25, count),
    0.65 * brand_loyal + rng.normal(0, 0.35, count),
    0.7 * brand_loyal + rng.normal(0, 0.3, count),
    0.85 * impulsive + rng.normal(0, 0.3, count),
    0.75 * impulsive + rng.normal(0, 0.35, count),
    0.5 * impulsive + 0.25 * brand_loyal + rng.normal(0, 0.4, count),
])
feature_names = [
    "discount_share", "budget_item_share", "price_comparisons",
    "repeat_brand_rate", "loyalty_visits", "brand_searches",
    "late_night_orders", "impulse_item_share", "average_basket_size",
]
standardized = StandardScaler().fit_transform(measurements)
model = FactorAnalysis(n_components=3, random_state=76)
scores = model.fit_transform(standardized)
print("Customer-factor score shape:", scores.shape)
for factor_index, loadings in enumerate(model.components_, start=1):
    strongest = np.argsort(np.abs(loadings))[-3:][::-1]
    summary = ", ".join(
        f"{feature_names[index]} ({loadings[index]:+.2f})" for index in strongest
    )
    print(f"Factor {factor_index} high-loading behaviours: {summary}")
print("Factor signs and ordering are arbitrary; interpret groups of related loadings.")
