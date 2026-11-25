"""Compare customer classification with and without PCA preprocessing."""

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features, labels = make_classification(
    n_samples=1200,
    n_features=30,
    n_informative=7,
    n_redundant=15,
    class_sep=1.0,
    random_state=70,
)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=70, stratify=labels
)
models = [
    ("all 30 features", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))),
    (
        "PCA (95% variance)",
        make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95, random_state=70),
            LogisticRegression(max_iter=1500),
        ),
    ),
]
for name, model in models:
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"{name}: test accuracy={accuracy_score(y_test, prediction):.3f}")
    if name.startswith("PCA"):
        print(f"  retained components: {model.named_steps['pca'].n_components_}")
print("PCA can reduce redundant inputs; held-out accuracy determines whether it helps here.")
