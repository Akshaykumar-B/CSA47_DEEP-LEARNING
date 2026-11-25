"""Reduce Iris measurements with PCA and compare species classification."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=68, stratify=iris.target
)
for name, transform in (
    ("original 4 features", "passthrough"),
    ("2 PCA components", PCA(n_components=2)),
):
    model = make_pipeline(
        StandardScaler(),
        transform,
        LogisticRegression(max_iter=1000, random_state=68),
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"{name}: accuracy={accuracy_score(y_test, prediction):.3f}")
print("PCA coordinates retain the largest variance directions, not necessarily every class cue.")
