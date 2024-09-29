import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier


def feature_importance_forest(
    X: ArrayLike, y: ArrayLike, feature_names: list[str]
) -> tuple[pd.Series[float], NDArray[np.floating]]:
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)

    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    return forest_importances, std
