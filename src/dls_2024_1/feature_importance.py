import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def feature_importance_forest(
    X: ArrayLike, y: ArrayLike, feature_names
) -> tuple[pd.Series, NDArray[np.floating]]:
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)

    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    return forest_importances, std


class FeatureImportanceForest:
    def __init__(self, X, y, feature_names) -> None:
        self.forest = RandomForestClassifier(random_state=0)
        self.forest.fit(X, y)

        importances = self.forest.feature_importances_
        self._forest_importances = pd.Series(importances, index=feature_names)
        self._std = np.std([tree.feature_importances_ for tree in self.forest.estimators_], axis=0)

    @property
    def forest_importances(self):
        return self._forest_importances

    def plot(self):
        fig, ax = plt.subplots()
        self._forest_importances.plot.bar(yerr=self._std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()


class PermutationImportanceForest:
    def __init__(self, X, y) -> None:
        self.forest = RandomForestClassifier(random_state=0)
        self.forest.fit(X, y)

        self._result = permutation_importance(
            self.forest, X, y, n_repeats=10, random_state=42, n_jobs=2
        )

        self._forest_importances = pd.Series(self._result["importances_mean"], index=X.columns)

    @property
    def forest_importances(self):
        return self._forest_importances

    def plot(self):
        fig, ax = plt.subplots()
        self._forest_importances.plot.bar(yerr=self._result["importances_std"], ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()
