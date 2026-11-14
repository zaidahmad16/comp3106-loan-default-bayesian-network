import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from loandefault.discretize import NUMERIC, apply_discretizer, fit_discretizer

NB_BINS = [3, 5, 8]
TREE_DEPTHS = [2, 3, 4, 5, 6, 8]
KNN_NEIGHBOURS = [5, 11, 21, 31]
SEED = 3106


def encode_columns(features, scale):
    categorical = [c for c in features.columns if c not in NUMERIC]
    numeric = [c for c in features.columns if c in NUMERIC]
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", numeric_step, numeric),
    ])


class NaiveBayesModel:
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, features, default):
        self.rules = fit_discretizer(features, self.n_bins)
        binned = apply_discretizer(features, self.rules)
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)
        codes = self.encoder.fit_transform(binned)
        # one extra slot per column holds labels never seen in training
        sizes = [len(c) + 1 for c in self.encoder.categories_]
        self.unseen = [len(c) for c in self.encoder.categories_]
        self.model = CategoricalNB(alpha=1.0, min_categories=sizes)
        self.model.fit(self._fix(codes), np.array(default))
        return self

    def _fix(self, codes):
        codes = codes.copy()
        for j in range(codes.shape[1]):
            codes[codes[:, j] < 0, j] = self.unseen[j]
        return codes.astype(int)

    def predict_proba(self, features):
        binned = apply_discretizer(features, self.rules)
        return self.model.predict_proba(self._fix(self.encoder.transform(binned)))


class TreeModel:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, features, default):
        self.model = make_pipeline(
            encode_columns(features, scale=False),
            DecisionTreeClassifier(max_depth=self.max_depth, random_state=SEED),
        )
        self.model.fit(features, np.array(default))
        return self

    def predict_proba(self, features):
        return self.model.predict_proba(features)


class KNNModel:
    def __init__(self, k):
        self.k = k

    def fit(self, features, default):
        self.model = make_pipeline(
            encode_columns(features, scale=True),
            KNeighborsClassifier(n_neighbors=self.k),
        )
        self.model.fit(features, np.array(default))
        return self

    def predict_proba(self, features):
        return self.model.predict_proba(features)
