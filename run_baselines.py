import pandas as pd

from loandefault.data import load_german, load_split
from loandefault.evaluate import cross_validate, make_folds
from loandefault.models import (
    KNN_NEIGHBOURS, NB_BINS, TREE_DEPTHS, KNNModel, NaiveBayesModel, TreeModel,
)

SCORES = ["accuracy", "precision", "recall", "auc"]


def make_naive_bayes(n_bins):
    def make_model():
        return NaiveBayesModel(n_bins)
    return make_model


def make_tree(max_depth):
    def make_model():
        return TreeModel(max_depth)
    return make_model


def make_knn(k):
    def make_model():
        return KNNModel(k)
    return make_model


def summarize(name, setting, results):
    row = {"model": name, "setting": setting}
    for score in SCORES:
        row[score] = results[score].mean()
        row[score + "_std"] = results[score].std()
    return row


features, default = load_german("data/german.data")
train, test = load_split("data/german_split.json")
x = features.iloc[train].reset_index(drop=True)
y = default.iloc[train].reset_index(drop=True)
folds = make_folds(y)

table = []
for n_bins in NB_BINS:
    results = cross_validate(make_naive_bayes(n_bins), x, y, folds)
    table.append(summarize("naive_bayes", "bins=" + str(n_bins), results))
for max_depth in TREE_DEPTHS:
    results = cross_validate(make_tree(max_depth), x, y, folds)
    table.append(summarize("decision_tree", "depth=" + str(max_depth), results))
for k in KNN_NEIGHBOURS:
    results = cross_validate(make_knn(k), x, y, folds)
    table.append(summarize("nearest_neighbour", "k=" + str(k), results))

print(pd.DataFrame(table).round(3).to_string(index=False))