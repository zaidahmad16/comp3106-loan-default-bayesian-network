import numpy as np
import pandas as pd

NUMERIC = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "people_liable",
]

RARE_FRACTION = 0.01
RARE = "other"


def make_labels(values, edges):
    if edges is None:
        return values.astype(str)
    codes = np.searchsorted(edges, values.to_numpy(dtype=float), side="left")
    return pd.Series(codes, index=values.index).astype(str)


def fit_discretizer(features, n_bins):
    rules = {}
    min_count = max(2, int(np.ceil(RARE_FRACTION * len(features))))

    for column in features.columns:
        values = features[column]
        edges = None
        if column in NUMERIC and values.nunique() > n_bins:
            cut_points = []
            for i in range(1, n_bins):
                cut_points.append(values.quantile(i / n_bins))
            edges = sorted(set(cut_points))

        labels = make_labels(values, edges)
        counts = labels.value_counts()
        keep = []
        for label in counts.index:
            if counts[label] >= min_count:
                keep.append(label)

        rules[column] = {"edges": edges, "keep": keep}
    return rules


def apply_discretizer(features, rules):
    result = pd.DataFrame(index=features.index)
    for column in features.columns:
        rule = rules[column]
        labels = make_labels(features[column], rule["edges"])
        result[column] = labels.where(labels.isin(rule["keep"]), RARE)
    return result