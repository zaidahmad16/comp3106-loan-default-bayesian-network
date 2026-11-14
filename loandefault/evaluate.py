import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

N_FOLDS = 5
N_REPEATS = 5
SEED = 3106
CUTOFF = 0.5


def make_folds(default):
    default = np.array(default)
    folds = []
    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(SEED + repeat)
        groups = []
        for k in range(N_FOLDS):
            groups.append([])
        for label in [0, 1]:
            indices = np.where(default == label)[0]
            rng.shuffle(indices)
            for position in range(len(indices)):
                groups[position % N_FOLDS].append(int(indices[position]))
        for k in range(N_FOLDS):
            train = []
            for j in range(N_FOLDS):
                if j != k:
                    train.extend(groups[j])
            folds.append((np.array(sorted(train)), np.array(sorted(groups[k]))))
    return folds


def get_scores(default, probabilities, cutoff=CUTOFF):
    default = np.array(default)
    probabilities = np.array(probabilities)
    predicted = (probabilities >= cutoff).astype(int)
    return {
        "accuracy": accuracy_score(default, predicted),
        "precision": precision_score(default, predicted, zero_division=0),
        "recall": recall_score(default, predicted, zero_division=0),
        "auc": roc_auc_score(default, probabilities),
    }


def cross_validate(make_model, features, default, folds):
    default = np.array(default)
    rows = []
    for i in range(len(folds)):
        train, test = folds[i]
        model = make_model()
        model.fit(features.iloc[train], default[train])
        probabilities = model.predict_proba(features.iloc[test])[:, 1]
        scores = get_scores(default[test], probabilities)
        scores["repeat"] = i // N_FOLDS
        scores["fold"] = i % N_FOLDS
        rows.append(scores)
    return pd.DataFrame(rows)
