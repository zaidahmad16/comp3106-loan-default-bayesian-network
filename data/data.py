import json
import numpy as np
import pandas as pd

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
]

TEST_SIZE = 0.2
SEED = 3106


def load_german(path):
    column_names = COLUMNS + ["label"]
    data = pd.read_csv(path, sep=r"\s+", header=None, names=column_names)

    if data.shape != (1000, 21):
        raise ValueError("expected 1000 rows and 21 columns, got " + str(data.shape))
    if data.isna().any().any():
        raise ValueError("german credit should have no missing values")

    default = (data["label"] == 2).astype(int)
    features = data.drop(columns=["label"])
    return features, default


def make_split(default):
    rng = np.random.default_rng(SEED)
    default = np.array(default)
    train = []
    test = []
    for label in [0, 1]:
        indices = np.where(default == label)[0]
        rng.shuffle(indices)
        test_count = int(round(len(indices) * TEST_SIZE))
        test.extend(indices[:test_count].tolist())
        train.extend(indices[test_count:].tolist())
    return np.array(sorted(train)), np.array(sorted(test))


def save_split(path, train_indices, test_indices):
    split = {"train": train_indices.tolist(), "test": test_indices.tolist()}
    with open(path, "w") as file:
        json.dump(split, file)


def load_split(path):
    with open(path) as file:
        split = json.load(file)
    return np.array(split["train"]), np.array(split["test"])