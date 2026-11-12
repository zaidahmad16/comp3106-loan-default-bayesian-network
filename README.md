# Explainable Bayesian Networks for Loan Default

COMP 3106 (Introduction to AI), Carleton University, Fall 2026. Zaid Ahmad.

## What this project is

Banks have to predict who is likely to default on a loan, and they also have to explain
that prediction to someone. A Bayesian network is a graph where each node is something we
know about the borrower (income, loan amount, credit history) and the arrows show which
things influence which. Every node has a probability table, so you can ask what happens to
a person's risk if the loan is shorter and get a real number back.

## Objectives

1. Learn Bayesian networks on two public credit datasets and compare them with other
   models from class on accuracy and cost.
2. Test whether limiting which arrows the network can learn (for example, a loan cannot
   cause someone's age) keeps accuracy while making the network easier to trust.
3. Put numbers on how explainable the networks are, instead of judging the graphs by eye.
4. Build a small demo: enter a borrower, see the predicted risk and what drives it.

## Methods

The Bayesian network code (arrow search, probability tables and inference) is written from
scratch in Python. scikit-learn is used only for the comparison models and the scores.

- **Data preparation:** number columns such as income are turned into ranges (3, 5 or 8,
  chosen on validation data). Very rare categories are merged, and a missing value becomes
  its own category.
- **Three networks:**
  1. Naive Bayes: the outcome points to every feature and the features do not connect.
  2. Learned network without rules: a search that starts with no arrows and tries adding,
     removing or flipping one, keeping a change if the network fits the data better.
  3. Learned network with rules: the same search, but features are put in an order
     (fixed traits, then financial history, then loan terms, then default) and arrows can
     only point forward. Each node has at most 3 incoming arrows.
- **Probabilities and questions:** tables are filled by counting, with a small added count
  so nothing ends up with probability zero. Questions are answered with variable
  elimination. A borrower is predicted to default when the chance is above a cutoff chosen
  on validation data.
- **Models to compare against:** Naive Bayes, a decision tree and nearest neighbour.

## Datasets

- German Credit: https://archive.ics.uci.edu/dataset/144 (put `german.data` in `data/`)
- Give Me Some Credit: https://www.kaggle.com/c/GiveMeSomeCredit (file `cs-training.csv`, put it in `data/`)

## Validation

For each dataset, 20% of the rows are kept aside as a test set and not touched until the
end. The other 80% is used to pick settings with 5-fold cross-validation, repeated 5 times.
Every split keeps the same share of defaults.

- **Prediction:** accuracy, precision, recall, F1 and AUC, plus the 5-to-1 cost on German
  Credit. On Give Me Some Credit the focus is precision, recall, F1 and AUC, because the
  classes are so uneven.

- **Explainability:** how often each arrow shows up when the network is relearned on 50
  random samples, whether what-if answers move in the obvious direction, and whether the
  Markov blanket of the default node matches the features the decision tree splits on.


## Results

To be filled in.