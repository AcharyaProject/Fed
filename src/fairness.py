"""
fairness.py
===========
Model training (the federated-averaged classifier h, held fixed after
certification) and the demographic-parity fairness metric F(P, h).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["age", "education", "hours", "occ_code"]


def federated_average_train(client_dfs, seed=0, feature_cols=None):
    """
    Simulates one round of FedAvg: trains a local logistic-regression
    model per client on (features -> Y), then averages coefficients
    weighted by client size. Returns (scaler, averaged model).

    NOTE: earlier versions of this function also returned a Lipschitz
    constant `L`, used by a Lipschitz-based fairness bound in bound.py.
    That bound was not Theorem 1 (see bound.py's module docstring) --
    Theorem 1 requires only that h be bounded in [0,1], which any
    predict_proba output already satisfies -- so no Lipschitz constant
    is needed anywhere in the corrected pipeline.

    `feature_cols` defaults to the ACS-style FEATURE_COLS but can be
    overridden for other datasets (e.g. FLamby's Fed-Heart-Disease),
    so this function is not tied to any one data source.
    """
    feature_cols = feature_cols or FEATURE_COLS
    all_X = np.vstack([df[feature_cols].values for df in client_dfs.values()])
    scaler = StandardScaler().fit(all_X)

    coefs, intercepts, weights = [], [], []
    for df in client_dfs.values():
        X = scaler.transform(df[feature_cols].values)
        y = df["Y"].values
        if len(np.unique(y)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, y)
        coefs.append(clf.coef_.ravel())
        intercepts.append(clf.intercept_[0])
        weights.append(len(df))

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    avg_coef = np.average(np.array(coefs), axis=0, weights=weights)
    avg_intercept = np.average(np.array(intercepts), weights=weights)

    global_clf = LogisticRegression()
    global_clf.coef_ = avg_coef.reshape(1, -1)
    global_clf.intercept_ = np.array([avg_intercept])
    global_clf.classes_ = np.array([0, 1])

    return scaler, global_clf


def demographic_parity(client_dfs, scaler, model, feature_cols=None):
    """Computes DP(P, h) pooled across all clients. `feature_cols` overridable as above."""
    feature_cols = feature_cols or FEATURE_COLS
    X = np.vstack([df[feature_cols].values for df in client_dfs.values()])
    A = np.concatenate([df["A"].values for df in client_dfs.values()])
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]

    p0 = probs[A == 0].mean() if (A == 0).any() else 0.0
    p1 = probs[A == 1].mean() if (A == 1).any() else 0.0
    return abs(p0 - p1)


def accuracy(client_dfs, scaler, model, feature_cols=None):
    feature_cols = feature_cols or FEATURE_COLS
    X = np.vstack([df[feature_cols].values for df in client_dfs.values()])
    y = np.concatenate([df["Y"].values for df in client_dfs.values()])
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    return (preds == y).mean()
