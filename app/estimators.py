"""
Reference estimators, used by the scenario grid and the test suite.

These exist so that the app can DEMONSTRATE the thing it was built to teach: that
a benchmark with known truth lets you score an estimator. They are deliberately
simple and deliberately correctly specified for our own data-generating process --
which makes them a usable oracle for testing the generator, and makes them a poor
template for real analysis. On real data you would reach for TMLE, AIPW or DML with
cross-fitting and a flexible learner library (Module 4).

WHY G-COMPUTATION IS WRITTEN THE LONG WAY
-----------------------------------------
The tempting shortcut is to regress Y on Z and covariates and read off the
coefficient on Z. That works only when the treatment effect is CONSTANT. As soon as
the effect varies with X -- which is the whole point of Tier 1 -- the coefficient
on Z is no longer the average treatment effect.

So we do it properly:
  1. fit a model for E[Y | Z, X], including Z-by-X interactions;
  2. predict every row TWICE, once forcing Z = 1 and once forcing Z = 0;
  3. average the difference.

Step 2 is the g-formula made literal: we ask the model what would have happened to
each person under each arm, including the arm they did not receive. That is exactly
the counterfactual quantity the design gives us the true answer for.
"""

import numpy as np
import pandas as pd


def _design(data, treatment_col, covariates, interactions):
    """Build [1, Z, X, Z*X] as a plain float matrix."""
    X = pd.get_dummies(data[list(covariates)], drop_first=True).values.astype(np.float64)
    Z = data[treatment_col].values.astype(np.float64).reshape(-1, 1)
    ones = np.ones((len(data), 1))
    if interactions:
        return np.hstack([ones, Z, X, Z * X]), X
    return np.hstack([ones, Z, X]), X


def naive_ate(data, outcome_col, treatment_col):
    """Treated mean minus control mean. Adjusts for nothing; biased under confounding."""
    treated = data.loc[data[treatment_col] == 1, outcome_col].mean()
    control = data.loc[data[treatment_col] == 0, outcome_col].mean()
    return float(treated - control)


def g_computation(data, outcome_col, treatment_col, covariates, interactions=True):
    """
    Estimate the ATE by outcome regression plus standardization (the g-formula).

    With `interactions=True` the fitted model can represent an effect that varies
    with X, so the returned average is a genuine ATE rather than a coefficient that
    happens to coincide with one.
    """
    design, X = _design(data, treatment_col, covariates, interactions)
    y = data[outcome_col].values.astype(np.float64)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    n = len(data)
    ones = np.ones((n, 1))
    one_col, zero_col = np.ones((n, 1)), np.zeros((n, 1))
    if interactions:
        d1 = np.hstack([ones, one_col, X, one_col * X])
        d0 = np.hstack([ones, zero_col, X, zero_col * X])
    else:
        d1 = np.hstack([ones, one_col, X])
        d0 = np.hstack([ones, zero_col, X])

    return float(np.mean(d1.dot(beta) - d0.dot(beta)))


def cate_predictions(data, outcome_col, treatment_col, covariates):
    """
    Per-person predicted benefit from the same interacted outcome model.

    This is the quantity the survcausal-pilot calls the CATE, and the one its
    risk/benefit quadrants are built from. Here it lets the app check whether an
    estimator recovers not just the average effect but its VARIATION -- which is
    only a meaningful question once tau(X) is non-constant.
    """
    design, X = _design(data, treatment_col, covariates, interactions=True)
    y = data[outcome_col].values.astype(np.float64)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    n = len(data)
    ones = np.ones((n, 1))
    one_col, zero_col = np.ones((n, 1)), np.zeros((n, 1))
    d1 = np.hstack([ones, one_col, X, one_col * X])
    d0 = np.hstack([ones, zero_col, X, zero_col * X])
    return d1.dot(beta) - d0.dot(beta)
