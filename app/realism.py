"""
Realism diagnostics: is the synthetic data actually like the real data?

WHY THIS FILE EXISTS
--------------------
The app used to tell participants that its output was "realistic and
near-indistinguishable from the observed sample", and then showed them a density
plot of Y -- which is 100% simulated and therefore says nothing whatsoever about
covariate realism. The claim was never checked.

That is a bad habit to model in a workshop about auditing causal claims. Module E
asks participants to accept, correct or reject a claim on the evidence; an app
that asserts its own quality without evidence is the thing they are being trained
to catch. So Tier 0 turns the claim into a measurement.

Three complementary checks, weakest to strongest:

  1. Standardized mean difference (SMD) -- per covariate, are the MEANS close?
  2. Kolmogorov-Smirnov distance (KS)   -- per covariate, are the WHOLE
                                           DISTRIBUTIONS close?
  3. Discriminator AUC                  -- can a machine learning model tell real
                                           rows from synthetic ones, using all
                                           covariates JOINTLY?

Why all three: a generator can match every marginal mean (SMD ~ 0) while getting
the shapes wrong (KS large). It can match every marginal distribution (KS ~ 0)
while destroying the CORRELATIONS between covariates -- and only the
discriminator, which sees whole rows, will notice that. Passing 1 and 2 but
failing 3 is the classic signature of a generator that has learned the margins
and not the joint.

THE HOLDOUT, AND WHY IT IS NOT OPTIONAL
---------------------------------------
These metrics must compare synthetic data against real rows the generator NEVER
TRAINED ON. Compared against its own training rows, a sufficiently flexible
generator scores well by memorizing -- which is both a wrong answer (it says
nothing about generalization) and a privacy problem (a generator that memorizes
patients can leak them). The caller is responsible for the split; see
tests/test_tier0.py and Step 2 in app.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def standardized_mean_difference(a, b):
    """
    SMD = (mean_a - mean_b) / sqrt( (var_a + var_b) / 2 )

    The difference in means expressed in pooled standard deviations, so it is
    unitless and comparable across covariates measured in dollars and in years.
    The convention in the applied literature -- and the one used by the companion
    survcausal-pilot R study for its arm-balance diagnostics -- is that |SMD| below
    0.1 counts as negligible imbalance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        # Identical constants are perfectly matched; a constant vs a non-constant
        # is maximally mismatched but has no scale to express it in.
        return 0.0 if np.allclose(a.mean(), b.mean()) else np.inf
    return float((a.mean() - b.mean()) / pooled)


def ks_distance(a, b):
    """
    Two-sample Kolmogorov-Smirnov statistic: the largest vertical gap between the
    two empirical cumulative distribution functions.

    Written out by hand rather than imported, because it is four lines and seeing
    them makes the statistic concrete: walk along the number line, at every point
    ask "what fraction of each sample is at or below here?", and report the
    biggest disagreement. 0 means the distributions coincide; 1 means they do not
    overlap at all.

    Unlike the SMD this notices differences in spread, skew and multimodality --
    two samples can have identical means and a KS distance of 0.4.
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side='right') / float(len(a))
    cdf_b = np.searchsorted(b, grid, side='right') / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def covariate_report(X_real, X_syn, feature_names):
    """
    Per-covariate SMD and KS distance, real (held out) vs synthetic.

    Returns a data frame sorted worst-first, because the interesting row is always
    the covariate the generator handled least well.
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    X_syn = np.asarray(X_syn, dtype=np.float64)

    rows = []
    for j, name in enumerate(feature_names):
        rows.append({
            'covariate': name,
            'real_mean': float(X_real[:, j].mean()),
            'synthetic_mean': float(X_syn[:, j].mean()),
            'smd': standardized_mean_difference(X_real[:, j], X_syn[:, j]),
            'ks': ks_distance(X_real[:, j], X_syn[:, j]),
        })

    report = pd.DataFrame(rows)
    return report.reindex(
        report['smd'].abs().sort_values(ascending=False).index).reset_index(drop=True)


def discriminator_auc(X_real, X_syn, seed=0, n_splits=5, n_estimators=200):
    """
    Train a classifier to tell real rows from synthetic rows. Report its
    cross-validated AUC.

    HOW TO READ THE NUMBER
    ----------------------
      ~0.50  the model cannot do better than a coin flip. The synthetic rows are
             statistically indistinguishable from real ones on these covariates.
             This is the target, and it is the honest operationalisation of the
             word the app used to use for free.
      ~0.65  detectable but subtle differences.
      >0.80  the two are easy to tell apart. Whatever you concluded from this
             synthetic benchmark may not transfer to the real data.

    Note the direction of the goal: this is one of the rare models you WANT to
    perform badly. Participants find that disorienting, which makes it memorable.

    WHY CROSS-VALIDATED
    -------------------
    A random forest with enough trees can separate almost any two finite samples
    if you let it score itself on the rows it trained on. Every prediction here
    comes from a fold that did not see that row, so the AUC reports detectable
    structure rather than memorized rows.

    Returns
    -------
    dict with 'auc', 'n_real', 'n_synthetic', 'interpretation'.
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    X_syn = np.asarray(X_syn, dtype=np.float64)

    X = np.vstack([X_real, X_syn])
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_syn))])

    # Folds cannot exceed the size of the smaller class, and 2 is the minimum that
    # means anything. A tiny upload should degrade to a usable answer, not crash.
    n_splits = int(max(2, min(n_splits, len(X_real), len(X_syn))))

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=int(seed),
                                 n_jobs=1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    auc = float(roc_auc_score(y, proba))

    return {
        'auc': auc,
        'n_real': int(len(X_real)),
        'n_synthetic': int(len(X_syn)),
        'n_splits': n_splits,
        'interpretation': interpret_auc(auc),
    }


def interpret_auc(auc):
    """Plain-language reading of a discriminator AUC, for display in the app."""
    if auc < 0.60:
        return "Indistinguishable - the discriminator is near chance."
    if auc < 0.70:
        return "Mostly realistic - small detectable differences."
    if auc < 0.80:
        return "Detectably synthetic - treat conclusions with caution."
    return "Easily separated - the generator is not reproducing this data."


def realism_report(X_real, X_syn, feature_names, seed=0):
    """
    Bundle the three checks into one result for the UI and the test suite.

    X_real should be the HELD-OUT real covariates, encoded the same way as X_syn.
    """
    return {
        'covariates': covariate_report(X_real, X_syn, feature_names),
        'discriminator': discriminator_auc(X_real, X_syn, seed=seed),
    }
