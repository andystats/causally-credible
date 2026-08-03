"""
The data-generating process  --  the "truth" half of Credence.

WHERE THIS SITS
---------------
cvae_model.py learns p(X | Z) from real data and can draw new covariate rows:
that is REALISM, and it contains no causal information at all. This file adds
TRUTH: it decides how treatment is assigned, what the outcome depends on, and --
crucially -- what the true treatment effect tau actually is. Because we chose
tau, we can score any estimator against it. That is the whole trick.

Everything here is a plain function of its arguments plus an explicit random
number generator. Nothing imports Shiny. That is deliberate: it means the entire
pipeline can be run headlessly from a test script (see tests/test_tier0.py), and
a participant can read the data-generating process end to end without wading
through reactive UI plumbing.

THE GENERATIVE ORDER, AND WHY IT LOOKS BACKWARDS
------------------------------------------------
In the real world X causes Z: sicker patients get treated differently. Here we
generate Z first, then draw X conditional on Z. That looks wrong, and it is worth
a minute of workshop time.

It is not wrong, because p(X, Z) = p(Z) p(X | Z) is a perfectly valid
factorization of the same joint distribution -- we are sampling from the joint,
not asserting a causal direction. Credence itself works this way, because a
conditional VAE finds p(X | Z) much easier to learn than the reverse.

What it does mean is that the STRENGTH of observed confounding is a by-product of
whatever the real data happened to contain, not a dial we can turn. Turning it
into a dial is a Tier 1 job, and so is the overlap knob. Tier 0 only makes the
existing mechanism honest and legible.

WHAT COUNTS AS "OBSERVED" -- A PROPERTY THIS FILE MUST PRESERVE
---------------------------------------------------------------
The exported covariates must FULLY determine the potential outcomes (apart from
U and the noise term). If the outcome secretly depended on something we did not
export, then no analyst could ever adjust for it, every estimator would look
biased, and the workshop would teach a false lesson about the methods rather than
a true one about the data.

This is why generate_cohort() below decodes the covariates to their original
schema and then RE-ENCODES them before computing f(X). The alternative -- using
the decoder's raw continuous scores -- would leave a categorical variable's
exported value only loosely related to the value that actually drove the outcome.
"""

import numpy as np
import pandas as pd

from cvae_model import decode_samples


# ---------------------------------------------------------------------------
# f(X): the prognostic index.
#
# THE BUG THIS REPLACES (worth showing participants side by side)
# ---------------------------------------------------------------
# The previous implementation was one line:
#
#     numeric_features = [i for i, name in enumerate(feature_names)
#                         if not any(cat in name for cat in ['_'])]
#     hX = X_sim[:, numeric_features].mean(axis=1)
#
# It has two independent defects.
#
# 1. SELECTION BY NAME. The filter keeps columns whose NAME CONTAINS NO
#    UNDERSCORE. The intent was to skip one-hot dummies, which encode_data names
#    "<col>_<level>". But it also silently drops any real covariate a user happens
#    to have named in snake_case -- prior_vaccine, days_since_index, bmi_baseline.
#    Those covariates then influence nothing, while still being exported, so the
#    participant sees a variable that looks like a confounder and is not one.
#    It survived because both built-in datasets happen to use names with no
#    underscores (age, education, priorPneumonia). The first person to upload
#    their own file would have hit it.
#
# 2. AVERAGING ON THE RAW SCALE. An unweighted mean of columns measured in
#    different units is dominated by whichever has the largest numbers. In the
#    LaLonde data, earnings in dollars (thousands) swamp age in years (tens), so
#    the confounding structure was effectively "earnings, plus rounding error" --
#    an accident of measurement units, not a modelling choice.
#
# The replacement standardizes every column first (so units cannot decide
# importance), applies WEIGHTS THE ANALYST CHOSE (so importance is a stated
# decision, visible in the UI), and rescales to a stated magnitude (so the
# strength of confounding is interpretable in outcome standard deviations).
# ---------------------------------------------------------------------------
class PrognosticIndex(object):
    """
    f(X) = confounding_strength * sd_y * ( sum_j w_j * z_j ) / s

    where z_j is column j standardized using the TRAINING data's mean and sd, and
    s is the standard deviation of the weighted sum on the training data.

    Fitting on the training data and reusing those constants (rather than
    recomputing them on each synthetic batch) means f(X) is the same function
    every time it is called. A function that silently redefined itself per batch
    would make two "identical" runs incomparable.

    Follows the sklearn fit/transform convention already used by the scaler in
    cvae_model.py.
    """

    def __init__(self, weights=None, confounding_strength=1.0):
        self.weights = weights
        self.confounding_strength = confounding_strength

    def fit(self, X_train, feature_names):
        X_train = np.asarray(X_train, dtype=np.float64)
        self.feature_names_ = list(feature_names)
        p = X_train.shape[1]

        self.mean_ = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        # A constant column has sd 0 and would divide to infinity. Setting its sd
        # to 1 makes its standardized value identically 0, i.e. it contributes
        # nothing -- which is the right answer for a covariate that never varies.
        self.sd_ = np.where(sd > 0, sd, 1.0)

        if self.weights is None:
            # Equal weights: every covariate matters the same amount. This is the
            # honest default -- it says "the analyst expressed no preference" --
            # and unlike the old raw-scale mean it actually delivers that, because
            # the columns have been standardized first.
            self.weights_ = np.ones(p) / p
        else:
            self.weights_ = np.asarray(self.weights, dtype=np.float64)
            if self.weights_.shape[0] != p:
                raise ValueError(
                    "weights has length {} but there are {} encoded covariates".format(
                        self.weights_.shape[0], p))

        raw_train = ((X_train - self.mean_) / self.sd_).dot(self.weights_)
        raw_sd = raw_train.std()
        self.raw_sd_ = raw_sd if raw_sd > 0 else 1.0
        return self

    def transform(self, X, sd_y):
        """Return f(X) on the outcome scale, given the outcome's standard deviation."""
        X = np.asarray(X, dtype=np.float64)
        raw = ((X - self.mean_) / self.sd_).dot(self.weights_)
        return self.confounding_strength * sd_y * raw / self.raw_sd_

    def weights_table(self):
        """The weights as a data frame, so the app can SHOW what it chose."""
        return pd.DataFrame({
            'covariate': self.feature_names_,
            'weight': self.weights_,
            'train_mean': self.mean_,
            'train_sd': self.sd_,
        })


def align_encoding(decoded_df, feature_names):
    """
    Re-encode a decoded covariate frame back onto the exact training columns.

    After decode_samples() a categorical column holds real category labels again.
    One-hot encoding it reproduces proper 0/1 dummies; reindexing onto
    feature_names puts the columns back in the training order and fills in any
    level that this particular synthetic batch happened not to draw.

    This is the step that guarantees the property described in the file header:
    what we export is exactly what determines the outcome.
    """
    re_encoded = pd.get_dummies(decoded_df, drop_first=False)
    re_encoded = re_encoded.reindex(columns=feature_names, fill_value=0)
    return re_encoded.values.astype(np.float64)


def assign_treatment(n, p_treat, rho, rng):
    """
    Draw treatment, confounded by an UNOBSERVED U.

    P(Z = 1 | U) = logistic( logit(p_treat) + rho * U ),  U ~ N(0, 1)

    rho is the strength of unmeasured confounding, and U is deliberately never
    exported when rho = 0 (it influences nothing, so showing it would only
    confuse). When rho > 0 the app does export it -- not because an analyst would
    ever have it, but so the workshop can demonstrate that adjusting for it
    restores the truth, which is the cleanest possible illustration of what
    "unmeasured" costs you.

    Note the intercept only fixes the treated fraction when rho = 0; for rho > 0
    the average of a logistic curve is not the logistic of the average, so the
    realised treated fraction drifts slightly from p_treat. Harmless here, but the
    kind of thing worth noticing out loud.

    Returns
    -------
    Z : np.ndarray of 0/1
    U : np.ndarray, the unobserved confounder
    """
    U = rng.standard_normal(n)
    if 0.0 < p_treat < 1.0:
        logit_p = np.log(p_treat / (1.0 - p_treat))
    else:
        logit_p = 0.0
    ps = 1.0 / (1.0 + np.exp(-(logit_p + rho * U)))
    Z = rng.binomial(1, ps)
    return Z, U


def potential_outcomes(fx, U, tau, rho, sd_y, rng, gamma=0.5):
    """
    Y(0) = f(X) + gamma * rho * U * sd_y + noise,   Y(1) = Y(0) + tau

    Two paths carry bias into a naive treated-minus-control comparison:

      through X : Z shifts the covariate distribution, and f(X) shifts Y(0).
                  An analyst CAN close this path, because X is exported.
      through U : rho moves both the treatment odds and Y(0). An analyst CANNOT
                  close this path, because U is not something you would have.

    Setting tau constant means every patient benefits identically. That is a real
    limitation -- it is why every consistent estimator returns the same answer
    here and the methods cannot be told apart -- and it is exactly what Tier 1
    addresses by making tau a function of X.

    Returns
    -------
    y0, y1 : np.ndarray
    """
    n = len(fx)
    noise = rng.normal(0.0, sd_y, n)
    y0 = fx + gamma * rho * U * sd_y + noise
    y1 = y0 + tau
    return y0, y1


def generate_cohort(model, index, original_df, feature_names, n, tau, rho, sd_y,
                    p_treat, outcome_col, treatment_col, seed, gamma=0.5,
                    sample_categories=True):
    """
    Run the whole generation pipeline once, reproducibly.

    Steps, in order:
      1. draw U and assign treatment Z            (unmeasured confounding enters)
      2. decode covariates X from the CVAE given Z (observed confounding enters)
      3. re-encode X so exported == causal        (see the file header)
      4. compute f(X), then the potential outcomes
      5. reveal only the outcome for the arm each row was actually assigned

    Step 5 is the fundamental problem of causal inference made literal: we compute
    both potential outcomes, then throw one away. The analyst gets y_obs; we keep
    y0 and y1 in the export so the workshop can check its own answers.

    `seed` controls every stochastic step, so the same seed reproduces the cohort
    exactly. Note that this is a SEPARATE seed from the one used to train the
    model: keeping the two apart means you can redraw a cohort without retraining,
    or retrain without moving the cohort, and know which change caused what. (The
    same split is used in the survcausal-pilot R study, which seeds the cohort draw
    and the estimator fits from disjoint streams for exactly this reason.)

    Returns
    -------
    dict with 'data' (the exported frame), 'fx', 'U', 'Z', and 'true_ate'.
    """
    rng = np.random.default_rng(int(seed))

    Z, U = assign_treatment(n, p_treat, rho, rng)

    X_encoded_raw = model.sample(Z, seed=int(seed))
    decoded = decode_samples(X_encoded_raw, feature_names, original_df,
                             treatment_col=treatment_col, outcome_col=outcome_col,
                             rng=rng, sample_categories=sample_categories)
    X_final = align_encoding(decoded, feature_names)

    fx = index.transform(X_final, sd_y)
    y0, y1 = potential_outcomes(fx, U, tau, rho, sd_y, rng, gamma=gamma)
    y_obs = np.where(Z == 1, y1, y0)

    out = pd.DataFrame({outcome_col: y_obs, treatment_col: Z})
    # Every covariate is exported, in its original schema. The previous version
    # capped this at the first ten encoded columns, so a wide upload came back
    # silently truncated -- and a truncated export cannot support adjustment.
    for col in decoded.columns:
        out[col] = decoded[col].values

    out['y0'] = y0
    out['y1'] = y1
    out['tau_true'] = tau
    if rho > 0:
        out['U'] = U

    return {
        'data': out,
        'fx': fx,
        'U': U,
        'Z': Z,
        'decoded': decoded,
        'X_encoded': X_final,
        # With a constant tau the sample ATE is tau exactly. Returned as its own
        # field anyway, because Tier 1's tau(X) will make this a computed average
        # over the cohort rather than a constant, and callers should not have to
        # change when that happens.
        'true_ate': float(np.mean(y1 - y0)),
    }
