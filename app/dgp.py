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
        self.raw_mean_ = float(raw_train.mean())
        self.raw_sd_ = raw_sd if raw_sd > 0 else 1.0
        return self

    def standardized(self, X):
        """
        The bare index, scaled to roughly unit standard deviation and centred near
        zero on the training data. This is the reusable building block: f(X) is this
        put on the outcome scale, and the propensity model in assign_treatment() is
        this put on the log-odds scale.
        """
        X = np.asarray(X, dtype=np.float64)
        raw = ((X - self.mean_) / self.sd_).dot(self.weights_)
        return (raw - self.raw_mean_) / self.raw_sd_

    def transform(self, X, sd_y):
        """Return f(X) on the outcome scale, given the outcome's standard deviation."""
        return self.confounding_strength * sd_y * self.standardized(X)

    def weights_table(self):
        """The weights as a data frame, so the app can SHOW what it chose."""
        return pd.DataFrame({
            'covariate': self.feature_names_,
            'weight': self.weights_,
            'train_mean': self.mean_,
            'train_sd': self.sd_,
        })


# ---------------------------------------------------------------------------
# tau(X): the treatment effect, allowed to depend on the covariates.
#
# WHY CONSTANT tau WAS A REAL LIMITATION
# --------------------------------------
# Until now every patient benefited identically. That is comfortable but it makes
# the app's own Step 4 uninformative: with a constant effect, g-computation, IPW,
# AIPW and TMLE all recover the SAME number, so a participant comparing them learns
# only that they agree. The methods cannot be told apart because the problem is not
# hard enough to separate them.
#
# It also leaves the most clinically important question unaskable. "Does this drug
# work?" and "does this drug work FOR THIS PATIENT?" are different questions, and
# only the second one requires tau to vary.
#
# THE PARAMETERIZATION, AND ONE PROPERTY WORTH GUARDING
# -----------------------------------------------------
#     tau(X) = tau0 + kappa * sd_y * m(X)
#
# where m(X) is a standardized covariate index with mean ~0 and sd ~1 on the
# training data. Because m is centred, E[tau(X)] = tau0: turning heterogeneity ON
# DOES NOT MOVE THE AVERAGE TREATMENT EFFECT. The number the user typed into the
# "True ATE" box stays the ATE, and only the spread around it changes.
#
# That property is deliberate. Without it, a participant who increased heterogeneity
# would see the ATE drift and could not tell whether an estimator had broken or the
# target had simply moved.
# ---------------------------------------------------------------------------
class EffectFunction(object):
    """
    tau(X) = tau0 + heterogeneity * sd_y * m(X)

    `heterogeneity` is in outcome standard deviations per standard deviation of the
    modifier index, so heterogeneity = 0 reproduces the constant-effect behaviour
    exactly and nothing downstream changes.

    By default the modifier index m is the SAME index used for prognosis, which
    encodes the common clinical situation where the patients who are sicker are also
    the patients with the most to gain. Passing a different `modifier` lets the
    workshop break that alignment and see what changes.
    """

    def __init__(self, tau0, heterogeneity=0.0, modifier=None):
        self.tau0 = float(tau0)
        self.heterogeneity = float(heterogeneity)
        self.modifier = modifier

    def transform(self, X, sd_y, index):
        m = (self.modifier or index).standardized(X)
        return self.tau0 + self.heterogeneity * sd_y * m

    def is_constant(self):
        return self.heterogeneity == 0.0


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


def _logit(p):
    return np.log(p / (1.0 - p)) if 0.0 < p < 1.0 else 0.0


def assign_treatment(n, p_treat, rho, rng):
    """
    LEARNED mechanism (the Tier 0 behaviour, and still the default).

    P(Z = 1 | U) = logistic( logit(p_treat) + rho * U ),  U ~ N(0, 1)

    Treatment is drawn first and the covariates are drawn conditional on it, so the
    observed confounding is whatever the CVAE learned from the real data. Realistic,
    and completely outside our control.

    rho is the strength of unmeasured confounding. U is never exported when rho = 0
    (it influences nothing). When rho > 0 the app does export it -- not because an
    analyst would ever have it, but so the workshop can demonstrate that adjusting
    for it restores the truth.

    Note the intercept only fixes the treated fraction when rho = 0; for rho > 0 the
    average of a logistic curve is not the logistic of the average, so the realised
    treated fraction drifts slightly. Harmless, but worth noticing out loud.

    Returns
    -------
    Z, U, propensity
    """
    U = rng.standard_normal(n)
    ps = 1.0 / (1.0 + np.exp(-(_logit(p_treat) + rho * U)))
    return rng.binomial(1, ps), U, ps


def assign_treatment_designed(m_ps, p_treat, rho, rng, overlap_strength):
    """
    DESIGNED mechanism: treatment is a function of the covariates we can see.

        P(Z = 1 | X, U) = logistic( logit(p_treat) + lambda * m(X) + rho * U )

    Here `overlap_strength` is lambda: the change in log-odds of treatment per
    standard deviation of the covariate index. It is the positivity dial.

        lambda = 0    every patient has the same treatment probability. This is a
                      randomised trial run on realistic covariates -- a genuinely
                      useful teaching setting, not a degenerate one.
        lambda = 1    mild confounding, comfortable overlap.
        lambda = 3    strong confounding. Propensities pile up near 0 and 1, some
                      patients are almost never treated, and estimators that divide
                      by the propensity start to come apart.

    THE TRADE WE ARE MAKING (worth a minute in the workshop)
    --------------------------------------------------------
    The learned mechanism above draws Z first and X second, so it inherits the real
    data's confounding but cannot vary it. This one draws X first and Z second --
    the correct causal direction -- so confounding and overlap become dials, but the
    arm-specific covariate structure is now OURS rather than the data's.

    You cannot have both from a conditional VAE: p(X | Z) and a controllable e(X)
    pull in opposite directions. Getting realistic covariates AND a designed
    propensity out of the same model is precisely what CausalMix's overlap
    regularizer is for, and is the strongest single argument for adopting it.

    Returns
    -------
    Z, U, propensity
    """
    n = len(m_ps)
    U = rng.standard_normal(n)
    logits = _logit(p_treat) + overlap_strength * m_ps + rho * U
    ps = 1.0 / (1.0 + np.exp(-logits))
    return rng.binomial(1, ps), U, ps


def overlap_diagnostics(ps, Z):
    """
    Positivity summary for a generated cohort.

    `pct_extreme` counts propensities below 0.05 or above 0.95 -- patients for whom
    one arm is nearly never observed, and where any estimator that reweights by 1/e
    is dividing by something close to zero.

    These are deliberately the same diagnostics the survcausal-pilot R study already
    computes (`pct_extreme`, `pct_clipped` in `run_causal()`). There they are
    reported but never varied, so they sit near zero in every design cell. Here they
    move, which is the point.
    """
    ps = np.asarray(ps, dtype=np.float64)
    return {
        'ps_min': float(ps.min()),
        'ps_max': float(ps.max()),
        'pct_extreme': float(np.mean((ps < 0.05) | (ps > 0.95))),
        'treated_frac': float(np.mean(Z)),
    }


def _draw_covariates(model, index, original_df, feature_names, Z_cond, rng, seed,
                     outcome_col, treatment_col, sample_categories):
    """Sample encoded covariates conditional on an arm vector, decode, re-encode."""
    X_raw = model.sample(Z_cond, seed=int(seed))
    decoded = decode_samples(X_raw, feature_names, original_df,
                             treatment_col=treatment_col, outcome_col=outcome_col,
                             rng=rng, sample_categories=sample_categories)
    return decoded, align_encoding(decoded, feature_names)


def potential_outcomes(fx, U, tau, rho, sd_y, rng, gamma=0.5):
    """
    Y(0) = f(X) + gamma * rho * U * sd_y + noise,   Y(1) = Y(0) + tau

    Two paths carry bias into a naive treated-minus-control comparison:

      through X : Z shifts the covariate distribution, and f(X) shifts Y(0).
                  An analyst CAN close this path, because X is exported.
      through U : rho moves both the treatment odds and Y(0). An analyst CANNOT
                  close this path, because U is not something you would have.

    `tau` may be a scalar (every patient benefits identically) or a per-patient
    array from EffectFunction. When it varies, y1 - y0 varies with it, and the
    average treatment effect becomes the MEAN of those individual effects rather
    than a single number chosen in advance.

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
                    sample_categories=True, mechanism='learned',
                    overlap_strength=1.0, effect=None):
    """
    Run the whole generation pipeline once, reproducibly.

    Steps, in order:
      1. obtain covariates X and treatment Z, by one of two mechanisms (below)
      2. re-encode X so exported == causal          (see the file header)
      3. compute f(X) and tau(X), then the potential outcomes
      4. reveal only the outcome for the arm each row was actually assigned

    Step 4 is the fundamental problem of causal inference made literal: we compute
    both potential outcomes, then throw one away. The analyst gets y_obs; we keep
    y0, y1 and the per-patient tau in the export so the workshop can check its own
    answers.

    MECHANISM
    ---------
    'learned'  -- draw Z, then X | Z from the CVAE. Confounding is inherited from
                  the real data: realistic, not adjustable. (Tier 0 behaviour.)
    'designed' -- draw X first, then Z from a propensity model in X whose steepness
                  is `overlap_strength`. Confounding and positivity become dials, at
                  the cost of arm-specific covariate structure being ours rather
                  than the data's. See assign_treatment_designed().

    `effect` is an EffectFunction; when omitted, the scalar `tau` is used and the
    effect is constant, exactly as before.

    `seed` controls every stochastic step, so the same seed reproduces the cohort.
    This is a SEPARATE seed from the one used to train the model: keeping the two
    apart means you can redraw a cohort without retraining, or retrain without
    moving the cohort, and know which change caused what. (The same split is used in
    the survcausal-pilot R study, which seeds the cohort draw and the estimator fits
    from disjoint streams for exactly this reason.)
    """
    rng = np.random.default_rng(int(seed))

    if mechanism == 'designed':
        # Covariates first. They are drawn at a coin-flip arm purely to give the
        # conditional decoder something to condition on; the arm that ends up in the
        # data is decided afterwards, by the propensity model.
        Z_seed = rng.binomial(1, p_treat, n)
        decoded, X_final = _draw_covariates(
            model, index, original_df, feature_names, Z_seed, rng, seed,
            outcome_col, treatment_col, sample_categories)
        m_ps = index.standardized(X_final)
        Z, U, ps = assign_treatment_designed(m_ps, p_treat, rho, rng, overlap_strength)
    else:
        Z, U, ps = assign_treatment(n, p_treat, rho, rng)
        decoded, X_final = _draw_covariates(
            model, index, original_df, feature_names, Z, rng, seed,
            outcome_col, treatment_col, sample_categories)

    fx = index.transform(X_final, sd_y)
    tau_i = effect.transform(X_final, sd_y, index) if effect is not None else tau

    y0, y1 = potential_outcomes(fx, U, tau_i, rho, sd_y, rng, gamma=gamma)
    y_obs = np.where(Z == 1, y1, y0)

    out = pd.DataFrame({outcome_col: y_obs, treatment_col: Z})
    # Every covariate is exported, in its original schema. The previous version
    # capped this at the first ten encoded columns, so a wide upload came back
    # silently truncated -- and a truncated export cannot support adjustment.
    for col in decoded.columns:
        out[col] = decoded[col].values

    out['y0'] = y0
    out['y1'] = y1
    # Per-patient truth. When the effect is constant this column is constant, so
    # nothing downstream has to know which case it is in.
    out['tau_true'] = tau_i
    if rho > 0:
        out['U'] = U

    return {
        'data': out,
        'fx': fx,
        'U': U,
        'Z': Z,
        'propensity': ps,
        'tau_true': np.asarray(tau_i) * np.ones(n),
        'decoded': decoded,
        'X_encoded': X_final,
        'overlap': overlap_diagnostics(ps, Z),
        # The ATE is now the AVERAGE of the individual effects. With a constant tau
        # that is tau exactly; with tau(X) it is a computed quantity, and it is the
        # number an ATE estimator should be scored against.
        'true_ate': float(np.mean(y1 - y0)),
    }
