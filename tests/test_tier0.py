"""
Verification gate for the Tier 0 fixes.

Run it directly -- no pytest required, so a workshop participant can execute it
on a laptop with nothing but the app's own dependencies:

    python3 tests/test_tier0.py

Each test corresponds to a numbered item in the plan's verification gate. The
tests are written to be READ as much as run: several of them re-implement the old
buggy behaviour alongside the new one and assert that the two disagree, so the
file doubles as an executable record of what was wrong and why it mattered.
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from cvae_model import ConditionalVAE, encode_data, decode_samples, set_seed
from dgp import PrognosticIndex, generate_cohort, align_encoding
from examples import load_example
from realism import realism_report, standardized_mean_difference, ks_distance


# ---------------------------------------------------------------------------
# Shared helper: the whole pipeline, exactly as the app runs it.
# ---------------------------------------------------------------------------
def fit_pipeline(df, outcome_col='outcome', treatment_col='treatment',
                 epochs=60, batch_size=128, train_seed=1, holdout_frac=0.25):
    """Split, train, and return everything the generation step needs."""
    rng = np.random.default_rng(train_seed)
    n = len(df)
    idx = rng.permutation(n)
    n_hold = max(20, int(round(holdout_frac * n)))
    hold_idx, train_idx = idx[:n_hold], idx[n_hold:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_hold = df.iloc[hold_idx].reset_index(drop=True)

    X_train, Z_train, feature_names = encode_data(
        df_train, outcome_col=outcome_col, treatment_col=treatment_col)
    X_hold, Z_hold, _ = encode_data(
        df_hold, outcome_col=outcome_col, treatment_col=treatment_col)

    set_seed(train_seed)
    model = ConditionalVAE(input_dim=X_train.shape[1], latent_dim=16)
    model.fit(X_train, Z_train, epochs=epochs, batch_size=batch_size,
              seed=train_seed, verbose=False)

    index = PrognosticIndex().fit(X_train, feature_names)

    return {
        'model': model, 'index': index, 'feature_names': feature_names,
        'df_train': df_train, 'df_hold': df_hold,
        'X_train': X_train, 'X_hold': X_hold, 'Z_hold': Z_hold,
        'sd_y': float(df[outcome_col].std()),
        'p_treat': float(df[treatment_col].mean()),
    }


def generate(pipe, df, outcome_col='outcome', treatment_col='treatment',
             n=1500, tau=-5.0, rho=0.0, seed=7):
    return generate_cohort(
        model=pipe['model'], index=pipe['index'], original_df=df,
        feature_names=pipe['feature_names'], n=n, tau=tau, rho=rho,
        sd_y=pipe['sd_y'], p_treat=pipe['p_treat'],
        outcome_col=outcome_col, treatment_col=treatment_col, seed=seed)


# ---------------------------------------------------------------------------
# Gate item 2 -- both built-in datasets run end to end.
# ---------------------------------------------------------------------------
def test_both_examples_run_end_to_end():
    for name in ('lalonde', 'pneumonia'):
        df, note = load_example(name)
        assert len(df) > 0, "{}: empty dataset".format(name)

        pipe = fit_pipeline(df, epochs=20)
        out = generate(pipe, df, n=500)
        data = out['data']

        assert len(data) == 500
        assert 'outcome' in data.columns and 'treatment' in data.columns
        assert np.isfinite(data['outcome'].values).all(), "{}: non-finite outcomes".format(name)

        # Every original covariate must survive to the export (defect 7: the old
        # code capped this at the first ten encoded columns).
        covariates = [c for c in df.columns if c not in ('outcome', 'treatment')]
        for col in covariates:
            assert col in data.columns, "{}: covariate {} missing from export".format(name, col)

        print("    {}: {} rows, {} columns exported. {}".format(
            name, len(data), data.shape[1], note.split('.')[0]))


# ---------------------------------------------------------------------------
# Gate item 3 -- the underscore regression (defect 4).
#
# Two datasets identical in every way except that one uses snake_case covariate
# names. f(X) must not care. The old selector cared a great deal.
# ---------------------------------------------------------------------------
def legacy_fx(X, feature_names):
    """The old f(X), reproduced verbatim so the test can show what it did."""
    numeric_features = [i for i, name in enumerate(feature_names)
                        if not any(cat in name for cat in ['_'])]
    if len(numeric_features) == 0:
        return np.zeros(X.shape[0])
    return X[:, numeric_features].mean(axis=1)


def test_underscore_names_do_not_change_fx():
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({
        'treatment': rng.binomial(1, 0.5, n),
        'outcome': rng.normal(0, 1, n),
        'age': rng.normal(60, 10, n),
        'prior_vaccine': rng.binomial(1, 0.4, n),      # snake_case
        'days_since_index': rng.normal(200, 50, n),    # snake_case
    })
    renamed = base.rename(columns={'prior_vaccine': 'priorvaccine',
                                   'days_since_index': 'dayssinceindex'})

    X_a, _, names_a = encode_data(base, outcome_col='outcome', treatment_col='treatment')
    X_b, _, names_b = encode_data(renamed, outcome_col='outcome', treatment_col='treatment')

    fx_a = PrognosticIndex().fit(X_a, names_a).transform(X_a, sd_y=1.0)
    fx_b = PrognosticIndex().fit(X_b, names_b).transform(X_b, sd_y=1.0)

    assert np.allclose(fx_a, fx_b), "f(X) changed when covariates were renamed"

    # And confirm the old code genuinely had the bug -- otherwise this test is
    # guarding nothing. With underscores it kept 1 of 3 covariates; without, all 3.
    kept_a = [nm for nm in names_a if '_' not in nm]
    kept_b = [nm for nm in names_b if '_' not in nm]
    assert len(kept_a) == 1 and len(kept_b) == 3, \
        "expected the legacy filter to drop the snake_case covariates"
    assert not np.allclose(legacy_fx(X_a, names_a), legacy_fx(X_b, names_b)), \
        "legacy f(X) was expected to differ under renaming"

    print("    new f(X) identical under renaming; legacy f(X) kept "
          "{}/{} vs {}/{} covariates".format(len(kept_a), len(names_a),
                                             len(kept_b), len(names_b)))


def test_fx_is_not_dominated_by_units():
    """
    Defect 5: an unweighted mean on the raw scale is decided by measurement units.

    Same data, one covariate rescaled from dollars-per-year to dollars. The new
    standardized index is unchanged; the legacy raw-scale mean is not.
    """
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        'treatment': rng.binomial(1, 0.5, n),
        'outcome': rng.normal(0, 1, n),
        'age': rng.normal(60, 10, n),
        'earnings': rng.normal(20000, 5000, n),
    })
    X, _, names = encode_data(df, outcome_col='outcome', treatment_col='treatment')

    X_scaled = X.copy()
    X_scaled[:, names.index('earnings')] *= 1000.0   # same variable, different unit

    fx = PrognosticIndex().fit(X, names).transform(X, sd_y=1.0)
    fx_scaled = PrognosticIndex().fit(X_scaled, names).transform(X_scaled, sd_y=1.0)
    assert np.allclose(fx, fx_scaled), "standardized f(X) should be unit-invariant"

    # The legacy index correlates ~1.0 with earnings alone: age contributed nothing.
    legacy = legacy_fx(X, names)
    r_earnings = np.corrcoef(legacy, X[:, names.index('earnings')])[0, 1]
    assert r_earnings > 0.99, "expected the legacy mean to be dominated by earnings"

    print("    standardized f(X) unit-invariant; legacy f(X) correlated "
          "{:.4f} with earnings alone".format(r_earnings))


# ---------------------------------------------------------------------------
# Gate item 4 -- reproducibility (defect 6).
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_the_cohort():
    df, _ = load_example('pneumonia')
    pipe = fit_pipeline(df, epochs=20, train_seed=3)

    a = generate(pipe, df, n=400, seed=42)['data']
    b = generate(pipe, df, n=400, seed=42)['data']
    c = generate(pipe, df, n=400, seed=43)['data']

    pd.testing.assert_frame_equal(a, b)
    assert not a['outcome'].equals(c['outcome']), "different seeds gave identical data"

    # Training is separately reproducible, and separately seeded -- so a cohort can
    # be redrawn without retraining, and vice versa.
    pipe_again = fit_pipeline(df, epochs=20, train_seed=3)
    assert np.allclose(pipe['model'].history_[-1]['neg_elbo'],
                       pipe_again['model'].history_[-1]['neg_elbo']), \
        "training was not reproducible at a fixed seed"

    print("    seed 42 reproduced exactly; seed 43 differed; training reproducible")


# ---------------------------------------------------------------------------
# Gate item 1 (partial) -- the minibatch fix (defects 1 and 2).
# ---------------------------------------------------------------------------
def test_minibatching_takes_many_steps_per_epoch():
    df, _ = load_example('pneumonia')
    X, Z, names = encode_data(df, outcome_col='outcome', treatment_col='treatment')

    set_seed(0)
    full = ConditionalVAE(input_dim=X.shape[1]).fit(
        X, Z, epochs=5, batch_size=None, seed=0, verbose=False)
    set_seed(0)
    mini = ConditionalVAE(input_dim=X.shape[1]).fit(
        X, Z, epochs=5, batch_size=128, seed=0, verbose=False)

    assert full.history_[-1]['steps'] == 1, "full-batch should take 1 step per epoch"
    assert mini.history_[-1]['steps'] == int(np.ceil(len(df) / 128.0))

    # The old default (50 epochs, full batch) is 50 gradient steps in total. The
    # new default reaches that in a single epoch on this dataset.
    assert mini.history_[-1]['neg_elbo'] < full.history_[-1]['neg_elbo'], \
        "minibatched training should reach a lower negative ELBO in the same epochs"

    print("    5 epochs: full-batch {} steps, neg-ELBO {:.2f} | minibatch {} steps, "
          "neg-ELBO {:.2f}".format(full.history_[-1]['steps'], full.history_[-1]['neg_elbo'],
                                   mini.history_[-1]['steps'], mini.history_[-1]['neg_elbo']))


# ---------------------------------------------------------------------------
# Gate item 1 (partial) -- decode_samples is wired in, and samples categories
# rather than taking the argmax (defect 3).
# ---------------------------------------------------------------------------
def test_categorical_round_trip_preserves_rare_levels():
    rng = np.random.default_rng(2)
    n = 600
    region = rng.choice(['north', 'south', 'rare'], n, p=[0.5, 0.45, 0.05])
    df = pd.DataFrame({
        'treatment': rng.binomial(1, 0.5, n),
        'outcome': rng.normal(0, 1, n),
        'age': rng.normal(60, 10, n),
        'region': region,
    })

    X, Z, names = encode_data(df, outcome_col='outcome', treatment_col='treatment')
    assert 'region_rare' in names, "one-hot encoding should expand region"

    # Decoder scores that imply P = (0.45, 0.40, 0.15) for every row.
    scores = np.tile(np.array([np.log(0.45), np.log(0.40), np.log(0.15)]), (n, 1))
    X_fake = np.zeros((n, len(names)))
    for k, level in enumerate(['north', 'south', 'rare']):
        X_fake[:, names.index('region_' + level)] = scores[:, k]
    X_fake[:, names.index('age')] = rng.normal(60, 10, n)

    sampled = decode_samples(X_fake, names, df, treatment_col='treatment',
                             outcome_col='outcome', rng=np.random.default_rng(0),
                             sample_categories=True)
    argmaxed = decode_samples(X_fake, names, df, treatment_col='treatment',
                              outcome_col='outcome', rng=np.random.default_rng(0),
                              sample_categories=False)

    assert set(sampled.columns) == {'age', 'region'}
    share_rare = (sampled['region'] == 'rare').mean()
    assert 0.10 < share_rare < 0.20, \
        "sampling should reproduce the 15% rare share, got {:.3f}".format(share_rare)
    assert (argmaxed['region'] == 'north').all(), \
        "argmax should collapse every row onto the modal category"

    # The re-encoding used by the DGP must give clean 0/1 dummies back.
    re_encoded = align_encoding(sampled, names)
    dummy_cols = [names.index('region_' + lv) for lv in ['north', 'south', 'rare']]
    assert set(np.unique(re_encoded[:, dummy_cols])) <= {0.0, 1.0}
    assert np.allclose(re_encoded[:, dummy_cols].sum(axis=1), 1.0)

    print("    sampling kept the rare level at {:.1%}; argmax collapsed it to 0%".format(
        share_rare))


# ---------------------------------------------------------------------------
# Gate item 5 -- the realism gate.
#
# The plan asks for both numbers, because the IMPROVEMENT is the teaching artifact.
# ---------------------------------------------------------------------------
def test_realism_improves_with_proper_training():
    df, _ = load_example('pneumonia')

    undertrained = fit_pipeline(df, epochs=1, batch_size=None, train_seed=5)
    trained = fit_pipeline(df, epochs=150, batch_size=128, train_seed=5)

    results = {}
    for label, pipe in (('undertrained', undertrained), ('trained', trained)):
        syn = generate(pipe, df, n=len(pipe['X_hold']), seed=11)
        rep = realism_report(pipe['X_hold'], syn['X_encoded'],
                             pipe['feature_names'], seed=0)
        results[label] = rep['discriminator']['auc']

    print("    discriminator AUC: undertrained {:.3f} -> trained {:.3f}".format(
        results['undertrained'], results['trained']))

    assert results['trained'] < results['undertrained'], \
        ("training must make the synthetic data harder to detect "
         "(undertrained {:.3f}, trained {:.3f})".format(
             results['undertrained'], results['trained']))


def test_realism_metrics_behave_on_known_input():
    """Sanity-check the metrics themselves against cases with known answers."""
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 2000)
    shifted = rng.normal(2, 1, 2000)

    assert abs(standardized_mean_difference(a, same)) < 0.15
    assert standardized_mean_difference(a, shifted) < -1.5
    assert ks_distance(a, same) < 0.10
    assert ks_distance(a, shifted) > 0.50
    # A sample against itself is a perfect match by construction.
    assert ks_distance(a, a) == 0.0
    print("    SMD and KS behave as expected on matched and shifted samples")


# ---------------------------------------------------------------------------
# Gate item 6 -- the truth gate.
#
# This is the most important test in the file. With rho = 0 there is no unmeasured
# confounding, so the ONLY path from treatment to outcome other than the effect
# itself runs through the exported covariates. A correctly specified adjustment
# must therefore recover tau. If it cannot, the generator is broken and every
# downstream conclusion the workshop draws is worthless.
# ---------------------------------------------------------------------------
def g_computation(data, outcome_col, treatment_col, covariates):
    """
    Ordinary least squares of Y on treatment plus covariates; return the
    coefficient on treatment.

    With a constant tau and a linear f(X) this specification is exactly correct,
    which is what makes it a usable oracle for testing the generator. It is NOT a
    recommendation to fit OLS to real data.
    """
    X = pd.get_dummies(data[list(covariates)], drop_first=True).values.astype(np.float64)
    Z = data[treatment_col].values.astype(np.float64).reshape(-1, 1)
    design = np.hstack([np.ones((len(data), 1)), Z, X])
    y = data[outcome_col].values.astype(np.float64)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def test_truth_gate_gcomputation_recovers_tau():
    df, _ = load_example('pneumonia')
    pipe = fit_pipeline(df, epochs=80, train_seed=9)

    tau = -5.0
    out = generate(pipe, df, n=4000, tau=tau, rho=0.0, seed=21)
    data = out['data']
    covariates = [c for c in df.columns if c not in ('outcome', 'treatment')]

    naive = (data.loc[data['treatment'] == 1, 'outcome'].mean()
             - data.loc[data['treatment'] == 0, 'outcome'].mean())
    adjusted = g_computation(data, 'outcome', 'treatment', covariates)

    # Monte Carlo error on the adjusted estimate is roughly 2 * sd_y / sqrt(n).
    se = 2.0 * pipe['sd_y'] / np.sqrt(len(data))
    tol = 4.0 * se

    print("    true tau {:.3f} | naive {:.3f} (bias {:+.3f}) | "
          "g-computation {:.3f} (bias {:+.3f}, tol +/-{:.3f})".format(
              tau, naive, naive - tau, adjusted, adjusted - tau, tol))

    assert abs(adjusted - tau) < tol, \
        ("g-computation failed to recover tau at rho=0: got {:.4f}, expected {:.4f} "
         "+/- {:.4f}".format(adjusted, tau, tol))

    # The exported data must actually determine the outcome. If align_encoding()
    # were skipped, the covariates would only loosely track what drove f(X) and
    # this residual check would blow up.
    resid_sd = np.std(data['y0'].values - out['fx'])
    assert resid_sd < 1.5 * pipe['sd_y'], \
        "y0 is not explained by the exported covariates plus noise"


def test_unmeasured_confounding_breaks_adjustment():
    """
    The complement of the truth gate: with rho > 0, adjusting for the exported
    covariates is no longer enough, because U is not among them. This is the
    lesson the app exists to teach, so it should be a test and not just a claim.
    """
    df, _ = load_example('pneumonia')
    pipe = fit_pipeline(df, epochs=80, train_seed=9)

    tau = -5.0
    out = generate(pipe, df, n=4000, tau=tau, rho=0.8, seed=22)
    data = out['data']
    covariates = [c for c in df.columns if c not in ('outcome', 'treatment')]

    adjusted = g_computation(data, 'outcome', 'treatment', covariates)
    assert 'U' in data.columns, "U should be exported when rho > 0"
    with_u = g_computation(data, 'outcome', 'treatment', covariates + ['U'])

    print("    rho=0.8: adjusted-for-X {:.3f} (bias {:+.3f}) | "
          "adjusted-for-X-and-U {:.3f} (bias {:+.3f})".format(
              adjusted, adjusted - tau, with_u, with_u - tau))

    assert abs(with_u - tau) < abs(adjusted - tau), \
        "adjusting for the unobserved confounder should reduce bias"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = [
    test_both_examples_run_end_to_end,
    test_underscore_names_do_not_change_fx,
    test_fx_is_not_dominated_by_units,
    test_same_seed_reproduces_the_cohort,
    test_minibatching_takes_many_steps_per_epoch,
    test_categorical_round_trip_preserves_rare_levels,
    test_realism_metrics_behave_on_known_input,
    test_realism_improves_with_proper_training,
    test_truth_gate_gcomputation_recovers_tau,
    test_unmeasured_confounding_breaks_adjustment,
]


def main():
    failures = []
    for test in TESTS:
        print("[ RUN  ] {}".format(test.__name__))
        try:
            test()
            print("[  OK  ] {}\n".format(test.__name__))
        except Exception:
            failures.append(test.__name__)
            print("[ FAIL ] {}".format(test.__name__))
            traceback.print_exc()
            print("")

    print("=" * 70)
    if failures:
        print("{} of {} tests FAILED: {}".format(len(failures), len(TESTS),
                                                 ", ".join(failures)))
        return 1
    print("all {} tests passed".format(len(TESTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
