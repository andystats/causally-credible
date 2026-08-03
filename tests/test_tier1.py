"""
Verification gate for the Tier 1 capability: tau(X), the overlap dial, and the
factorial scenario grid.

    python3 tests/test_tier1.py

Same conventions as test_tier0.py: plain asserts, no pytest, and several tests are
written to be read -- they assert the PROPERTY we designed for, so the design intent
survives in executable form.
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from cvae_model import ConditionalVAE, encode_data, set_seed
from dgp import (PrognosticIndex, EffectFunction, generate_cohort,
                 assign_treatment_designed, overlap_diagnostics)
from estimators import naive_ate, g_computation, cate_predictions
from examples import load_example
from scenarios import run_scenario_grid, summarize_grid

COVARIATES = ['age', 'priorPneumonia', 'priorVaccine']


def build_pipeline(epochs=80, seed=9):
    """Train once; every test below reuses it."""
    df, _ = load_example('pneumonia')
    X, Z, names = encode_data(df, outcome_col='outcome', treatment_col='treatment')
    set_seed(seed)
    model = ConditionalVAE(input_dim=X.shape[1]).fit(
        X, Z, epochs=epochs, batch_size=128, seed=seed, verbose=False)
    index = PrognosticIndex().fit(X, names)
    return {
        'df': df, 'model': model, 'index': index, 'feature_names': names,
        'sd_y': float(df['outcome'].std()), 'p_treat': float(df['treatment'].mean()),
    }


def generate(pipe, **kwargs):
    params = dict(
        model=pipe['model'], index=pipe['index'], original_df=pipe['df'],
        feature_names=pipe['feature_names'], n=4000, tau=-5.0, rho=0.0,
        sd_y=pipe['sd_y'], p_treat=pipe['p_treat'],
        outcome_col='outcome', treatment_col='treatment', seed=31)
    params.update(kwargs)
    return generate_cohort(**params)


PIPE = None


def pipe():
    global PIPE
    if PIPE is None:
        PIPE = build_pipeline()
    return PIPE


# ---------------------------------------------------------------------------
# tau(X)
# ---------------------------------------------------------------------------
def test_heterogeneity_does_not_move_the_average_effect():
    """
    The property the parameterization was designed around: turning heterogeneity on
    changes the SPREAD of the effect, not its average.

    Without this, a participant raising heterogeneity would watch the ATE drift and
    be unable to tell whether an estimator had broken or the target had moved.
    """
    p = pipe()
    flat = generate(p, effect=EffectFunction(tau0=-5.0, heterogeneity=0.0))
    varied = generate(p, effect=EffectFunction(tau0=-5.0, heterogeneity=1.0))

    assert np.std(flat['tau_true']) == 0.0, "heterogeneity=0 should give a constant effect"
    assert np.std(varied['tau_true']) > 0.5, "heterogeneity=1 should give real spread"

    # Both averages sit on tau0, up to the sampling variation of a centred index.
    assert abs(flat['true_ate'] - (-5.0)) < 0.01
    assert abs(varied['true_ate'] - (-5.0)) < 0.35, \
        "heterogeneity moved the ATE to {:.3f}".format(varied['true_ate'])

    print("    constant: ATE {:.3f}, sd(tau) {:.3f} | varied: ATE {:.3f}, sd(tau) {:.3f}".format(
        flat['true_ate'], np.std(flat['tau_true']),
        varied['true_ate'], np.std(varied['tau_true'])))


def test_heterogeneity_is_recoverable():
    """
    With tau(X) non-constant there is now something for a CATE estimator to find.
    An interacted outcome model should track the true per-patient effect.
    """
    p = pipe()
    out = generate(p, effect=EffectFunction(tau0=-5.0, heterogeneity=1.5),
                   mechanism='designed', overlap_strength=1.0)
    cate_hat = cate_predictions(out['data'], 'outcome', 'treatment', COVARIATES)
    corr = float(np.corrcoef(cate_hat, out['tau_true'])[0, 1])

    assert corr > 0.8, "CATE estimate should track the truth, got r = {:.3f}".format(corr)
    print("    correlation between estimated and true per-patient effect: {:.3f}".format(corr))


def test_constant_effect_path_is_unchanged():
    """Passing no EffectFunction must reproduce the Tier 0 behaviour exactly."""
    p = pipe()
    legacy = generate(p, tau=-5.0)
    explicit = generate(p, tau=-5.0, effect=EffectFunction(tau0=-5.0, heterogeneity=0.0))
    pd.testing.assert_frame_equal(legacy['data'], explicit['data'])
    print("    scalar tau and EffectFunction(heterogeneity=0) agree exactly")


# ---------------------------------------------------------------------------
# The overlap dial
# ---------------------------------------------------------------------------
def test_overlap_strength_degrades_positivity_monotonically():
    """
    Raising lambda must widen the propensity distribution and, past some point,
    push patients into the tails.

    Note what is NOT asserted: that lambda = 1 already produces extreme
    propensities. It does not, and should not -- lambda = 1 is the cell with real
    confounding and comfortable overlap, which is precisely the setting where
    adjustment is supposed to work. Confounding and positivity are separate
    failures, and the grid is built to keep them separate.
    """
    p = pipe()
    seen = []
    for lam in (0.0, 1.0, 3.0):
        out = generate(p, mechanism='designed', overlap_strength=lam)
        seen.append((lam, out['overlap']['pct_extreme'],
                     out['overlap']['ps_min'], out['overlap']['ps_max']))

    pct = [s[1] for s in seen]
    spread = [s[3] - s[2] for s in seen]

    assert pct[0] == 0.0, "lambda=0 is a randomised trial: no extreme propensities"
    assert pct[1] <= pct[2], "positivity must not improve as lambda rises: {}".format(pct)
    assert pct[2] > 0.05, "lambda=3 should materially violate positivity, got {:.3f}".format(pct[2])
    assert spread[2] > spread[1] > spread[0], "propensity spread must widen with lambda"

    for lam, pe, lo, hi in seen:
        print("    lambda={:.1f}  extreme={:.1%}  propensity range [{:.3f}, {:.3f}]".format(
            lam, pe, lo, hi))


def test_generator_underdisperses_the_covariate_index():
    """
    A recorded limitation, not a pass/fail on the design.

    The prognostic index has unit standard deviation on the real training data by
    construction. On synthetic covariates it comes out SMALLER, because the CVAE --
    trained with a squared-error reconstruction loss -- systematically shrinks
    variance. The practical consequence is that the overlap dial is weaker than its
    nominal setting: lambda = 3 behaves like roughly lambda = 2.6.

    This is asserted loosely, as a tripwire: if the ratio ever collapses further, the
    generator has degraded and the grid's axis labels stop meaning what they say.
    """
    p = pipe()
    X_real = encode_data(p['df'], outcome_col='outcome', treatment_col='treatment')[0]
    out = generate(p, mechanism='designed', overlap_strength=1.0)

    sd_real = float(p['index'].standardized(X_real).std())
    sd_syn = float(p['index'].standardized(out['X_encoded']).std())
    ratio = sd_syn / sd_real

    assert 0.6 < ratio < 1.2, "index dispersion ratio out of range: {:.3f}".format(ratio)
    print("    index sd: real {:.3f} vs synthetic {:.3f} (ratio {:.3f}) -- the dial "
          "runs ~{:.0f}% weak".format(sd_real, sd_syn, ratio, 100 * (1 - ratio)))


def test_lambda_zero_is_a_randomised_trial():
    """
    At lambda = 0 treatment does not depend on the covariates, so the naive
    comparison is already unbiased. This is the reference cell that makes every
    other cell interpretable.
    """
    p = pipe()
    out = generate(p, mechanism='designed', overlap_strength=0.0, rho=0.0)
    naive = naive_ate(out['data'], 'outcome', 'treatment')
    bias = naive - out['true_ate']

    se = 2.0 * p['sd_y'] / np.sqrt(len(out['data']))
    assert abs(bias) < 4.0 * se, \
        "randomised cell should be unbiased, got bias {:+.3f}".format(bias)
    print("    randomised: naive bias {:+.3f} (tolerance +/-{:.3f})".format(bias, 4 * se))


def test_confounding_biases_naive_but_not_adjustment():
    p = pipe()
    out = generate(p, mechanism='designed', overlap_strength=2.0, rho=0.0)
    data, true_ate = out['data'], out['true_ate']

    naive = naive_ate(data, 'outcome', 'treatment')
    gcomp = g_computation(data, 'outcome', 'treatment', COVARIATES)

    se = 2.0 * p['sd_y'] / np.sqrt(len(data))
    assert abs(naive - true_ate) > abs(gcomp - true_ate), "adjustment should help"
    assert abs(gcomp - true_ate) < 4.0 * se, \
        "g-computation should recover the ATE, bias {:+.3f}".format(gcomp - true_ate)

    print("    lambda=2, rho=0: naive bias {:+.3f} | g-computation bias {:+.3f}".format(
        naive - true_ate, gcomp - true_ate))


def test_designed_mechanism_still_passes_the_truth_gate_with_heterogeneity():
    """
    The Tier 0 truth gate, re-run with BOTH new features on at once. This is the
    combination most likely to break: a varying effect estimated under confounding.
    """
    p = pipe()
    out = generate(p, mechanism='designed', overlap_strength=1.5, rho=0.0,
                   effect=EffectFunction(tau0=-5.0, heterogeneity=1.0))
    gcomp = g_computation(out['data'], 'outcome', 'treatment', COVARIATES)
    se = 2.0 * p['sd_y'] / np.sqrt(len(out['data']))

    assert abs(gcomp - out['true_ate']) < 4.0 * se, \
        "truth gate failed with tau(X) and confounding: {:.3f} vs {:.3f}".format(
            gcomp, out['true_ate'])
    print("    true ATE {:.3f} | g-computation {:.3f} (tolerance +/-{:.3f})".format(
        out['true_ate'], gcomp, 4 * se))


def test_overlap_diagnostics_match_the_pilots_definition():
    ps = np.array([0.01, 0.04, 0.5, 0.5, 0.96, 0.99])
    Z = np.array([0, 0, 1, 0, 1, 1])
    d = overlap_diagnostics(ps, Z)
    assert abs(d['pct_extreme'] - 4.0 / 6.0) < 1e-12
    assert d['ps_min'] == 0.01 and d['ps_max'] == 0.99
    print("    pct_extreme counts propensities outside [0.05, 0.95]: {:.3f}".format(
        d['pct_extreme']))


# ---------------------------------------------------------------------------
# The factorial grid
# ---------------------------------------------------------------------------
def test_scenario_grid_is_complete_and_reproducible():
    p = pipe()
    kwargs = dict(model=p['model'], index=p['index'], original_df=p['df'],
                  feature_names=p['feature_names'], sd_y=p['sd_y'],
                  p_treat=p['p_treat'], outcome_col='outcome',
                  treatment_col='treatment', covariates=COVARIATES,
                  tau0=-5.0, n=1500, seed=7)

    grid = run_scenario_grid(**kwargs)
    again = run_scenario_grid(**kwargs)

    assert len(grid) == 3 * 2 * 2, "expected 12 cells, got {}".format(len(grid))
    pd.testing.assert_frame_equal(grid, again)
    assert grid['true_ate'].notna().all() and grid['gcomp'].notna().all()

    print("    {} cells, reproducible at a fixed seed".format(len(grid)))
    print(grid[['overlap', 'heterogeneity', 'hidden_bias', 'true_ate',
                'naive_bias', 'gcomp_bias', 'pct_extreme']].to_string(index=False))


def test_grid_separates_the_three_axes():
    """
    The point of a factorial: each axis should move something the others do not.
    If two axes always move the same column together, the design has failed and no
    conclusion drawn from it is attributable.
    """
    p = pipe()
    grid = run_scenario_grid(
        model=p['model'], index=p['index'], original_df=p['df'],
        feature_names=p['feature_names'], sd_y=p['sd_y'], p_treat=p['p_treat'],
        outcome_col='outcome', treatment_col='treatment', covariates=COVARIATES,
        tau0=-5.0, n=1500, seed=7)

    # Overlap drives positivity; the other two axes do not.
    by_overlap = grid.groupby('lambda')['pct_extreme'].mean()
    assert by_overlap.is_monotonic_increasing, "positivity should worsen with lambda"

    # Heterogeneity drives the spread of the effect; it should not drive positivity.
    by_kappa_extreme = grid.groupby('kappa')['pct_extreme'].mean()
    assert abs(by_kappa_extreme.iloc[0] - by_kappa_extreme.iloc[-1]) < 0.02, \
        "heterogeneity should not change positivity"
    by_kappa_sd = grid.groupby('kappa')['cate_sd'].mean()
    assert by_kappa_sd.iloc[-1] > by_kappa_sd.iloc[0], "heterogeneity should widen tau"

    # Hidden bias drives residual bias after adjustment; overlap alone should not.
    by_rho = grid.groupby('rho')['gcomp_bias'].apply(lambda s: s.abs().mean())
    assert by_rho.iloc[-1] > by_rho.iloc[0], \
        "unmeasured confounding should leave residual bias after adjustment"

    print("    positivity by lambda: {}".format(
        ", ".join("{:.3f}".format(v) for v in by_overlap)))
    print("    |adjusted bias| by rho: {}".format(
        ", ".join("{:.3f}".format(v) for v in by_rho)))


def test_grid_summary_reads_off_the_table():
    p = pipe()
    grid = run_scenario_grid(
        model=p['model'], index=p['index'], original_df=p['df'],
        feature_names=p['feature_names'], sd_y=p['sd_y'], p_treat=p['p_treat'],
        outcome_col='outcome', treatment_col='treatment', covariates=COVARIATES,
        tau0=-5.0, n=1500, seed=7)
    lines = summarize_grid(grid)
    assert len(lines) >= 4
    for line in lines:
        print("    - {}".format(line))


TESTS = [
    test_heterogeneity_does_not_move_the_average_effect,
    test_constant_effect_path_is_unchanged,
    test_heterogeneity_is_recoverable,
    test_overlap_diagnostics_match_the_pilots_definition,
    test_lambda_zero_is_a_randomised_trial,
    test_overlap_strength_degrades_positivity_monotonically,
    test_generator_underdisperses_the_covariate_index,
    test_confounding_biases_naive_but_not_adjustment,
    test_designed_mechanism_still_passes_the_truth_gate_with_heterogeneity,
    test_scenario_grid_is_complete_and_reproducible,
    test_grid_separates_the_three_axes,
    test_grid_summary_reads_off_the_table,
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
