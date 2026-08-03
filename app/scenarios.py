"""
The factorial scenario grid.

WHY A GRID AND NOT A DEMO
-------------------------
It is easy to build a demo that shows an estimator failing. It is much harder, and
much more useful, to show WHY it failed -- and a single dramatic scenario can never
do that, because several things are wrong at once.

So we borrow the design of the companion R study, `survcausal-pilot`. Its headline
result is a 2x2 factorial: confounding shape crossed with hazard shape, at a single
fixed effect size, so that any difference between cells is attributable to ONE
cause. Its README puts it plainly -- the alternative "would change several things at
once and make the cause ambiguous".

Here we cross three axes that Tier 1 made independent:

    overlap        lambda : how steeply treatment depends on the covariates
    heterogeneity  kappa  : how much the treatment effect varies across patients
    hidden bias    rho    : how strongly an unmeasured confounder acts

Every other quantity -- the generator, the covariates, tau0, the sample size -- is
held fixed. That is the whole discipline: move one thing at a time.

WHAT EACH CELL REPORTS
----------------------
    true ATE      the average of the individual effects actually generated
    naive         treated mean minus control mean; adjusts for nothing
    g-computation outcome regression + standardization, with interactions
    pct extreme   share of patients with propensity < 0.05 or > 0.95

The last column is the one to watch. It is the positivity diagnostic that the R
pilot already computes and never varies -- across all four of its design cells it
reads 0.00000, 0.00000, 0.00036, 0.00047. A diagnostic that never moves teaches
nothing. Here it moves, and you can watch estimator bias move with it.
"""

import numpy as np
import pandas as pd

from dgp import EffectFunction, generate_cohort
from estimators import naive_ate, g_computation, cate_predictions


# Default levels. Chosen so the three columns of the grid are qualitatively
# different rather than merely numerically different -- a grid whose cells all look
# the same is a grid nobody learns from.
DEFAULT_OVERLAP = [
    (0.0, "randomised"),      # every patient equally likely to be treated
    (1.0, "mild"),            # comfortable overlap, real confounding
    (3.0, "poor"),            # propensities pile up at 0 and 1
]
DEFAULT_HETEROGENEITY = [
    (0.0, "constant"),        # every patient benefits identically
    (1.0, "varies"),          # effect varies by one outcome SD per SD of the index
]
DEFAULT_RHO = [
    (0.0, "none"),            # exchangeability holds given X
    (0.5, "present"),         # an unmeasured confounder is at work
]


def run_scenario_grid(model, index, original_df, feature_names, sd_y, p_treat,
                      outcome_col, treatment_col, covariates, tau0=-5.0, n=2000,
                      seed=101, overlap_levels=None, heterogeneity_levels=None,
                      rho_levels=None, progress=None):
    """
    Generate one cohort per design cell and score two estimators on each.

    Returns a tidy data frame, one row per cell. Deliberately tidy rather than
    pre-formatted, so the same table can be rendered in the app, exported to CSV, or
    read straight into R.

    Every cell is seeded from the same base `seed` plus its position in the grid, so
    the whole table is reproducible and any single cell can be re-run in isolation.
    """
    overlap_levels = overlap_levels or DEFAULT_OVERLAP
    heterogeneity_levels = heterogeneity_levels or DEFAULT_HETEROGENEITY
    rho_levels = rho_levels or DEFAULT_RHO

    rows = []
    cell = 0
    total = len(overlap_levels) * len(heterogeneity_levels) * len(rho_levels)

    for lam, lam_label in overlap_levels:
        for kappa, kappa_label in heterogeneity_levels:
            for rho, rho_label in rho_levels:
                cell += 1
                if progress is not None:
                    progress(cell, total)

                effect = EffectFunction(tau0=tau0, heterogeneity=kappa)
                result = generate_cohort(
                    model=model, index=index, original_df=original_df,
                    feature_names=feature_names, n=n, tau=tau0, rho=rho,
                    sd_y=sd_y, p_treat=p_treat, outcome_col=outcome_col,
                    treatment_col=treatment_col,
                    # Distinct seed per cell, derived from the base seed so the whole
                    # grid moves together when the user changes it.
                    seed=seed + 1000 * cell,
                    mechanism='designed', overlap_strength=lam, effect=effect)

                data = result['data']
                true_ate = result['true_ate']

                naive = naive_ate(data, outcome_col, treatment_col)
                gcomp = g_computation(data, outcome_col, treatment_col, covariates)

                # How well is the VARIATION in benefit recovered? Only a meaningful
                # question when the effect actually varies; correlation with a
                # constant is undefined, so we report NaN rather than a fake number.
                if kappa > 0:
                    cate_hat = cate_predictions(data, outcome_col, treatment_col, covariates)
                    cate_corr = float(np.corrcoef(cate_hat, result['tau_true'])[0, 1])
                    cate_sd = float(np.std(result['tau_true']))
                else:
                    cate_corr, cate_sd = np.nan, 0.0

                rows.append({
                    'overlap': lam_label,
                    'lambda': lam,
                    'heterogeneity': kappa_label,
                    'kappa': kappa,
                    'hidden_bias': rho_label,
                    'rho': rho,
                    'true_ate': true_ate,
                    'naive': naive,
                    'naive_bias': naive - true_ate,
                    'gcomp': gcomp,
                    'gcomp_bias': gcomp - true_ate,
                    'pct_extreme': result['overlap']['pct_extreme'],
                    'ps_min': result['overlap']['ps_min'],
                    'ps_max': result['overlap']['ps_max'],
                    'treated_frac': result['overlap']['treated_frac'],
                    'cate_sd': cate_sd,
                    'cate_corr': cate_corr,
                })

    return pd.DataFrame(rows)


def summarize_grid(grid):
    """
    Reduce the grid to the handful of statements it supports.

    Returned as plain sentences because the point of the grid is not the numbers,
    it is what they let you say. Each of these is computed from the table, so the
    text can never contradict the numbers it sits beside -- the same discipline the
    survcausal-pilot vignette uses for its own prose.
    """
    lines = []

    clean = grid[(grid['rho'] == 0)]
    if len(clean):
        worst = clean.loc[clean['gcomp_bias'].abs().idxmax()]
        best = clean.loc[clean['gcomp_bias'].abs().idxmin()]
        lines.append(
            "With no hidden bias, adjustment works best under {} overlap "
            "(bias {:+.3f}) and worst under {} overlap (bias {:+.3f}).".format(
                best['overlap'], best['gcomp_bias'],
                worst['overlap'], worst['gcomp_bias']))

    rand = grid[(grid['lambda'] == 0) & (grid['rho'] == 0)]
    if len(rand):
        lines.append(
            "Randomised cells: naive bias averages {:+.3f} -- with no confounding, "
            "the unadjusted comparison is already unbiased.".format(
                rand['naive_bias'].mean()))

    confounded = grid[(grid['lambda'] > 0) & (grid['rho'] == 0)]
    if len(confounded):
        lines.append(
            "Confounded cells: naive bias averages {:+.3f} against g-computation's "
            "{:+.3f}.".format(confounded['naive_bias'].mean(),
                              confounded['gcomp_bias'].mean()))

    hidden = grid[grid['rho'] > 0]
    if len(hidden):
        lines.append(
            "With an unmeasured confounder, adjusting for the observed covariates "
            "leaves an average bias of {:+.3f} -- adjustment cannot fix what it "
            "cannot see.".format(hidden['gcomp_bias'].mean()))

    extreme = grid.loc[grid['pct_extreme'].idxmax()]
    lines.append(
        "Positivity ranges from {:.1%} to {:.1%} of patients at extreme propensity; "
        "the worst cell is {} overlap.".format(
            grid['pct_extreme'].min(), grid['pct_extreme'].max(), extreme['overlap']))

    return lines
