"""The same observed-data comparison in guided and advanced experiments."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estimators import naive_ate, g_computation


def compare_estimates(run):
    data = run['result']['data']
    y, z = run['outcome_col'], run['treatment_col']
    if data[z].nunique() != 2:
        raise ValueError('Both treatment groups are needed to compare estimates. Generate another cohort.')
    # Only the exported measured covariates enter adjustment. Never join oracle data.
    covariates = list(run['result']['decoded'].columns)
    truth = float(np.mean(run['result']['tau_true']))
    estimates = [naive_ate(data, y, z), g_computation(data, y, z, covariates)]
    if not np.isfinite(estimates).all():
        raise ValueError('This cohort did not produce finite estimates.')
    return pd.DataFrame({'method': ['Unadjusted', 'Adjusted (g-computation)'],
                         'estimate': estimates, 'error': np.asarray(estimates) - truth,
                         'truth': truth})


def comparison_plot(comparison, reveal=True, units='outcome units'):
    """Point estimates from one cohort; intentionally no invented uncertainty bars."""
    # Shiny resizes figures to their container after this function returns.
    # Constrained layout recalculates margins at draw time, including on phones.
    fig, ax = plt.subplots(figsize=(8.4, 2.7), facecolor='white', layout='constrained')
    values = comparison['estimate'].to_numpy()
    truth = float(comparison['truth'].iloc[0])
    visible = np.r_[values, 0, truth] if reveal else np.r_[values, 0]
    span = max(float(np.ptp(visible)), 1.0)
    ax.set_xlim(visible.min() - .22 * span, visible.max() + .28 * span)
    ax.axvline(0, color='#cbd5e1', linewidth=1, zorder=0)
    if reveal:
        ax.axvline(truth, color='#2855a6', linestyle='--', linewidth=1.5,
                   label=f'True cohort effect: {truth:+.2f}')
        ax.legend(loc='lower right', frameon=False, fontsize=9)
    for y, value, color in zip([1, 0], values, ['#b85c26', '#087f79']):
        ax.scatter(value, y, s=105, color=color, zorder=3)
        ax.annotate(f'{value:+.2f}', (value, y), xytext=(10, 8),
                    textcoords='offset points', fontsize=11, color=color, weight='bold')
    ax.set_yticks([1, 0], ['Unadjusted', 'Adjusted'])
    ax.set_ylim(-.7, 1.55)
    ax.set_xlabel(f'Average treatment effect ({units})')
    ax.grid(axis='x', alpha=.15)
    ax.grid(axis='y', visible=False)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return fig
