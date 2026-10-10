"""Validated inputs and immutable experiment snapshots for the Shiny session."""
from dataclasses import dataclass, replace
from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd

from cvae_model import ConditionalVAE, encode_data, decode_samples, set_seed
from dgp import PrognosticIndex, EffectFunction, generate_cohort, align_encoding
from realism import realism_report
from scenarios import run_scenario_grid


def treatment_levels(series):
    return sorted(series.dropna().unique().tolist(), key=lambda v: (type(v).__name__, repr(v)))


def prepare_data(df, outcome_col, treatment_col, treated_level=None):
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError('Column names must be unique strings.')
    if outcome_col == treatment_col or {outcome_col, treatment_col} - set(df.columns):
        raise ValueError('Choose different, valid outcome and treatment columns.')
    if len(df.columns) < 3:
        raise ValueError('Include at least one covariate as well as outcome and treatment.')
    clean = df.dropna().copy().reset_index(drop=True)
    if len(clean) < 40:
        raise ValueError('At least 40 complete rows are needed for training and held-out diagnostics.')
    levels = treatment_levels(clean[treatment_col])
    if len(levels) != 2:
        raise ValueError('Treatment must have two levels after rows with missing values are removed.')
    if set(levels) == {0, 1}:
        treated_level = 1
    elif treated_level is None or treated_level not in levels:
        raise ValueError('Choose which treatment level represents treated patients.')
    control = next(v for v in levels if v != treated_level)
    clean[treatment_col] = (clean[treatment_col] == treated_level).astype(int)
    try:
        clean[outcome_col] = pd.to_numeric(clean[outcome_col], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('The outcome must be numeric.') from exc
    values = clean[outcome_col].to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.std() == 0:
        raise ValueError('The outcome must be finite and have positive variation.')
    # Validate the whole input, including unsupported types and infinite covariates.
    encode_data(clean, outcome_col, treatment_col)
    return {'data': clean, 'outcome_col': outcome_col, 'treatment_col': treatment_col,
            'dropped_rows': len(df) - len(clean),
            'treatment_mapping': {'0': str(control), '1': str(treated_level)}}


def fit_experiment(prepared, *, seed=1, holdout_frac=.25, epochs=150, batch_size=128):
    if not .1 <= holdout_frac <= .5 or int(epochs) != epochs or not 1 <= epochs <= 400:
        raise ValueError('Choose a holdout fraction from 0.1 to 0.5 and 1–400 epochs.')
    if int(seed) != seed or seed < 0:
        raise ValueError('Training seed must be a nonnegative integer.')
    df = prepared['data']
    y, z = prepared['outcome_col'], prepared['treatment_col']
    order = np.random.default_rng(seed).permutation(len(df))
    n_hold = max(2, int(round(holdout_frac * len(df))))
    hold = df.iloc[order[:n_hold]].reset_index(drop=True)
    train = df.iloc[order[n_hold:]].reset_index(drop=True)
    if train[z].nunique() != 2:
        raise ValueError('The training split needs both treatment arms. Change the seed or holdout fraction.')
    X, Z, names = encode_data(train, y, z)
    # A missing known level produces an all-zero column in exactly the same
    # position. An unseen level is rejected with an actionable validation message.
    Xh, Zh, _ = encode_data(hold, y, z, schema=names.schema)
    set_seed(seed)
    model = ConditionalVAE(X.shape[1]).fit(X, Z, epochs=epochs, batch_size=batch_size,
                                         seed=seed, verbose=False)
    return {'model': model, 'index': PrognosticIndex().fit(X, names),
            'feature_names': names, 'X_train': X, 'X_hold': Xh, 'Z_hold': Zh,
            'df_train': train, 'df_hold': hold, 'n_train': len(train), 'n_hold': len(hold),
            'batch_size': batch_size, 'seed': seed, 'prepared': prepared,
            'outcome_col': y, 'treatment_col': z, 'sd_y': float(train[y].std()),
            'p_treat': float(train[z].mean())}


def generate_experiment(state, *, n=1500, tau0=-5, rho=0, seed=7,
                        heterogeneity=0, mechanism='designed', overlap_strength=1):
    if (int(n) != n or not 100 <= n <= 10000 or int(seed) != seed or seed < 0
            or not np.isfinite([tau0, rho, heterogeneity, overlap_strength]).all()
            or not 0 <= rho <= 1 or not 0 <= heterogeneity <= 3 or not 0 <= overlap_strength <= 4):
        raise ValueError('Use finite parameters within the displayed ranges and an integer nonnegative seed.')
    params = dict(n=int(n), tau=float(tau0), rho=float(rho), seed=int(seed),
                  mechanism=mechanism, overlap_strength=float(overlap_strength))
    result = generate_cohort(state['model'], state['index'], state['df_train'],
                             state['feature_names'], sd_y=state['sd_y'], p_treat=state['p_treat'],
                             outcome_col=state['outcome_col'], treatment_col=state['treatment_col'],
                             effect=EffectFunction(tau0, heterogeneity), **params)
    # Fidelity is checked against the held-out arm mix, using exactly those Zs.
    raw = state['model'].sample(state['Z_hold'], seed=seed + 1)
    decoded = decode_samples(raw, state['feature_names'], state['df_train'],
                             outcome_col=state['outcome_col'], treatment_col=state['treatment_col'],
                             rng=np.random.default_rng(seed + 1))
    report = realism_report(state['X_hold'], align_encoding(decoded, state['feature_names']),
                            state['feature_names'], seed=seed)
    return {'result': result, 'realism': report,
            'outcome_col': state['outcome_col'], 'treatment_col': state['treatment_col'],
            'parameters': {**params, 'heterogeneity': heterogeneity,
                           'training_seed': state['seed'],
                           'treatment_mapping': state['prepared']['treatment_mapping']}}


def analysis_summary(run):
    data, result = run['result']['data'], run['result']
    y, z = run['outcome_col'], run['treatment_col']
    # Recompute from per-person truth so a stale input control cannot redefine it.
    truth = float(np.mean(result['tau_true']))
    naive = float(data.loc[data[z] == 1, y].mean() - data.loc[data[z] == 0, y].mean())
    return {'true_ate': truth, 'naive': naive, 'error': naive - truth}


def answer_bundle(run):
    """Observed data and answers share row order, never a feature matrix."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('observed.csv', run['result']['data'].to_csv(index=False))
        archive.writestr('answer_key.csv', run['result']['oracle'].to_csv(index=False))
        manifest = {'schema_version': 1, 'row_alignment': 'CSV row order, one-based after header',
                    'outcome_col': run['outcome_col'], 'treatment_col': run['treatment_col'],
                    'parameters': run['parameters'], 'true_ate': run['result']['true_ate'],
                    'effect_definition': 'tau0 + kappa * training_outcome_sd * training_centered_modifier',
                    'overlap_probability': run['result']['overlap']['kind']}
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        archive.writestr('README.txt', 'Analyze observed.csv only. answer_key.csv contains counterfactual outcomes, '
                         'individual effects and the hidden confounder; these must not enter learner features. '
                         'The two files align by row position. Compare before sorting either file.\n')
    return buffer.getvalue()


@dataclass(frozen=True)
class ExperimentSession:
    confirmed: object = None
    trained: object = None
    run: object = None
    grid: object = None

    def confirm(self, df, outcome_col, treatment_col, treated_level=None):
        return ExperimentSession(confirmed=prepare_data(df, outcome_col, treatment_col, treated_level))

    def train(self, **kwargs):
        if self.confirmed is None:
            raise ValueError('Confirm the data before training.')
        return ExperimentSession(confirmed=self.confirmed, trained=fit_experiment(self.confirmed, **kwargs))

    def generate(self, **kwargs):
        if self.trained is None:
            raise ValueError('Train the model before generating data.')
        return replace(self, run=generate_experiment(self.trained, **kwargs))

    def run_grid(self, *, tau0, n, seed):
        if self.trained is None:
            raise ValueError('Train the model before running the grid.')
        if int(n) != n or not 200 <= n <= 8000 or int(seed) != seed or seed < 0 or not np.isfinite(tau0):
            raise ValueError('Use 200–8000 rows, a nonnegative integer seed and finite baseline effect.')
        s = self.trained
        grid = run_scenario_grid(s['model'], s['index'], s['df_train'], s['feature_names'],
                                 s['sd_y'], s['p_treat'], s['outcome_col'], s['treatment_col'],
                                 s['feature_names'].schema.columns, tau0=tau0, n=n, seed=seed)
        return replace(self, grid=grid)
