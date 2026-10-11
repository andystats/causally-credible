"""A fitted covariate contract shared by training, holdout and generation."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def column_kind(series):
    if (isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series.dtype)):
        return {'kind': 'categorical'}
    if not (pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)):
        raise ValueError(f"Unsupported covariate type for {series.name!r}: {series.dtype}")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Covariate {series.name!r} contains missing or infinite values.")
    levels = np.unique(values)
    if len(levels) == 2:
        return {'kind': 'binary', 'low': levels[0], 'high': levels[1]}
    if np.equal(levels, np.round(levels)).all():
        return {'kind': 'integer', 'min': levels.min(), 'max': levels.max()}
    return {'kind': 'continuous'}


@dataclass
class CovariateSchema:
    columns: list
    specs: dict
    feature_names: list

    @classmethod
    def fit(cls, frame):
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("Provide at least one covariate with unique column names.")
        specs, names = {}, []
        # Numeric names keep their familiar display labels. Category mappings are
        # explicit; no decoder relies on prefixes or splitting strings.
        numeric_names = {c for c in frame if column_kind(frame[c])['kind'] != 'categorical'}
        used = set(numeric_names)
        for col in frame:
            spec = column_kind(frame[col])
            if spec['kind'] == 'categorical':
                if frame[col].isna().any():
                    raise ValueError(f"Covariate {col!r} contains missing values.")
                levels = sorted(frame[col].unique().tolist(), key=lambda x: (type(x).__name__, repr(x)))
                spec['levels'] = levels
                spec['frequencies'] = np.array([(frame[col] == v).mean() for v in levels])
                encoded = []
                for level in levels:
                    name = f'{col}_{level}'
                    while name in used:
                        name += '#'
                    used.add(name)
                    encoded.append(name)
            else:
                encoded = [col]
            spec['features'] = encoded
            spec['indices'] = list(range(len(names), len(names) + len(encoded)))
            names.extend(encoded)
            specs[col] = spec
        return cls(list(frame.columns), specs, names)

    def encode(self, frame):
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing covariates: {sorted(missing)}')
        blocks = []
        for col in self.columns:
            spec = self.specs[col]
            if spec['kind'] == 'categorical':
                unknown = ~frame[col].isin(spec['levels'])
                if unknown.any():
                    raise ValueError(
                        f"Unseen category in {col!r}. Use more training data or another "
                        "training seed / holdout fraction so every level appears in training.")
                blocks.append(np.column_stack([(frame[col] == v).to_numpy(dtype=float)
                                               for v in spec['levels']]))
            else:
                blocks.append(frame[[col]].to_numpy(dtype=float))
        encoded = np.column_stack(blocks)
        if not np.isfinite(encoded).all():
            raise ValueError('Covariates must be finite and complete.')
        return encoded

    def decode(self, samples, rng=None, sample_categories=True):
        rng = np.random.default_rng(0) if rng is None else rng
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(self.feature_names):
            raise ValueError('Generated covariates do not match the fitted schema.')
        if not np.isfinite(samples).all():
            raise ValueError('The generator returned non-finite covariates.')
        decoded = {}
        for col in self.columns:
            spec = self.specs[col]
            values = samples[:, spec['indices']]
            if spec['kind'] == 'categorical':
                # The MSE decoder reconstructs one-hot values, NOT logits.
                # Project onto nonnegative probabilities, then normalize. If the
                # entire block is invalid, use the training marginal explicitly.
                probs = np.clip(values, 0, 1)
                totals = probs.sum(axis=1, keepdims=True)
                probs = np.divide(probs, totals, out=np.tile(spec['frequencies'], (len(values), 1)),
                                  where=totals > 0)
                if sample_categories:
                    draws = rng.random(len(values))
                    picks = (draws[:, None] > probs.cumsum(axis=1)).sum(axis=1)
                    picks = np.minimum(picks, len(spec['levels']) - 1)
                else:
                    picks = probs.argmax(axis=1)
                decoded[col] = pd.Categorical([spec['levels'][i] for i in picks], categories=spec['levels'])
            elif spec['kind'] == 'binary':
                p = np.clip((values[:, 0] - spec['low']) / (spec['high'] - spec['low']), 0, 1)
                draws = rng.random(len(p)) < p if sample_categories else p >= .5
                decoded[col] = np.where(draws, spec['high'], spec['low'])
            elif spec['kind'] == 'integer':
                decoded[col] = np.clip(np.round(values[:, 0]), spec['min'], spec['max'])
            else:
                decoded[col] = values[:, 0]
        return pd.DataFrame(decoded)


class FeatureNames(list):
    """List-compatible feature labels carrying their fitted encoding contract."""
    def __init__(self, schema):
        super().__init__(schema.feature_names)
        self.schema = schema


def fitted_schema(feature_names, original_covariates=None):
    schema = getattr(feature_names, 'schema', None)
    if schema is None and original_covariates is not None:
        schema = CovariateSchema.fit(original_covariates)
    if schema is None or list(feature_names) != schema.feature_names:
        raise ValueError('Reuse the fitted feature schema returned by encode_data().')
    return schema
