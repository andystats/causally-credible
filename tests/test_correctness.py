"""Regression tests for the October app audit; run directly or with unittest."""
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from cvae_model import encode_data, decode_samples
from dgp import EffectFunction, PrognosticIndex, generate_cohort, align_encoding
from examples import load_example
from realism import realism_report, interpret_auc
from schema import CovariateSchema
from scenarios import run_scenario_grid
from workflow import ExperimentSession, prepare_data, analysis_summary, answer_bundle


class SchemaTests(unittest.TestCase):
    def test_perfect_one_hot_preserves_one_percent_category(self):
        frame = pd.DataFrame({'group': ['common'] * 9900 + ['rare'] * 100})
        schema = CovariateSchema.fit(frame)
        decoded = schema.decode(schema.encode(frame), rng=np.random.default_rng(7))
        self.assertEqual((decoded.group == 'rare').sum(), 100)

    def test_probability_sampling_preserves_rare_marginal(self):
        schema = CovariateSchema.fit(pd.DataFrame({'group': ['common', 'rare']}))
        samples = np.tile([.99, .01], (20000, 1))
        self.assertAlmostEqual((schema.decode(samples).group == 'rare').mean(), .01, delta=.003)

    def test_zero_probability_fallback_uses_training_marginal(self):
        schema = CovariateSchema.fit(pd.DataFrame({'group': ['a'] * 99 + ['b']}))
        decoded = schema.decode(np.zeros((20000, 2)))
        self.assertAlmostEqual((decoded.group == 'b').mean(), .01, delta=.003)

    def test_prefixes_and_encoded_name_collisions_round_trip(self):
        frame = pd.DataFrame({'x': ['a', 'b'], 'x_y': ['c', 'd'], 'x_a': [2., 3.]})
        schema = CovariateSchema.fit(frame)
        self.assertEqual(len(schema.feature_names), len(set(schema.feature_names)))
        decoded = schema.decode(schema.encode(frame))
        for col in frame:
            np.testing.assert_array_equal(frame[col], decoded[col])

    def test_string_and_numeric_categorical_types_are_preserved(self):
        frame = pd.DataFrame({'text': pd.Series(['one', 'two'], dtype='string'),
                              'coded': pd.Categorical([10, 20]), 'id': [101, 102]})
        schema = CovariateSchema.fit(frame)
        decoded = schema.decode(schema.encode(frame))
        for col in frame:
            np.testing.assert_array_equal(frame[col], decoded[col])

    def test_holdout_missing_level_retains_training_feature_order(self):
        schema = CovariateSchema.fit(pd.DataFrame({'site': ['a', 'b', 'c']}))
        heldout = schema.encode(pd.DataFrame({'site': ['b', 'a']}))
        np.testing.assert_array_equal(heldout, [[0, 1, 0], [1, 0, 0]])

    def test_holdout_unseen_level_is_explicitly_rejected(self):
        schema = CovariateSchema.fit(pd.DataFrame({'site': ['a', 'b']}))
        with self.assertRaisesRegex(ValueError, 'Unseen category'):
            schema.encode(pd.DataFrame({'site': ['a', 'c']}))

    def test_diagnostic_rejects_unaligned_matrices(self):
        with self.assertRaisesRegex(ValueError, 'aligned matrices'):
            realism_report(np.ones((5, 2)), np.ones((5, 3)), ['a', 'b', 'c'])

    def test_chance_discriminator_does_not_claim_equivalence(self):
        self.assertIn('not proof', interpret_auc(.5))


class InputTests(unittest.TestCase):
    def setUp(self):
        self.df, _ = load_example('pneumonia')

    def test_confirm_preserves_zero_one_even_when_first_row_is_treated(self):
        self.assertEqual(self.df.treatment.iloc[0], 1)
        result = prepare_data(self.df, 'outcome', 'treatment')
        pd.testing.assert_series_equal(result['data'].treatment, self.df.treatment)

    def test_reordering_does_not_change_treatment_meaning(self):
        for frame in [self.df, self.df.iloc[::-1]]:
            result = prepare_data(frame, 'outcome', 'treatment')
            np.testing.assert_array_equal(result['data'].treatment, frame.treatment)

    def test_nonbinary_labels_require_explicit_treated_level(self):
        frame = self.df.assign(treatment=self.df.treatment.map({0: 'usual', 1: 'new'}))
        with self.assertRaisesRegex(ValueError, 'Choose which'):
            prepare_data(frame, 'outcome', 'treatment')
        result = prepare_data(frame, 'outcome', 'treatment', 'new')
        np.testing.assert_array_equal(result['data'].treatment, self.df.treatment)
        self.assertEqual(result['treatment_mapping'], {'0': 'usual', '1': 'new'})

    def test_both_arms_are_rechecked_after_complete_case_filtering(self):
        self.df.loc[self.df.treatment == 1, 'outcome'] = np.nan
        with self.assertRaisesRegex(ValueError, 'after rows'):
            prepare_data(self.df, 'outcome', 'treatment')

    def test_invalid_roles_and_outcomes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different'):
            prepare_data(self.df, 'outcome', 'outcome')
        for outcome in [np.ones(len(self.df)), np.repeat(np.inf, len(self.df)), ['word'] * len(self.df)]:
            with self.assertRaises(ValueError):
                prepare_data(self.df.assign(outcome=outcome), 'outcome', 'treatment')


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.df, _ = load_example('pneumonia')
        cls.trained = ExperimentSession().confirm(cls.df, 'outcome', 'treatment').train(epochs=3)
        cls.generated = cls.trained.generate(n=200, heterogeneity=1)

    def test_summary_uses_mean_truth_not_first_patient(self):
        run = self.generated.run
        summary = analysis_summary(run)
        self.assertAlmostEqual(summary['true_ate'], run['result']['tau_true'].mean())
        self.assertNotAlmostEqual(summary['true_ate'], run['result']['tau_true'][0])
        self.assertAlmostEqual(summary['error'], summary['naive'] - summary['true_ate'])

    def test_reconfirm_clears_model_results_and_grid(self):
        full = self.generated.run_grid(tau0=-5, n=200, seed=7)
        other, _ = load_example('lalonde')
        fresh = full.confirm(other, 'outcome', 'treatment')
        self.assertEqual(len(fresh.confirmed['data']), 300)
        self.assertIsNone(fresh.trained)
        self.assertIsNone(fresh.run)
        self.assertIsNone(fresh.grid)
        with self.assertRaisesRegex(ValueError, 'Train the model'):
            fresh.generate()

    def test_retrain_clears_old_run_and_grid(self):
        retrained = self.generated.train(seed=2, epochs=1)
        self.assertIsNone(retrained.run)
        self.assertIsNone(retrained.grid)
        self.assertEqual(retrained.trained['seed'], 2)

    def test_confirmation_freezes_uploaded_data(self):
        original = self.df.copy()
        confirmed = ExperimentSession().confirm(original, 'outcome', 'treatment')
        original['outcome'] = 999
        self.assertFalse((confirmed.confirmed['data'].outcome == 999).all())

    def test_realism_sample_conditions_on_actual_heldout_treatments(self):
        state = self.trained.trained
        with patch.object(state['model'], 'sample', wraps=state['model'].sample) as sampler:
            self.trained.generate(n=200)
        np.testing.assert_array_equal(sampler.call_args_list[-1].args[0], state['Z_hold'])

    def test_answer_bundle_separates_observed_data_and_oracle(self):
        run = self.generated.run
        self.assertFalse({'y0', 'y1', 'tau_true', 'U'} & set(run['result']['data']))
        with zipfile.ZipFile(BytesIO(answer_bundle(run))) as bundle:
            observed = pd.read_csv(bundle.open('observed.csv'))
            oracle = pd.read_csv(bundle.open('answer_key.csv'))
            manifest = json.loads(bundle.read('manifest.json'))
        self.assertEqual(len(observed), len(oracle))
        np.testing.assert_allclose(observed.outcome, np.where(observed.treatment == 1, oracle.y1, oracle.y0))
        self.assertAlmostEqual((oracle.y1 - oracle.y0).mean(), manifest['true_ate'])
        self.assertEqual(manifest['parameters']['treatment_mapping'], {'0': '0', '1': '1'})

    def test_colliding_upload_names_survive_unchanged_in_observed_frame(self):
        frame = self.df.copy()
        for col in ['U', 'y0', 'y1', 'tau_true', 'id']:
            frame[col] = np.arange(len(frame)) + 100
        state = ExperimentSession().confirm(frame, 'outcome', 'treatment').train(epochs=1)
        result = state.generate(n=100, rho=.5).run['result']
        for col in ['U', 'y0', 'y1', 'tau_true', 'id']:
            np.testing.assert_array_equal(result['data'][col], result['decoded'][col])

    def test_learned_overlap_is_not_reported_as_observed_propensity(self):
        run = self.trained.generate(n=100, mechanism='learned').run
        self.assertIsNone(run['result']['propensity'])
        self.assertEqual(run['result']['overlap']['kind'], 'unavailable')

    def test_designed_hidden_confounding_probability_is_labelled_oracle(self):
        result = self.trained.generate(n=100, rho=.5).run['result']
        self.assertEqual(result['overlap']['kind'], 'P(Z=1|X,U)')

    def test_grid_uses_same_covariates_and_noise_when_only_effect_changes(self):
        import scenarios
        captures = []
        real_generate = scenarios.generate_cohort
        def capture(*args, **kwargs):
            out = real_generate(*args, **kwargs)
            captures.append(out)
            return out
        s = self.trained.trained
        with patch.object(scenarios, 'generate_cohort', side_effect=capture):
            run_scenario_grid(s['model'], s['index'], s['df_train'], s['feature_names'], s['sd_y'],
                              s['p_treat'], 'outcome', 'treatment', s['feature_names'].schema.columns,
                              n=200, overlap_levels=[(1, 'mild')],
                              heterogeneity_levels=[(0, 'constant'), (1, 'varies')], rho_levels=[(0, 'none')])
        np.testing.assert_array_equal(captures[0]['X_encoded'], captures[1]['X_encoded'])
        np.testing.assert_array_equal(captures[0]['oracle'].y0, captures[1]['oracle'].y0)
        np.testing.assert_array_equal(captures[0]['Z'], captures[1]['Z'])
        self.assertFalse(np.array_equal(captures[0]['oracle'].y1, captures[1]['oracle'].y1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
