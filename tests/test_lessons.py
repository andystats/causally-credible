"""Scientific and export checks for the fixed guided teaching experiments."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from comparison import compare_estimates, comparison_plot
from lessons import LESSONS, teaching_cohort, prepare_lessons, run_lesson, experiment_record


class GuidedLessonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.trained = prepare_lessons()
        cls.results = {key: run_lesson(cls.trained, key, 'uncertain') for key in LESSONS}

    def test_source_cohort_is_reproducible_and_explicitly_continuous(self):
        pd.testing.assert_frame_equal(teaching_cohort(), teaching_cohort())
        self.assertGreater(teaching_cohort().score_change.nunique(), 100)
        self.assertEqual(set(teaching_cohort().program), {0, 1})

    def test_lessons_share_covariates_effect_and_noise(self):
        baseline = self.results['randomized']['run']['result']
        noise = baseline['oracle'].y0 - baseline['fx']
        for key, result in self.results.items():
            run = result['run']['result']
            pd.testing.assert_frame_equal(run['decoded'], baseline['decoded'])
            np.testing.assert_array_equal(run['tau_true'], np.full(2000, 5.))
            actual_noise = run['oracle'].y0 - run['fx'] - .5 * LESSONS[key].hidden * self.trained['sd_y'] * run['U']
            np.testing.assert_allclose(actual_noise, noise, atol=1e-12)

    def test_randomized_assignment_has_constant_probability(self):
        result = self.results['randomized']['run']['result']
        self.assertEqual(np.ptp(result['propensity']), 0)
        self.assertEqual(result['overlap']['kind'], 'P(Z=1|X)')

    def test_fixed_examples_demonstrate_the_intended_contrasts(self):
        # Broad tolerances protect the teaching behavior, not exact fitted weights.
        randomized = self.results['randomized']['comparison']['error']
        measured = self.results['measured']['comparison']['error']
        hidden = self.results['hidden']['comparison']['error']
        self.assertTrue((randomized.abs() < 2).all())
        self.assertGreater(abs(measured.iloc[0]), 4)
        self.assertLess(abs(measured.iloc[1]), 2)
        self.assertGreater(abs(hidden.iloc[1]), 2)
        self.assertGreater(abs(hidden.iloc[1]), abs(measured.iloc[1]))

    def test_adjustment_does_not_read_oracle_values(self):
        run = deepcopy(self.results['hidden']['run'])
        before = compare_estimates(run)
        run['result']['oracle'].loc[:, :] = 999999
        pd.testing.assert_frame_equal(before, compare_estimates(run))

    def test_single_arm_cannot_produce_a_comparison(self):
        run = deepcopy(self.results['randomized']['run'])
        run['result']['data']['program'] = 1
        with self.assertRaisesRegex(ValueError, 'Both treatment groups'):
            compare_estimates(run)

    def test_invalid_predictions_and_unknown_lessons_are_rejected(self):
        for key, prediction in [('randomized', ''), ('hidden', 'adjusted'), ('missing', 'uncertain')]:
            with self.assertRaises(ValueError):
                run_lesson(self.trained, key, prediction)

    def test_student_record_does_not_reveal_truth_early(self):
        record = json.loads(experiment_record(self.results['measured'], 'My explanation.', reveal=False))
        self.assertFalse(record['truth_revealed'])
        self.assertNotIn('tau', record['settings'])
        for estimate in record['estimates']:
            self.assertEqual(set(estimate), {'method', 'estimate'})
        self.assertEqual(record['reflection'], 'My explanation.')

    def test_revealed_record_contains_correct_errors_and_reflection(self):
        record = json.loads(experiment_record(self.results['hidden'], 'Measured ≠ all causes.', reveal=True))
        self.assertEqual(record['reflection'], 'Measured ≠ all causes.')
        for estimate in record['estimates']:
            self.assertAlmostEqual(estimate['estimate'] - estimate['truth'], estimate['error'])

    def test_plot_only_includes_truth_after_reveal(self):
        for reveal in (False, True):
            fig = comparison_plot(self.results['measured']['comparison'], reveal=reveal)
            labels = [line.get_label() for line in fig.axes[0].lines]
            self.assertEqual(any(label.startswith('True cohort effect:') for label in labels), reveal)
            plt.close(fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
