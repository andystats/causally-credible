"""Fixed, paired teaching experiments. No Shiny state or student answers live here."""
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from comparison import compare_estimates
from workflow import prepare_data, fit_experiment, generate_experiment


@dataclass(frozen=True)
class Lesson:
    key: str
    title: str
    question: str
    change: str
    prediction: str
    explanation: str
    reflection: str
    overlap: float
    hidden: float


LESSONS = {
    'randomized': Lesson(
        'randomized', 'Randomized assignment',
        'Can a fair comparison still miss the true effect?',
        'Everyone has the same chance of receiving the program. No measured or hidden confounding.',
        'Which estimate do you expect to be closer to the truth?',
        'Random assignment removes systematic confounding. Both estimates can still miss the '
        'truth in a finite cohort. Adjustment need not be the closer estimate in every run.',
        'Why can an estimate differ from truth even when treatment is randomized?', 0., 0.),
    'measured': Lesson(
        'measured', 'Measured confounding',
        'What changes when the groups start out different?',
        'Compared with lesson 1, assignment now depends on the measured characteristics. '
        'Hidden confounding remains off.',
        'Will adjusting for the measured characteristics help?',
        'Measured characteristics now influence both treatment and outcomes. G-computation '
        'compares predicted outcomes under both treatments for the same cohort. Here its model '
        'matches the chosen outcome mechanism, so adjustment can address this measured confounding.',
        'What information allows adjustment to improve this comparison?', 1.25, 0.),
    'hidden': Lesson(
        'hidden', 'Hidden confounding',
        'Can adjustment fix a cause we did not measure?',
        'Compared with lesson 2, a hidden factor now affects assignment and outcomes. '
        'The measured assignment rule stays the same.',
        'Will the adjusted estimate still recover the effect?',
        'The hidden factor affects both treatment and outcomes, but is excluded from both '
        'estimators. Adjusting for the measured characteristics cannot generally remove this '
        'source of confounding. More data alone would not remove it either.',
        'What would you need to change about the study, rather than just the estimator?', 1.25, 1.),
}

PREDICTIONS = {
    '': 'Choose a prediction…',
    'unadjusted': 'The unadjusted estimate will be closer',
    'adjusted': 'The adjusted estimate will be closer',
    'similar': 'They will be similarly close',
    'recover': 'Adjustment will recover the effect',
    'residual': 'Adjustment will help, but leave confounding error',
    'unchanged': 'Neither estimate will be affected',
    'uncertain': 'I’m not sure yet',
}


def prediction_choices(key):
    keys = (['', 'recover', 'residual', 'unchanged', 'uncertain'] if key == 'hidden'
            else ['', 'unadjusted', 'adjusted', 'similar', 'uncertain'])
    return {key: PREDICTIONS[key] for key in keys}


def teaching_cohort():
    """A fictional training-program study with continuous, explicitly named units."""
    rng = np.random.default_rng(20261010)
    n = 1200
    baseline = rng.normal(60, 10, n)
    experience = rng.integers(0, 21, n)
    prior = rng.binomial(1, .4, n)
    logit = -.2 + .04 * (baseline - 60) + .05 * (experience - 10) + .5 * prior
    return pd.DataFrame({
        'program': rng.binomial(1, 1 / (1 + np.exp(-logit))),
        # Used only to set the scale of the simulated outcome, not to fit its mechanism.
        'score_change': rng.normal(0, 10, n),
        'baseline_score': baseline,
        'experience_years': experience,
        'prior_training': prior,
    })


def prepare_lessons(epochs=150):
    prepared = prepare_data(teaching_cohort(), 'score_change', 'program')
    return fit_experiment(prepared, seed=1, holdout_frac=.25, epochs=epochs, batch_size=128)


def run_lesson(trained, key, prediction):
    if key not in LESSONS:
        raise ValueError('Choose one of the three guided lessons.')
    if prediction not in prediction_choices(key) or not prediction:
        raise ValueError('Make a prediction first. “I’m not sure yet” is a useful starting point too.')
    lesson = LESSONS[key]
    run = generate_experiment(trained, n=2000, tau0=5, seed=7,
                              mechanism='designed', overlap_strength=lesson.overlap,
                              rho=lesson.hidden, heterogeneity=0)
    return {'lesson': key, 'prediction': prediction, 'run': run,
            'comparison': compare_estimates(run)}


def experiment_record(result, reflection='', reveal=False):
    """A learner record; hidden truth stays out until explicitly revealed."""
    columns = ['method', 'estimate'] + (['error', 'truth'] if reveal else [])
    parameters = dict(result['run']['parameters'])
    if not reveal:
        parameters.pop('tau', None)
    record = {
        'lesson': LESSONS[result['lesson']].title,
        'source': 'Fictional simulated training-program cohort; not empirical evidence.',
        'outcome': 'Change in performance score, in points. Higher is better.',
        'estimand': 'Average effect of program versus no program in this generated cohort.',
        'prediction': PREDICTIONS[result['prediction']], 'reflection': reflection,
        'truth_revealed': reveal,
        'estimates': result['comparison'][columns].to_dict(orient='records'),
        'settings': parameters,
        'interpretation': 'Point estimates from one cohort; errors are not Monte Carlo bias.',
    }
    return json.dumps(record, indent=2, ensure_ascii=False)
