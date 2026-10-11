"""
Built-in example datasets.

Pulled out of app.py so that the verification gate (tests/test_tier0.py) can run
the full pipeline on exactly the data the workshop will use, without starting a
web server. Anything the tests cannot reach is effectively untested, and the two
example datasets are the two paths every participant takes.

Each loader returns (data_frame, note). The note is displayed to the user and
exists because one of these datasets is not what its name suggests -- see below.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Every example is generated at a fixed seed. Previously the LaLonde fallback used
# unseeded numpy draws, so "load example" produced a different dataset on every
# click and nothing downstream could be reproduced.
EXAMPLE_SEED = 20260802


def load_lalonde():
    """
    LaLonde NSW, if the CSV is present.

    IMPORTANT AND SLIGHTLY EMBARRASSING: app/lalonde_nsw.csv is not in the
    repository. The original code silently fell back to 100 rows of unseeded
    random noise and still labelled it "LaLonde NSW" in the UI, so a participant
    could spend the whole exercise drawing conclusions from pure noise while
    believing they were looking at a famous evaluation dataset.

    We keep the fallback -- the app must still run -- but it now says what it is.
    Fabricating a stand-in and calling it LaLonde would be the same error again.
    """
    path = Path(__file__).parent / "lalonde_nsw.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df.rename(columns={"treat": "treatment", "re78": "outcome"})
        return df, "Loaded LaLonde NSW from {}.".format(path.name)

    rng = np.random.default_rng(EXAMPLE_SEED)
    n = 300
    df = pd.DataFrame({
        'treatment': rng.binomial(1, 0.5, n),
        'outcome': rng.normal(5000, 1000, n),
        'age': rng.integers(18, 65, n),
        'education': rng.integers(8, 18, n),
    })
    note = ("NOT LaLonde. app/lalonde_nsw.csv is missing, so this is a seeded "
            "placeholder with no real-world meaning. Add the CSV to use the real "
            "dataset.")
    return df, note


def load_pneumonia():
    """
    The pneumococcal-vaccine teaching cohort from Module 4, regenerated here so
    Module 5 continues the same worked example.

    Confounding is built in on purpose: age drives prior pneumonia, prior vaccine
    and treatment. The `outcome` column is a placeholder -- Step 3 replaces it
    with simulated potential outcomes, and only its standard deviation is used.

    This is a teaching example, not clinical evidence about pneumococcal vaccination.
    """
    rng = np.random.default_rng(EXAMPLE_SEED)
    n = 1000

    age = rng.normal(65, 15, n)
    prior_pneumonia = rng.binomial(1, 1 / (1 + np.exp(-(-2 + 0.03 * age))), n)
    prior_vaccine = rng.binomial(
        1, 1 / (1 + np.exp(-(-1 + 0.02 * age + 0.5 * prior_pneumonia))), n)
    treatment = rng.binomial(
        1, 1 / (1 + np.exp(-(-2.5 + 0.025 * age + 0.8 * prior_pneumonia
                             + 1.2 * prior_vaccine))), n)
    outcome = rng.normal(0, 1, n)

    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'age': age,
        'priorPneumonia': prior_pneumonia,
        'priorVaccine': prior_vaccine,
    })
    return df, "Loaded the Module 4 pneumonia teaching cohort ({} rows).".format(n)


LOADERS = {
    'lalonde': load_lalonde,
    'pneumonia': load_pneumonia,
}


def load_example(name):
    """Look up and run a loader by key. Returns (data_frame, note)."""
    if name not in LOADERS:
        raise KeyError("unknown example dataset: {}".format(name))
    return LOADERS[name]()
