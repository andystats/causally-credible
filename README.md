# Causally credible data demo (with app) 

## Based on the **Credence Framework**

**Generate semi-synthetic data with known causal effects to validate inference methods**

## Overview

The Credence framework (Parikh et al., ICML 2022) addresses a fundamental challenge in causal inference: how do you know your estimator works on real-world data when you never observe ground truth?

Traditional simulation approaches use hand-specified data generating processes with parametric assumptions that may inadvertently favor certain estimators. Credence takes a different approach by **separating realism from truth**:

| Concept | Description |
|---------|-------------|
| **Realism** | Learn covariate distributions p(X\|Z) from empirical data using a Conditional VAE |
| **Truth** | Inject known treatment effects (τ) and unmeasured confounding (ρ) |
| **Evidence** | Measure whether the realism is real, against held-out rows the model never saw |
| **Result** | Credible benchmarks that maintain realistic data structure while providing ground truth |

Note that only `p(X|Z)` is learned from your data. The outcome mechanism `f(X)` and
the effect `τ` are chosen by the analyst — so this buys realism in the covariates
and their relationship to treatment, not in the outcome model. That distinction is
worth keeping in view when interpreting any benchmark built this way.

## Live App

Try the app without any installation:

**[https://andystats.shinyapps.io/causally-credible_tutorial/](https://andystats.shinyapps.io/causally-credible_tutorial/)**

## Running Locally

### Requirements

- Python 3.12 (tested on macOS)

### Setup

```bash
# Clone the repository
git clone https://github.com/andystats/causally-credible.git
cd causally-credible

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r app/requirements-lock.txt

# Run the app
shiny run app/app.py
```

The app will be available at `http://localhost:8000`.

## Tutorial

See the [tutorial PDF](tutorial/tutorial.pdf) for a comprehensive walkthrough of the Credence framework, including:

- Why traditional simulations fall short
- The Credence pipeline: Learn → Generate → Compare
- Technical details on the Conditional VAE
- Hands-on exercises with the app

## Workflow

The app opens in **Guided lessons**, with an **Advanced workspace** for uploads and
custom experiments. Guided and advanced state are independent: changing an upload,
model setting, or simulation control cannot change a completed guided lesson.

### Guided lessons: predict, compare, explain

The worked example asks whether a fictional training program improves the average
change in a performance score. Outcomes are continuous score points; higher is
better. The source cohort is simulated too, so this example teaches the workflow
without making empirical claims about a real program.

| Lesson | Assignment mechanism | Teaching question |
|--------|----------------------|-------------------|
| Randomized assignment | λ = 0, ρ = 0 | Can a fair comparison still miss the true effect? |
| Measured confounding | λ = 1.25, ρ = 0 | Does adjustment help when groups differ in measured characteristics? |
| Hidden confounding | λ = 1.25, ρ = 1 | Can adjustment remove a common cause that was not measured? |

Choose a prediction, including “I'm not sure yet,” and run the experiment. The
first run trains the source-cohort generator; subsequent lessons reuse it. Compare
unadjusted and g-computation estimates, reveal the true cohort effect, and write an
explanation. The adjusted linear model includes treatment interactions and matches
the chosen outcome mechanism; it is a teaching reference, not a general-purpose
analysis recommendation. Plots show point estimates without confidence intervals.

The lessons share the measured covariates, 2,000-row sample size, effect, and
outcome-noise draws. Their effect is constant at +5 score points. Different
assignment rules change who receives the program; hidden confounding also changes
potential outcomes through U. These are single-cohort errors, not Monte Carlo bias.

Predictions, reflections, reveal choices, and completed results remain available
when moving between lessons in the same session. Refreshing the app starts a new
session. Download the experiment record to preserve a prediction and reflection;
it excludes truth, errors, and the effect parameter until truth is revealed. The
observed CSV always excludes oracle columns. The answer ZIP becomes available
after reveal. The reveal control supports a teaching exercise, not access control:
the source code and instructor documentation contain the truth.

The generator's held-out diagnostic is under “What is learned, what is chosen,
and what counts as evidence?” A good covariate match does not validate the chosen
causal mechanism. Repeated simulations and additional estimator families are
outside these first three lessons.

### Advanced workspace

The original controls remain available in separate navigation tabs:

1. **Load Data**: Upload your CSV or use built-in examples (LaLonde NSW, Pneumonia Vaccine)
2. **Learn Structure**: Train a Conditional VAE to model p(X|Z), holding out a slice for step 4
3. **Generate Data**: Choose a baseline effect, heterogeneity and unmeasured confounding; inspect the actual cohort ATE
4. **Compare & Export**: Compare unadjusted, adjusted, and true effects; check held-out realism; download observed data and a separate answer key
5. **Compare Scenarios**: Run a paired 12-cell grid with common covariate and noise draws

## Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| **τ₀ (baseline effect)** | Effect at the training-centered modifier value zero; equals the ATE only for constant effects | -100 to +100 |
| **ρ (rho)** | Strength of unmeasured confounding | 0.0 to 1.0 |
| **Training seed** | Seeds model initialisation and batch shuffling | nonnegative integer |
| **Generation seed** | Seeds the cohort draw, separately from training | nonnegative integer |

When ρ = 0, exchangeability holds given the exported covariates and consistent
estimators can target the actual cohort-average effect, subject to overlap and model
conditions. Hidden confounding can bias adjustment based only on the observed covariates.

The two seeds are deliberately separate: you can redraw a cohort without retraining,
or retrain without moving the cohort, and know which change caused what you see.

## Is the synthetic data actually realistic?

Step 4 measures it rather than claiming it, comparing synthetic covariates against
**held-out rows the generator never trained on**:

| Check | What it catches |
|-------|-----------------|
| **Standardized mean difference** | Are the means close? |
| **Kolmogorov–Smirnov distance** | Are the whole distributions close, not just the means? |
| **Discriminator AUC** | Can a random forest tell real rows from synthetic ones, using all covariates jointly? |

The discriminator is the strongest of the three, because it is the only one that
sees whole rows and can therefore notice broken *correlations* between covariates.
**AUC near 0.5** means this discriminator did not distinguish the samples under
this evaluation design. It does not prove equivalent distributions or validate
every dependence relevant to a downstream analysis.

## Heterogeneous effects, overlap, and the scenario grid

| Control | What it does |
|---------|--------------|
| **κ (heterogeneity)** | Makes the effect vary across patients: τ(X) = τ₀ + κ·σ_Y·m(X). The modifier is centered on training data, so the generated cohort mean may differ from τ₀. Both τ₀ and the actual ATE are reported. |
| **λ (overlap)** | The positivity dial. With ρ = 0, λ = 0 is a randomised trial on realistic covariates; 1 = real confounding with comfortable overlap; 3 = propensities piled up at 0 and 1. |
| **Mechanism** | *Learned* draws Z then X\|Z (confounding inherited from your data, not adjustable). *Designed* draws X then Z\|X (confounding and overlap become dials). You cannot have both from a conditional VAE — that tension is what CausalMix's overlap regulariser addresses. |

Constant effects still allow comparisons of finite-sample error and robustness.
Heterogeneity additionally enables comparison of individual-effect recovery.

**Step 5** crosses overlap × heterogeneity × hidden bias into a 12-cell factorial,
using common random draws for covariates, treatment assignment and outcome noise.
The table reports one-run estimation errors, not Monte Carlo bias. It is inspired
by the companion R study [`survcausal-pilot`](https://github.com/ishuryak/survcausal-pilot).

## Verification

```bash
python3 tests/test_tier0.py    # generator correctness and realism
python3 tests/test_tier1.py    # tau(X), the overlap dial, the factorial grid
python3 tests/test_correctness.py  # schema, treatment semantics, state and export regressions
python3 tests/test_lessons.py  # paired guided experiments, estimator comparison and truth reveal
```

Runs the full pipeline headlessly on both built-in datasets and checks, among other
things, that a correctly specified adjustment recovers τ when ρ = 0. If that fails,
the generator is wrong and nothing downstream can be trusted.

## Input and export contract

Numeric 0/1 treatment labels are preserved. Other labels require an explicit
choice of treated level. Confirmation validates complete cases, two treatment
arms, a numeric varying outcome, and at least 40 complete rows. Loading new data
or changing column roles clears the model and downstream results; training and
simulation controls invalidate the dependent stages.

One fitted schema is reused for training, holdout and decoding. A category absent
from holdout retains its training column. A category found only in holdout stops
training with a validation message: change the training seed/holdout fraction or
supply more data so training covers all levels. Categories are decoded as
reconstructed probabilities, not logits. This remains a post-processing
approximation to an MSE model; it does not guarantee fidelity.

`observed.csv` contains outcome, treatment and input covariates only.
The optional ZIP also contains `answer_key.csv` (counterfactuals, individual
truth and hidden confounder) and `manifest.json`. The files align by row order;
keep answer-key columns out of learner features. Uploaded names such as `U` or
`y0` remain ordinary covariates and cannot be overwritten by the separate oracle.

Designed-mode assignment probabilities are labelled `P(Z=1|X)` or `P(Z=1|X,U)`.
The latter is an oracle diagnostic. Learned-mode overlap given observed X is
reported as unavailable instead of incorrectly showing its marginal arm draw.

The October fixes change generated draws and the export contract. Older scripts
that expected oracle columns in the observed CSV should read the separate answer
key explicitly. Reproducibility is for the same code, inputs and runtime; pinning
versions is not a promise of bit-identical results on every hardware platform.

The hosted app is a separate deployment; opening or merging this PR does not
update that service.

## Development log

[`devlog/devlog.pdf`](devlog/devlog.tex) — a Beamer companion to the Module 5
tutorial that records how this app is being built, and teaches the Credence
framework, its evolution toward CausalMix, and the connection to `survcausal-pilot`
along the way. These August slides and the tutorial PDF are historical teaching
materials, not the current input/export contract. In particular, their claims
about a fixed ATE under heterogeneity and mixed oracle exports are superseded
by the definitions above. A broader teaching-material refresh is deferred.
Build with `latexmk -pdf devlog.tex`.

## References

- Parikh H, Vajao C, Xu L, Tchetgen Tchetgen E (2022). [Validating Causal Inference Methods](https://arxiv.org/abs/2202.04208). *ICML*.
- Kingma DP, Welling M (2014). Auto-Encoding Variational Bayes. *ICLR*.

## Author

**Andy Wilson**
[wilson.stats@gmail.com](mailto:wilson.stats@gmail.com) | [tao-rwd.com](https://tao-rwd.com)
