# Tao of RWD · Causal inference lab

Learn how study design changes causal estimates: predict a result, run a simulation,
compare estimates with known truth, and explain what changed.

The app adapts the **Credence framework** (Parikh et al., 2022). A conditional VAE
learns covariate patterns; assignment, outcomes, and treatment effects are specified
by the simulation. Held-out checks assess covariate fidelity, not the validity of
the chosen causal mechanism.

## Guided lessons

A fictional training program provides a continuous outcome: change in performance
score, in points. Higher is better. The source cohort is also simulated; these
results are teaching examples, not evidence about a real program.

| Lesson | Assignment | Question |
|--------|------------|----------|
| Randomized assignment | λ = 0, ρ = 0 | Can a fair comparison miss the true effect? |
| Measured confounding | λ = 1.25, ρ = 0 | Does adjustment help when measured characteristics differ? |
| Hidden confounding | λ = 1.25, ρ = 1 | Can adjustment fix a cause we did not measure? |

Choose a prediction and run the experiment. Compare unadjusted and g-computation
estimates, reveal the true effect, and record an explanation. Adjustment uses a
linear outcome model with treatment interactions that matches the chosen outcome
mechanism. Plots show point estimates without confidence intervals.

Each lesson reuses the source model, 2,000 measured-covariate draws, and outcome
noise, with a constant effect of +5 points. Assignment rules change; hidden
confounding also changes potential outcomes through U. Errors describe one cohort,
not bias across repeated studies.

Results and notes persist while switching lessons within a session. Download the
experiment record to keep them after closing or refreshing the app. Before truth
is revealed, the record omits truth, errors, and the effect parameter. The observed
CSV excludes oracle columns; the answer ZIP becomes available after reveal.
Reveal is a teaching device, not access control: the source and this documentation
include the true effect.

Open **How the simulation works** for generator diagnostics and assumptions.

## Advanced workspace

Custom experiments have separate state from guided lessons.

1. **Data:** Upload a CSV or load an example. The pneumonia cohort is simulated;
   the LaLonde option uses a clearly labelled placeholder unless its CSV is supplied.
2. **Train:** Fit the conditional VAE, reserving holdout rows for diagnostics.
3. **Generate:** Set the effect, assignment mechanism, overlap, heterogeneity,
   hidden confounding, cohort size, and generation seed.
4. **Compare & export:** Inspect estimates, true effects, and held-out diagnostics;
   download observed data and a separate answer key.
5. **Scenario grid:** Compare 12 combinations of overlap, heterogeneity, and hidden
   confounding with common random draws.

**How it works** contains equations, notation, and suggestions for extending the
estimator comparison.

| Setting | Meaning |
|---------|---------|
| τ₀ | Baseline effect at the training-centered modifier value zero. Equals the cohort ATE when effects are constant. |
| κ | Effect heterogeneity: τ(X) = τ₀ + κ·σY·m(X). The generated cohort ATE can differ from τ₀. |
| λ | Strength of measured covariates in designed treatment assignment. Larger values can reduce overlap. |
| ρ | Strength of the hidden cause shared by treatment and outcomes. |
| Learned assignment | Draw Z, then X\|Z. Arm-specific patterns come from the fitted generator. |
| Designed assignment | Draw X, then Z\|X,U. Changing assignment changes arm-specific covariate distributions. |
| Training / generation seeds | Retrain the model or redraw the cohort independently. |

With ρ = 0, exchangeability holds given the exported covariates; identification
and estimation still require overlap and suitable model conditions. Adjustment
on measured covariates does not generally remove hidden confounding.

Diagnostics compare generated covariates with held-out source rows: SMD measures
mean differences, KS distance measures distributional differences, and a
cross-validated discriminator uses whole rows. AUC near 0.5 means this diagnostic
did not distinguish the samples; it does not prove equivalent distributions.

## Run locally

Python 3.12 is tested on macOS.

```bash
git clone https://github.com/andystats/causally-credible.git
cd causally-credible
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements-lock.txt
shiny run app/app.py
```

Open `http://localhost:8000`.

The [hosted app](https://andystats.shinyapps.io/causally-credible_tutorial/) is a
separate deployment. Pushing or merging changes here does not update it.

## Input and export contract

Numeric 0/1 treatment labels retain their meaning. Other labels require an
explicit treated-level choice. Confirmation requires two arms, a numeric varying
outcome, and at least 40 complete rows. New data, changed column roles, or changed
settings invalidate the affected downstream stages.

One fitted schema serves training, holdout, and decoding. A category found only
in holdout stops training: change the seed or holdout fraction, or provide more
data. Category decoding is a post-processing approximation to an MSE model and
does not guarantee fidelity.

Downloads use the `tao_rwd_` prefix. In answer bundles:

- `observed.csv` contains outcome, treatment, and input covariates.
- `answer_key.csv` contains potential outcomes, individual effects, and the hidden
  confounder. Keep these columns out of estimator inputs.
- `manifest.json` records the simulation settings. Files align by row order.

Uploaded names such as `U` or `y0` remain ordinary covariates. Designed assignment
probabilities are labelled `P(Z=1|X)` or oracle `P(Z=1|X,U)`; learned-mode overlap
given observed X is reported as unavailable.

Older scripts that expected oracle columns in the observed CSV must read the
separate answer key. Reproducibility requires the same code, inputs, and runtime;
pinned dependencies do not guarantee identical results across hardware.

## Verification

```bash
python tests/test_tier0.py
python tests/test_tier1.py
python tests/test_correctness.py
python tests/test_lessons.py
```

The suites check generator behavior, effect heterogeneity, overlap, schema and
state handling, exports, and the paired guided lessons.

## Background materials

The [tutorial](tutorial/tutorial.pdf) and [development log](devlog/devlog.pdf)
retain historical content under Tao of RWD branding. Some features, equations,
and export examples describe earlier versions; use the contract above for the
current app. The [development-log source](devlog/devlog.tex) builds with
`latexmk -pdf devlog.tex` from its directory.

## References and author

- Parikh H, Vajao C, Xu L, Tchetgen Tchetgen E (2022).
  [Validating Causal Inference Methods](https://arxiv.org/abs/2202.04208). ICML.
- Kingma DP, Welling M (2014). Auto-Encoding Variational Bayes. ICLR.

Andy Wilson · [Tao of RWD](https://tao-rwd.com) ·
[wilson.stats@gmail.com](mailto:wilson.stats@gmail.com)
