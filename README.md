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

- Python 3.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/andystats/causally-credible.git
cd causally-credible

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt

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

The app follows a 4-step process:

1. **Load Data**: Upload your CSV or use built-in examples (LaLonde NSW, Pneumonia Vaccine)
2. **Learn Structure**: Train a Conditional VAE to model p(X|Z), holding out a slice for step 4
3. **Generate Data**: Create synthetic data with known τ (true effect) and ρ (unmeasured confounding)
4. **Analyze & Export**: Check realism against the holdout, compare naive estimates to ground truth, download

## Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| **τ (tau)** | True average treatment effect to inject | -10 to +10 |
| **ρ (rho)** | Strength of unmeasured confounding | 0.0 to 1.0 |
| **Training seed** | Seeds model initialisation and batch shuffling | any integer |
| **Generation seed** | Seeds the cohort draw, separately from training | any integer |

When ρ = 0, exchangeability holds given the exported covariates and consistent
estimators should recover τ. As ρ increases, unmeasured confounding biases all
methods—useful for sensitivity analysis.

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
**AUC ≈ 0.5 is the target** — it means the generator has fooled it completely. This
is the rare model you want to perform badly.

## Heterogeneous effects, overlap, and the scenario grid

| Control | What it does |
|---------|--------------|
| **κ (heterogeneity)** | Makes the effect vary across patients: τ(X) = τ₀ + κ·σ_Y·m(X). Because m is centred, **E[τ(X)] = τ₀** — turning heterogeneity on changes the spread, never the average. |
| **λ (overlap)** | The positivity dial. 0 = a randomised trial on realistic covariates; 1 = real confounding with comfortable overlap; 3 = propensities piled up at 0 and 1. |
| **Mechanism** | *Learned* draws Z then X\|Z (confounding inherited from your data, not adjustable). *Designed* draws X then Z\|X (confounding and overlap become dials). You cannot have both from a conditional VAE — that tension is what CausalMix's overlap regulariser addresses. |

Constant τ is why the earlier version could not tell TMLE, AIPW and g-computation
apart: with an identical effect for everyone they all recover the same number.

**Step 5** crosses overlap × heterogeneity × hidden bias into a 12-cell factorial,
holding everything else fixed so each difference has one cause — the design of the
companion R study [`survcausal-pilot`](https://github.com/ishuryak/survcausal-pilot).

## Verification

```bash
python3 tests/test_tier0.py    # generator correctness and realism
python3 tests/test_tier1.py    # tau(X), the overlap dial, the factorial grid
```

Runs the full pipeline headlessly on both built-in datasets and checks, among other
things, that a correctly specified adjustment recovers τ when ρ = 0. If that fails,
the generator is wrong and nothing downstream can be trusted.

## Development log

[`devlog/devlog.pdf`](devlog/devlog.tex) — a Beamer companion to the Module 5
tutorial that records how this app is being built, and teaches the Credence
framework, its evolution toward CausalMix, and the connection to `survcausal-pilot`
along the way. Build with `latexmk -pdf devlog.tex`.

## References

- Parikh H, Vajao C, Xu L, Tchetgen Tchetgen E (2022). [Validating Causal Inference Methods](https://arxiv.org/abs/2202.04208). *ICML*.
- Kingma DP, Welling M (2014). Auto-Encoding Variational Bayes. *ICLR*.

## Author

**Andy Wilson**
[wilson.stats@gmail.com](mailto:wilson.stats@gmail.com) | [tao-rwd.com](https://tao-rwd.com)
