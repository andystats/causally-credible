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
| **Result** | Credible benchmarks that maintain realistic data structure while providing ground truth |

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
2. **Learn Structure**: Train a Conditional VAE to model p(X|Z)
3. **Generate Data**: Create synthetic data with known τ (true effect) and ρ (unmeasured confounding)
4. **Analyze & Export**: Compare naive estimates to ground truth, download for further analysis

## Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| **τ (tau)** | True average treatment effect to inject | -10 to +10 |
| **ρ (rho)** | Strength of unmeasured confounding | 0.0 to 1.0 |

When ρ = 0, exchangeability holds and consistent estimators should recover τ. As ρ increases, unmeasured confounding biases all methods—useful for sensitivity analysis.

## References

- Parikh H, Vajao C, Xu L, Tchetgen Tchetgen E (2022). [Validating Causal Inference Methods](https://arxiv.org/abs/2202.04208). *ICML*.
- Kingma DP, Welling M (2014). Auto-Encoding Variational Bayes. *ICLR*.

## Author

**Andy Wilson**
[wilson.stats@gmail.com](mailto:wilson.stats@gmail.com) | [tao-rwd.com](https://tao-rwd.com)
