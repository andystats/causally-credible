"""
Credence Tutorial - instats SSC 2026 Workshop
Module 5: Validating Causal Methods with Semi-Synthetic Data

A Python Shiny app for generating semi-synthetic data with known causal
effects via a Conditional VAE, and guiding the user on how to analyze
the data to recover the true effect.

Based on Parikh et al. (2022) "Validating Causal Inference Methods" (ICML).
"""

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# The app is deliberately thin. The three things worth understanding live in
# their own modules so they can be read -- and tested -- without Shiny in the way:
#   cvae_model.py  learns p(X | Z)                       (realism)
#   dgp.py         chooses tau, rho and f(X)             (truth)
#   realism.py     measures whether the realism is real  (evidence)
# tests/test_tier0.py runs all three headlessly.
from cvae_model import ConditionalVAE, encode_data, set_seed
from dgp import PrognosticIndex, generate_cohort
from examples import load_example
from realism import realism_report

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Custom CSS with instats branding and MathJax support
custom_css = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">

<script>
  MathJax = {
    tex: {
      inlineMath: [['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$']],
      processEscapes: true
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<script>
  // Aggressively prevent unwanted scrolling
  (function() {
    let savedScrollPos = 0;
    let lockScroll = false;

    function saveScroll() {
      if (!lockScroll) {
        savedScrollPos = window.pageYOffset || document.documentElement.scrollTop;
      }
    }

    function lockAndRestore() {
      lockScroll = true;
      const targetScroll = savedScrollPos;

      // Restore immediately and repeatedly
      for (let i = 0; i <= 200; i += 10) {
        setTimeout(function() {
          window.scrollTo(0, targetScroll);
        }, i);
      }

      // Unlock after animations settle
      setTimeout(function() {
        lockScroll = false;
      }, 250);
    }

    document.addEventListener('DOMContentLoaded', function() {
      // Save scroll position constantly
      window.addEventListener('scroll', saveScroll, {passive: true});

      // Intercept file input clicks
      document.addEventListener('click', function(e) {
        saveScroll();

        // Check if clicking on or near a file input
        let target = e.target;
        for (let i = 0; i < 5; i++) {
          if (!target) break;
          if (target.tagName === 'INPUT' && target.type === 'file') {
            lockAndRestore();
            break;
          }
          if (target.querySelector && target.querySelector('input[type="file"]')) {
            lockAndRestore();
            break;
          }
          target = target.parentElement;
        }
      }, true);

      // Also catch focus events
      document.addEventListener('focus', function(e) {
        if (e.target.type === 'file' || e.target.closest('input[type="file"]')) {
          lockAndRestore();
        }
      }, true);

      // Catch actual scroll events and prevent them during lock
      window.addEventListener('scroll', function() {
        if (lockScroll) {
          window.scrollTo(0, savedScrollPos);
        }
      }, {passive: false});
    });
  })();
</script>

<style>
    :root {
        --instats-blue: #4A9EFF;
        --instats-accent: #4A9EFF;
        --instats-orange: #fd7e14;
        --instats-green: #1c8b4a;
        --cc-muted: #6c757d;
        --cc-border: #e5e5ea;
        --cc-bg: #ffffff;
    }
    html {
        overflow-anchor: none;
    }
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans';
        letter-spacing: 0;
        overflow-anchor: none;
    }
    /* Prevent focus from scrolling */
    * {
        scroll-margin-top: 0;
        scroll-margin-bottom: 0;
    }
    input:focus, button:focus, select:focus {
        scroll-margin: 0;
    }
    .container-fluid { max-width: 980px; }
    .instats-header {
        background: linear-gradient(135deg, var(--instats-blue), #3a7fcf);
        color: white;
        padding: 20px 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 1.8em;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2em;
        color: white;
    }
    .app-subtitle {
        color: rgba(255,255,255,0.9);
        margin-bottom: 0;
        font-size: 1em;
    }
    .workshop-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .cc-card {
        background: var(--cc-bg);
        border: 1px solid var(--cc-border);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .step-panel {
        background: white;
        border: 1px solid var(--cc-border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #dee2e6;
    }
    .step-active {
        border-left-color: var(--instats-orange);
    }
    .step-complete {
        border-left-color: var(--instats-green);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        color: var(--cc-muted);
        font-size: 0.9em;
    }
    details {
        border: 1px solid var(--cc-border);
        border-radius: 12px;
        padding: 12px 14px;
        background: #fff;
        margin-top: 10px;
    }
    details summary {
        cursor: pointer;
        font-weight: 600;
    }
    .math-note {
        font-size: 0.95em;
        color: #222;
        padding: 10px 0;
    }
    code, pre {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        background: #f5f5f5;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .btn-primary {
        background-color: var(--instats-blue);
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .btn-primary:hover {
        background-color: #3a7fcf;
    }
    .btn-success {
        background-color: var(--instats-green);
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .module-ref {
        background: #e8f4ff;
        border-left: 3px solid var(--instats-blue);
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95em;
    }
</style>
"""

# UI Definition
app_ui = ui.page_fluid(
    ui.HTML(custom_css),

    # instats Header
    ui.tags.div(
        {"class": "instats-header"},
        ui.tags.span({"class": "workshop-badge"}, "instats SSC 2026 Workshop"),
        ui.tags.div({"class": "app-title"}, "Module 5: Credence Framework Tutorial"),
        ui.tags.p(
            {"class": "app-subtitle"},
            "Generate realistic synthetic data with known ground truth to validate causal inference methods"
        ),
    ),

    # Overview Panel
    ui.tags.div(
        {"class": "cc-card"},
        ui.h3("Overview"),
        ui.p(
            "This tutorial follows the Credence framework (Parikh et al., ICML 2022) to build "
            "semi-synthetic datasets anchored at the empirical distribution, with a known causal contrast."
        ),
        ui.tags.ul(
            ui.tags.li(
                ui.strong("Observed confounding:"),
                " learned from the empirical relationship ",
                ui.tags.code("p(X|Z)"),
                ui.HTML(". Only \\(p(X\\mid Z)\\) is learned from your data &mdash; the outcome "
                        "mechanism and the effect are chosen by you in Step 3. How closely the "
                        "generated covariates match the real ones is <strong>measured in "
                        "Step 4</strong>, against rows the model never saw, rather than asserted "
                        "here.")
            ),
            ui.tags.li(
                ui.strong("Unobserved confounding:"),
                ui.HTML(" injected via a latent variable "),
                ui.tags.code("U"),
                ui.HTML(" with tunable strength \\((\\rho)\\).")
            ),
            ui.tags.li(
                ui.strong("Ground truth:"),
                ui.HTML(" you set the Average Treatment Effect (ATE) \\((\\tau)\\); Credence also supports heterogeneous effects \\((\\tau(X))\\) and bias functions.")
            ),
        ),

        # Connection to Module 4
        ui.tags.div(
            {"class": "module-ref"},
            ui.HTML("<strong>From Module 4:</strong> We used <code>simcausal</code> to generate data with parametric assumptions. "
                    "Here, we use a CVAE to learn the <em>actual</em> covariate structure from real data, "
                    "making our synthetic benchmarks more realistic for validating TMLE and other methods.")
        ),

        # Math Notes Details
        ui.tags.details(
            ui.tags.summary("Math notes"),
            ui.tags.div(
                {"class": "math-note"},
                ui.HTML("\\(\\newcommand{\\E}{\\mathbb{E}}\\)"),
                ui.p(ui.HTML("We use a Conditional Variational Autoencoder (VAE) to learn a deep generative model of the covariates, specifically modeling the complex distribution \\(p(X \\mid Z)\\). This VAE has two main parts:")),
                ui.tags.ul(
                    ui.tags.li(ui.HTML("<strong>The Encoder's Job (\\(q_\\phi(u \\mid x,z)\\)): Data → Latent Space</strong> <br> The encoder network is responsible for 'encoding' the data. It takes the original, high-dimensional <strong>covariates \\((x)\\)</strong> for a given treatment group \\((z)\\) and compresses this rich information into a new, lower-dimensional space called the <strong>latent space</strong>. The encoded representation of the data within this space is <strong>\\((u)\\)</strong>. The encoder's primary job is to find the most efficient summary of the data by learning a probability distribution for \\(u\\).")),
                    ui.tags.li(ui.HTML("<strong>The Decoder's Job (\\(p_\\theta(x \\mid u,z)\\)): Latent Space → Data</strong> <br> The decoder network acts as the 'recipe' for reversing the process. It takes a point \\((u)\\) sampled from the latent space and translates it back into the original data format, reconstructing (or generating new) high-dimensional covariates \\((x)\\). This reconstruction is the decoder's job, effectively taking a compressed summary and turning it back into realistic data."))
                ),
                ui.p(ui.HTML("The VAE is trained by optimizing the ELBO (Evidence Lower Bound) loss function. This objective balances two critical goals: ensuring the decoded data is a faithful <strong>reconstruction</strong> of the original, while also <strong>regularizing</strong> the latent space to be smooth and well-behaved, which is crucial for generating new data.")),
                ui.p("To create a simulation with a known 'ground truth' answer, we can then define the rules for this synthetic world:"),
                ui.tags.ul(
                    ui.tags.li(ui.HTML("<strong>Unobserved Confounding:</strong> We can specify how an unobserved variable \\(U\\) influences treatment selection, for instance: \\(\\Pr(Z=1 \\mid U) = \\text{logistic}(\\alpha + \\rho U)\\).")),
                    ui.tags.li(ui.HTML("<strong>Potential Outcomes:</strong> This same confounder can also affect the outcomes. We can define potential outcomes as \\(Y(0) = f(X) + \\gamma U + \\varepsilon\\) and \\(Y(1)=Y(0)+\\tau\\). By design, the true Average Treatment Effect (ATE) we want the causal methods to find is simply \\(\\E[Y(1){-}Y(0)] = \\tau\\)."))
                )
            )
        ),

        # ELBO Intuition Details
        ui.tags.details(
            ui.tags.summary("ELBO Intuition"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p("The ELBO (Evidence Lower Bound) balances two goals:"),
                ui.tags.ul(
                    ui.tags.li("Reconstruction: How well the decoder recreates the input data."),
                    ui.tags.li("Regularization: Ensures the latent space follows a standard normal distribution.")
                ),
                ui.p(ui.HTML("Mathematically: $$\\mathscr{L}(\\theta,\\phi) = \\mathbb{E}_{q_\\phi(u \\mid x,z)}[\\log p_\\theta(x \\mid u,z)] - \\mathrm{KL}(q_\\phi(u \\mid x,z) \\Vert p(u))$$")),
                ui.p("Maximizing ELBO helps the model learn realistic data generation while keeping latent representations smooth and generalizable.")
            )
        ),

        # Notation Details
        ui.tags.details(
            ui.tags.summary("Notation"),
            ui.tags.ul(
                ui.tags.li(ui.HTML("\\(X\\): covariates; \\(Z\\): binary treatment; \\(Y\\): outcome; \\(Y(0), Y(1)\\): potential outcomes.")),
                ui.tags.li(ui.HTML("\\(U\\): unobserved confounder; \\(f(X)\\): baseline outcome function.")),
                ui.tags.li(ui.HTML("\\(\\tau\\): Average Treatment Effect (ATE); \\(\\alpha\\): treatment-logit intercept; \\(\\rho\\): strength of unobserved confounding.")),
                ui.tags.li(ui.HTML("Encoder \\((q_\\phi(u \\mid x,z))\\), Decoder \\((p_\\theta(x \\mid u,z))\\); ELBO \\((\\mathscr{L}(\\theta,\\phi))\\)."))
            )
        ),
    ),

    # Step 1: Load Data
    ui.tags.div(
        {"class": "step-panel step-active", "id": "step1"},
        ui.h3("Step 1: Load Observational Data"),
        ui.p(ui.HTML("Upload a CSV to anchor the covariate structure \\((X)\\) and treatment \\((Z)\\). This provides realistic dependence patterns for the generator.")),

        ui.tags.details(
            ui.tags.summary("What and why"),
            ui.p(
                "You select an outcome ", ui.tags.code("Y"), " and a binary treatment ", ui.tags.code("Z"),
                ". All other variables are treated as covariates ", ui.tags.code("X"), "."
            ),
            ui.p(
                "We learn ", ui.tags.code("p(X|Z)"), " to anchor realistic covariate distributions conditional on ",
                ui.tags.code("Z"), ". This captures observed differences in ", ui.tags.code("X"),
                " between treated and control groups (the observed confounding structure). We do not fit an outcome model here; outcomes are simulated later via ",
                ui.tags.code("f(X)"), ", ", ui.tags.code("τ"), ", and ", ui.tags.code("U"), "."
            ),
            ui.p(
                ui.strong("Expectations:"), " ", ui.tags.code("Z"), " must be binary; we perform complete-case analysis (rows with missing values are dropped) for simplicity; ",
                ui.tags.code("Y"), " is treated as numeric; factors are supported in ", ui.tags.code("X"), "."
            ),
            ui.p(
                ui.strong("Note:"), " The Credence framework supports effect heterogeneity and bias functions of covariates. This app uses a constant ",
                ui.tags.code("τ"), " and a simple ", ui.tags.code("f(X)"), " for clarity."
            )
        ),

        ui.row(
            ui.column(
                8,
                ui.input_file("csv_file", "Upload Your CSV", accept=[".csv"], multiple=False)
            ),
            ui.column(
                4,
                ui.input_select("example_dataset", "Or load example:",
                              choices={"": "None", "lalonde": "LaLonde NSW", "pneumonia": "Pneumonia Vaccine (M4)"}),
                ui.input_action_button("load_example", "Load Example", class_="btn-secondary")
            )
        ),
        ui.output_ui("example_note"),
        ui.output_ui("csv_controls"),
        ui.input_action_button("process_data", "Confirm Data", class_="btn-primary"),
        ui.output_text_verbatim("data_summary")
    ),

    # Step 2: Learn Model
    ui.tags.div(
        {"class": "step-panel", "id": "step2"},
        ui.h3("Step 2: Learn Data Structure"),
        ui.p(ui.HTML("Train a Conditional VAE to learn \\((p(X\\mid Z))\\), capturing how covariates shift across treatment groups (observed confounding).")),

        ui.tags.details(
            ui.tags.summary("Under the hood"),
            ui.p(ui.HTML("We fit an encoder \\((q_\\phi(u \\mid x,z))\\) and decoder \\((p_\\theta(x \\mid u,z))\\). Training maximizes the ELBO (reconstruction + regularization). This lets us later sample new \\((X)\\) consistent with a chosen \\((Z)\\).")),
            ui.p("Interpretation: If treated and control have different covariate patterns, the model learns those differences, so generated data inherit realistic observed bias.")
        ),

        ui.tags.details(
            ui.tags.summary("Why the batch size matters (and a bug this app used to have)"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p(ui.HTML(
                    "Training takes <strong>one gradient step per batch</strong>. With "
                    "<code>batch size = full dataset</code>, an epoch is a single step, so "
                    "50 epochs means 50 steps in total &mdash; nowhere near convergence. "
                    "That is what this app did until now, while telling you the output was "
                    "&ldquo;near-indistinguishable&rdquo; from your data.")),
                ui.p(ui.HTML(
                    "Set the batch size to the full dataset and train for a few epochs to "
                    "<em>reproduce the old behaviour</em>, then compare the realism score in "
                    "Step 4. The failure mode is more convincing when you can see it.")),
                ui.p(ui.HTML(
                    "The <strong>holdout</strong> is set aside before training and never shown "
                    "to the model. Step 4 compares synthetic data against those unseen rows, "
                    "because a flexible generator scored against its own training data can win "
                    "by memorising &mdash; which is both a meaningless result and a privacy risk."))
            )
        ),

        ui.row(
            ui.column(3, ui.input_slider("epochs", "Training Epochs",
                                         min=1, max=400, value=150, step=1)),
            ui.column(3, ui.input_select("batch_size", "Batch Size",
                                         choices={"32": "32", "64": "64", "128": "128",
                                                  "256": "256", "0": "Full dataset (the old bug)"},
                                         selected="128")),
            ui.column(3, ui.input_slider("holdout_frac", "Holdout Fraction",
                                         min=0.1, max=0.5, value=0.25, step=0.05)),
            ui.column(3, ui.input_numeric("train_seed", "Training Seed", value=1, min=0, step=1))
        ),
        ui.input_action_button("learn_model", "Train CVAE Model", class_="btn-primary"),
        ui.output_text_verbatim("training_output"),
        ui.output_plot("elbo_plot")
    ),

    # Step 3: Generate Data
    ui.tags.div(
        {"class": "step-panel", "id": "step3"},
        ui.h3("Step 3: Generate Biased Synthetic Data"),
        ui.p("Create new data using the learned structure. You set the ground-truth effect (ATE) and inject unmeasured confounding via a latent ", ui.tags.code("U"), "."),

        ui.tags.details(
            ui.tags.summary("Data-generating process"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p(ui.HTML("Assignment: \\((\\Pr(Z{=}1 \\mid U) = \\text{logistic}(\\alpha + \\rho U))\\). Larger \\((\\rho)\\) means stronger unobserved confounding.")),
                ui.p(ui.HTML("Outcomes: \\((Y(0) = f(X) + \\gamma U + \\varepsilon,\\; Y(1)=Y(0)+\\tau)\\). The ATE is \\((\\tau)\\).")),
                ui.p(ui.HTML("\\(f(X)\\) summarizes observed confounding learned from the data; \\(U\\) violates exchangeability in a controlled way.")),
                ui.p(ui.HTML("This app uses a constant \\(\\tau\\); the Credence framework supports heterogeneity \\(\\tau(X)\\) and bias functions of \\(X\\)."))
            )
        ),

        ui.tags.details(
            ui.tags.summary("What f(X) is, and why it is now spelled out"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p(ui.HTML(
                    "\\(f(X)\\) is the <strong>prognostic index</strong>: how the observed "
                    "covariates move the untreated outcome \\(Y(0)\\). It is the entire "
                    "observed-confounding mechanism, so it should be a stated modelling "
                    "choice rather than an accident.")),
                ui.p(ui.HTML(
                    "It previously was an accident. \\(f(X)\\) was an unweighted mean of the "
                    "covariates <em>on their raw scale</em>, restricted to columns whose name "
                    "contained no underscore. So earnings in dollars drowned out age in years, "
                    "and any covariate you happened to name <code>prior_vaccine</code> was "
                    "silently dropped &mdash; exported to you as a confounder while influencing "
                    "nothing.")),
                ui.p(ui.HTML(
                    "Now every covariate is standardised first, then combined with "
                    "<strong>weights shown in the table below</strong>. Equal weights is the "
                    "default and means what it says. Turning the weights into a control, and "
                    "letting the effect \\(\\tau\\) vary with \\(X\\), is the next step for "
                    "this app."))
            )
        ),

        ui.row(
            ui.column(3, ui.input_numeric("tau", "True ATE (τ)", value=-5, min=-100, max=100, step=0.5)),
            ui.column(3, ui.input_numeric("rho", "Unmeasured Confounding (ρ)", value=0.0, min=0.0, max=1.0, step=0.1)),
            ui.column(3, ui.input_numeric("n_sim", "Sample Size", value=1500, min=100, max=10000, step=100)),
            ui.column(3, ui.input_numeric("gen_seed", "Generation Seed", value=7, min=0, step=1))
        ),
        ui.tags.p(
            {"class": "text-muted", "style": "font-size: 0.9em; margin-top: -10px;"},
            ui.HTML("ATE typical range: -10 to 10. ρ ranges: 0.0 (none) to 0.6+ (strong). <br><strong>Note:</strong> When ρ=0, the unobserved confounder U has zero influence and is excluded from the downloaded dataset.")
        ),
        ui.tags.p(
            {"class": "text-muted", "style": "font-size: 0.9em;"},
            ui.HTML("The <strong>generation seed</strong> is separate from the training seed on "
                    "purpose: it lets you redraw a cohort without retraining the model, or "
                    "retrain without moving the cohort, so you always know which change caused "
                    "what you are looking at. Same seed, same data &mdash; every time.")
        ),
        ui.input_action_button("generate_data", "Generate Synthetic Data", class_="btn-primary"),
        ui.output_text_verbatim("generation_output"),
        ui.output_ui("fx_weights")
    ),

    # Step 4: Analyze & Export
    ui.tags.div(
        {"class": "step-panel", "id": "step4"},
        ui.h3("Step 4: Analyze & Export"),
        ui.p("Inspect how naive comparisons depart from the truth, and reflect on what is needed to recover the causal contrast."),

        ui.tags.details(
            ui.tags.summary("Why naive is biased"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p(ui.HTML("Naive difference equals \\((\\tau + [\\mathbb{E}(Y(0) \\mid Z{=}1) - \\mathbb{E}(Y(0) \\mid Z{=}0)])\\). If either observed confounding (via \\((X)\\)) or unobserved \\((U)\\) shifts \\((Y(0))\\) across groups, bias appears."))
            )
        ),

        ui.output_ui("analysis_results"),
        ui.output_plot("outcome_plot"),

        ui.tags.hr(),
        ui.h4("Is the synthetic data actually realistic?"),
        ui.p(ui.HTML(
            "Everything above rests on a claim: that these synthetic covariates resemble "
            "your real ones. That claim used to be asserted and never checked. Here it is "
            "measured, against the <strong>holdout rows the model never saw</strong>.")),

        ui.tags.details(
            ui.tags.summary("How to read these three numbers"),
            ui.tags.div(
                {"class": "math-note"},
                ui.tags.ol(
                    ui.tags.li(ui.HTML(
                        "<strong>SMD</strong> &mdash; difference in means, in pooled standard "
                        "deviations. Below 0.1 is the usual threshold for negligible.")),
                    ui.tags.li(ui.HTML(
                        "<strong>KS distance</strong> &mdash; the largest gap between the two "
                        "cumulative distributions. Catches differences in spread and shape that "
                        "a difference in means cannot.")),
                    ui.tags.li(ui.HTML(
                        "<strong>Discriminator AUC</strong> &mdash; a random forest is asked to "
                        "tell real rows from synthetic ones, cross-validated so it cannot cheat "
                        "by memorising. <strong>0.5 is the target</strong>: it means the "
                        "generator has fooled it completely. This is the rare model you want to "
                        "perform badly."))
                ),
                ui.p(ui.HTML(
                    "Why all three: a generator can match every mean (SMD &asymp; 0) and still "
                    "get the shapes wrong (KS large). It can match every individual "
                    "distribution and still destroy the <em>correlations between</em> "
                    "covariates &mdash; and only the discriminator, which sees whole rows at "
                    "once, will catch that.")),
                ui.p(ui.HTML(
                    "<strong>Try this:</strong> go back to Step 2, set the batch size to "
                    "&ldquo;Full dataset&rdquo; with 5 epochs, retrain, and regenerate. Watch "
                    "the AUC climb toward 1.0. That is what the old version of this app was "
                    "shipping."))
            )
        ),
        ui.output_ui("realism_results"),

        ui.tags.hr(),
        ui.h4("Next Steps: Apply Methods from Module 4"),
        ui.p(
            "You now have a benchmark with known truth. Use the methods from Module 4 to analyze this data:"
        ),

        ui.tags.div(
            {"class": "module-ref"},
            ui.tags.ol(
                ui.tags.li(ui.HTML("<strong>G-computation:</strong> Fit <code>Q(A,W) = E[Y|A,W]</code> with Super Learner, then average over the population.")),
                ui.tags.li(ui.HTML("<strong>TMLE:</strong> Apply the targeting step to optimize for the ATE. Use the <code>tmle</code> R package as shown in Module 4.")),
                ui.tags.li(ui.HTML("<strong>DML/AIPW:</strong> Use cross-fitting with <code>DoubleML</code> for doubly-robust estimation.")),
            ),
            ui.p(ui.HTML("<strong>Try it:</strong> Download the data below and compare how each method performs against the known truth, especially as you vary ρ!"))
        ),

        ui.tags.details(
            ui.tags.summary("Full Causal Roadmap (from Module 2)"),
            ui.tags.ol(
                ui.tags.li(ui.HTML("<strong>Specify the question.</strong> Define target population, exposure(s), outcome(s), time, and contrast (ATE/ATT/ATC).")),
                ui.tags.li(ui.HTML("<strong>Specify a causal model.</strong> Draw a DAG; consider unmeasured and time-varying confounding; missingness/censoring.")),
                ui.tags.li(ui.HTML("<strong>Define the causal parameter.</strong> Use counterfactuals (Y(0), Y(1)) to formalize the estimand (e.g., τ = E[Y(1) − Y(0)]).")),
                ui.tags.li(ui.HTML("<strong>Describe the observed data/statistical model.</strong> What did we actually measure? Favor flexible functional forms.")),
                ui.tags.li(ui.HTML("<strong>Assess identifiability.</strong> State and evaluate assumptions (consistency, conditional exchangeability, positivity). Consider alternative designs (DiD, IV, front-door) when appropriate.")),
                ui.tags.li(ui.HTML("<strong>Define the statistical parameter.</strong> Map the causal estimand to a function of the observed data distribution.")),
                ui.tags.li(ui.HTML("<strong>Choose and implement an estimator.</strong> Prefer robust learners (e.g., Super Learner + TMLE/DML) with cross-fitting; check overlap; obtain valid CIs.")),
                ui.tags.li(ui.HTML("<strong>Conduct sensitivity analyses.</strong> Quantify residual confounding (Rosenbaum bounds, VanderWeele formulas, E-values), use negative controls, and report tipping points.")),
                ui.tags.li(ui.HTML("<strong>Interpret.</strong> State whether results are causal or associational; articulate assumptions and limitations; discuss transportability."))
            ),
            ui.p(ui.HTML("<em>See: Dang & Balzer (2023). Start with the Target Trial Protocol, Then Follow the Roadmap for Causal Inference. Epidemiology.</em>"))
        ),

        ui.tags.hr(),
        ui.h4("Citations & Acknowledgments"),
        ui.tags.p(
            ui.HTML("<strong>Credence Framework:</strong> Parikh, H., Vajao, C., Xu, L., Tchetgen Tchetgen, E. (2022). <em>Validating Causal Inference Methods</em>. Proceedings of the 39th International Conference on Machine Learning (PMLR 162). "),
            ui.tags.a("arXiv PDF", href="https://arxiv.org/pdf/2202.04208", target="_blank")
        ),
        ui.tags.p(
            ui.HTML("<strong>Workshop Materials:</strong> This app is part of the instats SSC 2026 Accreditation Workshop on Modern Analysis of Real-World Data Following the Causal Roadmap & Applications of Causal AI.")
        ),

        ui.download_button("download_data", "Download Synthetic Dataset (.csv)", class_="btn-success"),
    ),
)


# Server Logic
def server(input, output, session):
    # Reactive values
    data_loaded = reactive.Value(None)
    data_note = reactive.Value("")
    data_processed = reactive.Value(None)
    # Everything produced by Step 2 travels together in one dict: the model, the
    # prognostic index fitted on the same rows, and the holdout the model never saw.
    # Keeping them in a single value makes it impossible to pair a model with the
    # wrong holdout, which is the sort of mistake that silently flatters a metric.
    train_state = reactive.Value(None)
    sim_data = reactive.Value(None)      # the exported frame (download / plots read this)
    sim_extras = reactive.Value(None)    # fx, U, encoded covariates, realism report

    # Load example data. The datasets themselves live in examples.py so the test
    # suite exercises exactly what the workshop runs.
    @reactive.Effect
    @reactive.event(input.load_example)
    def _():
        name = input.example_dataset()
        if not name:
            return
        try:
            df, note = load_example(name)
            data_loaded.set(df)
            data_note.set(note)
        except Exception as e:
            data_note.set("Could not load example: {}".format(e))

    @output
    @render.ui
    def example_note():
        note = data_note.get()
        if not note:
            return None
        # A loud warning when the dataset is not what its label claims. The LaLonde
        # option falls back to a placeholder when app/lalonde_nsw.csv is absent, and
        # the old app said nothing at all -- so a participant could analyse pure
        # noise for an hour believing it was a famous evaluation dataset.
        is_warning = note.startswith("NOT ") or note.startswith("Could not")
        colour = "#fd7e14" if is_warning else "#1c8b4a"
        return ui.tags.div(
            {"class": "module-ref",
             "style": "border-left-color: {};".format(colour)},
            ui.HTML("<strong>{}</strong> {}".format(
                "Heads up:" if is_warning else "Loaded:", note))
        )

    # Load uploaded CSV
    @reactive.Effect
    @reactive.event(input.csv_file)
    def _():
        file_info = input.csv_file()
        if file_info is not None:
            df = pd.read_csv(file_info[0]["datapath"])
            data_loaded.set(df)

    # Show column selectors
    @output
    @render.ui
    def csv_controls():
        df = data_loaded.get()
        if df is None:
            return None

        # Smart defaults
        outcome_default = "outcome" if "outcome" in df.columns else df.columns[0]
        treatment_default = "treatment" if "treatment" in df.columns else df.columns[1]

        return ui.TagList(
            ui.input_select("outcome_col", "Outcome (Y) Column",
                          choices=list(df.columns), selected=outcome_default),
            ui.input_select("treatment_col", "Treatment (Z) Column",
                          choices=list(df.columns), selected=treatment_default)
        )

    # Process data
    @reactive.Effect
    @reactive.event(input.process_data)
    def _():
        df = data_loaded.get()
        if df is None:
            return

        # Check treatment is binary
        if df[input.treatment_col()].nunique() != 2:
            return

        # Create processed dataset
        processed = df.copy()
        processed = processed.dropna()

        # Ensure treatment is 0/1
        z_vals = processed[input.treatment_col()].unique()
        z_map = {z_vals[0]: 0, z_vals[1]: 1}
        processed[input.treatment_col()] = processed[input.treatment_col()].map(z_map)

        data_processed.set(processed)

    # Show data summary
    @output
    @render.text
    def data_summary():
        df = data_processed.get()
        if df is None:
            return ""

        return f"Data confirmed: {len(df)} complete observations.\nTreatment: {input.treatment_col()}\nOutcome: {input.outcome_col()}"

    # Train CVAE
    @reactive.Effect
    @reactive.event(input.learn_model)
    def _():
        df = data_processed.get()
        if df is None:
            return

        outcome_col = input.outcome_col()
        treatment_col = input.treatment_col()
        seed = int(input.train_seed())

        # Split BEFORE training. The holdout is the only honest yardstick for the
        # realism check in Step 4: measured against its own training rows, a
        # flexible generator can score well by memorising them.
        split_rng = np.random.default_rng(seed)
        order = split_rng.permutation(len(df))
        n_hold = max(20, int(round(float(input.holdout_frac()) * len(df))))
        n_hold = min(n_hold, len(df) - 20) if len(df) > 40 else max(1, len(df) // 4)

        df_hold = df.iloc[order[:n_hold]].reset_index(drop=True)
        df_train = df.iloc[order[n_hold:]].reset_index(drop=True)

        X_train, Z_train, feature_names = encode_data(
            df_train, outcome_col=outcome_col, treatment_col=treatment_col)
        X_hold, _, _ = encode_data(
            df_hold, outcome_col=outcome_col, treatment_col=treatment_col)

        batch_size = int(input.batch_size())
        batch_size = None if batch_size <= 0 else batch_size

        # Seed before construction: torch draws the initial weights from its global
        # generator at __init__ time, so seeding inside fit() would already be late.
        set_seed(seed)
        model = ConditionalVAE(input_dim=X_train.shape[1], latent_dim=16)
        model.fit(X_train, Z_train, epochs=int(input.epochs()), lr=1e-3,
                  batch_size=batch_size, seed=seed, verbose=True)

        # f(X) is fitted on the same training rows, so the standardisation constants
        # it uses come from data the model was allowed to see.
        index = PrognosticIndex().fit(X_train, feature_names)

        train_state.set({
            'model': model, 'index': index, 'feature_names': feature_names,
            'X_train': X_train, 'X_hold': X_hold,
            'df_train': df_train, 'df_hold': df_hold,
            'n_train': len(df_train), 'n_hold': len(df_hold),
            'batch_size': batch_size, 'seed': seed,
        })

    # Training output
    @output
    @render.text
    def training_output():
        state = train_state.get()
        if state is None:
            return ""

        history = state['model'].history_
        first, last = history[0], history[-1]
        steps_per_epoch = last['steps']
        return (
            "CVAE training complete.\n"
            "  trained on {} rows, {} held out for the realism check\n"
            "  {} epochs x {} gradient step(s) per epoch = {} steps total\n"
            "  negative ELBO per observation: {:.4f} -> {:.4f}\n"
            "  final split: reconstruction {:.4f} + KL {:.4f}"
        ).format(
            state['n_train'], state['n_hold'],
            len(history), steps_per_epoch, len(history) * steps_per_epoch,
            first['neg_elbo'], last['neg_elbo'], last['recon'], last['kl'])

    # The training curve. Reconstruction and KL are drawn separately because the
    # interesting behaviour is the trade-off between them: reconstruction drops
    # quickly, then the KL term starts to resist and progress slows.
    @output
    @render.plot
    def elbo_plot():
        state = train_state.get()
        if state is None:
            return None

        history = state['model'].history_
        epochs = [h['epoch'] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

        axes[0].plot(epochs, [h['neg_elbo'] for h in history],
                     color='#4A9EFF', linewidth=2)
        axes[0].set_title('Negative ELBO per observation (lower is better)')
        axes[0].set_xlabel('Epoch')

        axes[1].plot(epochs, [h['recon'] for h in history],
                     color='#fd7e14', linewidth=2, label='Reconstruction')
        axes[1].plot(epochs, [h['kl'] for h in history],
                     color='#1c8b4a', linewidth=2, label='KL divergence')
        axes[1].set_title('The two competing pressures')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # Generate synthetic data. The whole data-generating process lives in dgp.py --
    # see generate_cohort() there for the step-by-step, which is worth reading
    # aloud in the workshop because it is the part that defines "truth".
    @reactive.Effect
    @reactive.event(input.generate_data)
    def _():
        state = train_state.get()
        df = data_processed.get()
        if state is None or df is None:
            return

        outcome_col = input.outcome_col()
        treatment_col = input.treatment_col()
        n = int(input.n_sim())
        seed = int(input.gen_seed())

        result = generate_cohort(
            model=state['model'], index=state['index'], original_df=df,
            feature_names=state['feature_names'], n=n,
            tau=float(input.tau()), rho=float(input.rho()),
            sd_y=float(df[outcome_col].std()),
            p_treat=float(df[treatment_col].mean()),
            outcome_col=outcome_col, treatment_col=treatment_col, seed=seed)

        # Realism is measured on a synthetic batch the same size as the holdout, and
        # drawn at the holdout's own treatment assignments, so the comparison is not
        # confounded by a different treated/control mix.
        holdout_check = generate_cohort(
            model=state['model'], index=state['index'], original_df=df,
            feature_names=state['feature_names'], n=len(state['X_hold']),
            tau=float(input.tau()), rho=float(input.rho()),
            sd_y=float(df[outcome_col].std()),
            p_treat=float(df[treatment_col].mean()),
            outcome_col=outcome_col, treatment_col=treatment_col, seed=seed + 1)

        report = realism_report(state['X_hold'], holdout_check['X_encoded'],
                                state['feature_names'], seed=seed)

        sim_data.set(result['data'])
        sim_extras.set({'result': result, 'realism': report})

    # Generation output
    @output
    @render.text
    def generation_output():
        df = sim_data.get()
        extras = sim_extras.get()
        if df is None or extras is None:
            return ""
        result = extras['result']
        return (
            "Synthetic data generated: {} observations, {} columns.\n"
            "  true ATE: {:.3f}   |   treated fraction: {:.3f}\n"
            "  all {} covariates exported in their original schema"
        ).format(len(df), df.shape[1], result['true_ate'],
                 float(np.mean(result['Z'])), result['decoded'].shape[1])

    # The f(X) weights, shown rather than hidden. This is the observed-confounding
    # mechanism in full: if a covariate has weight 0 it cannot confound anything.
    @output
    @render.ui
    def fx_weights():
        state = train_state.get()
        if state is None or sim_data.get() is None:
            return None

        table = state['index'].weights_table()
        display = table[['covariate', 'weight']].copy()
        display['weight'] = display['weight'].map(lambda v: "{:.4f}".format(v))

        return ui.TagList(
            ui.tags.h5("f(X): how each covariate moves the untreated outcome"),
            ui.HTML(display.to_html(index=False, border=0,
                                    classes="table table-sm", justify="left")),
            ui.tags.p(
                {"class": "text-muted", "style": "font-size: 0.9em;"},
                ui.HTML("Covariates are standardised before weighting, so these are "
                        "comparable across variables measured in different units. "
                        "Equal weights is the default; the old code produced weights "
                        "that were an accident of measurement scale.")
            )
        )

    # The realism verdict.
    @output
    @render.ui
    def realism_results():
        extras = sim_extras.get()
        state = train_state.get()
        if extras is None or state is None:
            return None

        report = extras['realism']
        disc = report['discriminator']
        covariates = report['covariates'].copy()

        auc = disc['auc']
        colour = "#1c8b4a" if auc < 0.60 else ("#fd7e14" if auc < 0.80 else "#dc3545")
        worst = covariates.iloc[0]

        display = covariates[['covariate', 'real_mean', 'synthetic_mean', 'smd', 'ks']].copy()
        for col in ('real_mean', 'synthetic_mean', 'smd', 'ks'):
            display[col] = display[col].map(lambda v: "{:.3f}".format(v))
        display.columns = ['Covariate', 'Real (holdout)', 'Synthetic', 'SMD', 'KS']

        return ui.TagList(
            ui.row(
                ui.column(4, ui.tags.div(
                    {"class": "metric-card"},
                    ui.tags.div({"class": "metric-label"}, "Discriminator AUC"),
                    ui.tags.div({"class": "metric-value", "style": "color: {};".format(colour)},
                                "{:.3f}".format(auc)),
                    ui.tags.div({"class": "metric-label"}, "0.5 = indistinguishable")
                )),
                ui.column(4, ui.tags.div(
                    {"class": "metric-card"},
                    ui.tags.div({"class": "metric-label"}, "Largest |SMD|"),
                    ui.tags.div({"class": "metric-value"}, "{:.3f}".format(abs(worst['smd']))),
                    ui.tags.div({"class": "metric-label"}, str(worst['covariate']))
                )),
                ui.column(4, ui.tags.div(
                    {"class": "metric-card"},
                    ui.tags.div({"class": "metric-label"}, "Compared against"),
                    ui.tags.div({"class": "metric-value"}, str(disc['n_real'])),
                    ui.tags.div({"class": "metric-label"}, "held-out rows, unseen in training")
                ))
            ),
            ui.tags.div(
                {"class": "module-ref", "style": "border-left-color: {};".format(colour)},
                ui.HTML("<strong>Verdict:</strong> {}".format(disc['interpretation']))
            ),
            ui.HTML(display.to_html(index=False, border=0,
                                    classes="table table-sm", justify="left")),
            ui.tags.p(
                {"class": "text-muted", "style": "font-size: 0.9em;"},
                ui.HTML("Sorted worst-first by |SMD|. Rows near the top are where the "
                        "generator is least faithful &mdash; and therefore where a "
                        "conclusion drawn here is least likely to transfer to your real data.")
            )
        )

    # Analysis results
    @output
    @render.ui
    def analysis_results():
        df = sim_data.get()
        if df is None:
            return None

        true_ate = df['tau_true'].iloc[0]
        y_col = input.outcome_col()
        z_col = input.treatment_col()

        naive_ate = df[df[z_col] == 1][y_col].mean() - df[df[z_col] == 0][y_col].mean()
        bias = naive_ate - true_ate

        return ui.TagList(
            ui.h4("Analysis Results"),
            ui.row(
                ui.column(4, ui.tags.div(
                    {"class": "metric-card"},
                    ui.tags.div({"class": "metric-label"}, "True ATE"),
                    ui.tags.div({"class": "metric-value", "style": "color: #4A9EFF;"}, f"{true_ate:.2f}")
                )),
                ui.column(4, ui.tags.div(
                    {"class": "metric-card"},
                    ui.tags.div({"class": "metric-label"}, "Naive Estimate"),
                    ui.tags.div({"class": "metric-value"}, f"{naive_ate:.2f}")
                )),
                ui.column(4, ui.tags.div(
                    {"class": "metric-card"},
                    ui.tags.div({"class": "metric-label"}, "Bias"),
                    ui.tags.div({"class": "metric-value", "style": "color: #dc3545;"}, f"{bias:.2f}")
                ))
            )
        )

    # Outcome plot
    @output
    @render.plot
    def outcome_plot():
        df = sim_data.get()
        if df is None:
            return None

        y_col = input.outcome_col()
        z_col = input.treatment_col()

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot density for each treatment group with instats colors
        for z_val, color, label in [(0, '#6c757d', 'Control'), (1, '#4A9EFF', 'Treated')]:
            df_sub = df[df[z_col] == z_val][y_col]
            df_sub.plot.density(ax=ax, color=color, alpha=0.7, label=label, linewidth=2)

        ax.set_xlabel('Outcome (Y)')
        ax.set_ylabel('Density')
        ax.set_title('Outcome Distributions by Treatment Group')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    # Download handler
    @session.download(
        filename=lambda: f"credence_instats_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def download_data():
        df = sim_data.get()
        if df is None:
            yield ""
            return

        # Convert to CSV and yield as string
        csv_string = df.to_csv(index=False)
        yield csv_string


# Create app
app = App(app_ui, server)
