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

from cvae_model import ConditionalVAE, encode_data, decode_samples

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
                ", so generated data are realistic and near-indistinguishable from the observed sample."
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

        ui.input_slider("epochs", "Training Epochs", min=10, max=200, value=50, step=10),
        ui.input_action_button("learn_model", "Train CVAE Model", class_="btn-primary"),
        ui.output_text_verbatim("training_output")
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

        ui.row(
            ui.column(4, ui.input_numeric("tau", "True ATE (τ)", value=-5, min=-100, max=100, step=0.5)),
            ui.column(4, ui.input_numeric("rho", "Unmeasured Confounding (ρ)", value=0.0, min=0.0, max=1.0, step=0.1)),
            ui.column(4, ui.input_numeric("n_sim", "Sample Size", value=1500, min=100, max=10000, step=100))
        ),
        ui.tags.p(
            {"class": "text-muted", "style": "font-size: 0.9em; margin-top: -10px;"},
            ui.HTML("ATE typical range: -10 to 10. ρ ranges: 0.0 (none) to 0.6+ (strong). <br><strong>Note:</strong> When ρ=0, the unobserved confounder U has zero influence and is excluded from the downloaded dataset.")
        ),
        ui.input_action_button("generate_data", "Generate Synthetic Data", class_="btn-primary"),
        ui.output_text_verbatim("generation_output")
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
    data_processed = reactive.Value(None)
    cvae_model = reactive.Value(None)
    encoded_features = reactive.Value(None)
    sim_data = reactive.Value(None)

    # Load example data
    @reactive.Effect
    @reactive.event(input.load_example)
    def _():
        if input.example_dataset() == "lalonde":
            try:
                # Try to load from file if available
                lalonde_path = Path(__file__).parent / "lalonde_nsw.csv"
                if lalonde_path.exists():
                    df = pd.read_csv(lalonde_path)
                    df = df.rename(columns={"treat": "treatment", "re78": "outcome"})
                else:
                    # Create a minimal example if file doesn't exist
                    df = pd.DataFrame({
                        'treatment': np.random.binomial(1, 0.5, 100),
                        'outcome': np.random.randn(100) * 1000 + 5000,
                        'age': np.random.randint(18, 65, 100),
                        'education': np.random.randint(8, 18, 100),
                    })
                data_loaded.set(df)
            except Exception as e:
                print(f"Error loading example: {e}")
        elif input.example_dataset() == "pneumonia":
            # Create pneumonia vaccine example similar to Module 4
            np.random.seed(123)
            n = 1000
            age = np.random.normal(65, 15, n)
            prior_pneumonia = np.random.binomial(1, 1/(1+np.exp(-(-2 + 0.03*age))), n)
            prior_vaccine = np.random.binomial(1, 1/(1+np.exp(-(-1 + 0.02*age + 0.5*prior_pneumonia))), n)
            # Treatment with confounding
            treatment = np.random.binomial(1, 1/(1+np.exp(-(-2.5 + 0.025*age + 0.8*prior_pneumonia + 1.2*prior_vaccine))), n)
            # Outcome (will be replaced by our simulation)
            outcome = np.random.normal(0, 1, n)

            df = pd.DataFrame({
                'treatment': treatment,
                'outcome': outcome,
                'age': age,
                'priorPneumonia': prior_pneumonia,
                'priorVaccine': prior_vaccine
            })
            data_loaded.set(df)

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

        # Encode data
        X, Z, feature_names = encode_data(
            df,
            outcome_col=input.outcome_col(),
            treatment_col=input.treatment_col()
        )

        # Train model
        model = ConditionalVAE(input_dim=X.shape[1], latent_dim=16)
        model.fit(X, Z, epochs=input.epochs(), lr=1e-3, verbose=True)

        cvae_model.set(model)
        encoded_features.set((X, Z, feature_names))

    # Training output
    @output
    @render.text
    def training_output():
        model = cvae_model.get()
        if model is None:
            return ""
        return "CVAE training complete!"

    # Generate synthetic data
    @reactive.Effect
    @reactive.event(input.generate_data)
    def _():
        model = cvae_model.get()
        df = data_processed.get()
        if model is None or df is None:
            return

        n = input.n_sim()
        tau = input.tau()
        rho = input.rho()

        # Generate confounded treatment assignment
        U = np.random.randn(n)
        pz = df[input.treatment_col()].mean()

        # Logistic function for treatment probability
        logit_pz = np.log(pz / (1 - pz)) if pz > 0 and pz < 1 else 0
        ps_confounded = 1 / (1 + np.exp(-(logit_pz + rho * U)))
        z_sim = np.random.binomial(1, ps_confounded)

        # Sample covariates conditional on treatment
        X_sim = model.sample(z_sim, n_samples=n)

        # Generate outcomes
        X_encoded, _, feature_names = encoded_features.get()

        # Get numeric features
        numeric_features = [i for i, name in enumerate(feature_names) if not any(cat in name for cat in ['_'])]
        if len(numeric_features) > 0:
            hX = X_sim[:, numeric_features].mean(axis=1)
        else:
            hX = np.zeros(n)

        sdy = df[input.outcome_col()].std()

        # Potential outcomes
        y0 = hX + (0.5 * rho * U * sdy) + np.random.randn(n) * sdy
        y1 = y0 + tau
        y_obs = np.where(z_sim == 1, y1, y0)

        # Create synthetic dataset
        sim_df = pd.DataFrame({
            input.outcome_col(): y_obs,
            input.treatment_col(): z_sim,
            'y0': y0,
            'y1': y1,
            'tau_true': tau
        })

        # Only include U column if rho > 0 (i.e., if U actually affects anything)
        # When rho=0, U is generated but has zero influence, so we exclude it for clarity
        if rho > 0:
            sim_df['U'] = U

        # Add covariates (simplified - using encoded values)
        for i, name in enumerate(feature_names[:min(10, len(feature_names))]):
            sim_df[f'X_{name}'] = X_sim[:, i]

        sim_data.set(sim_df)

    # Generation output
    @output
    @render.text
    def generation_output():
        df = sim_data.get()
        if df is None:
            return ""
        return f"Synthetic data generated: {len(df)} observations."

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
