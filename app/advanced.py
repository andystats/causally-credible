"""Tao of RWD: custom experiments based on the Credence framework."""

from shiny import ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from examples import load_example
from scenarios import summarize_grid
from workflow import ExperimentSession, treatment_levels, answer_bundle
from comparison import compare_estimates, comparison_plot

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Shared styles and MathJax
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



<style>
    :root {
        --lab-blue: #2855a6;
        --lab-orange: #b85c26;
        --lab-green: #087f79;
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
        border-left-color: var(--lab-orange);
    }
    .step-complete {
        border-left-color: var(--lab-green);
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
        background-color: var(--lab-blue);
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .btn-primary:hover {
        background-color: #3a7fcf;
    }
    .btn-success {
        background-color: var(--lab-green);
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .teaching-note {
        background: #e8f4ff;
        border-left: 3px solid var(--lab-blue);
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95em;
    }
</style>
"""

advanced_ui = ui.navset_tab(
    ui.nav_panel('1 · Data', ui.tags.div(
        {"class": "step-panel step-active", "id": "step1"},
        ui.h3("Step 1: Load Observational Data"),
        ui.p(ui.HTML("Upload a CSV to anchor the covariate structure \\((X)\\) and treatment \\((Z)\\). This provides realistic dependence patterns for the generator.")),

        ui.tags.details(
            ui.tags.summary("Data requirements"),
            ui.p("Choose a numeric outcome and a binary treatment. Remaining columns are covariates; "
                 "numeric and categorical values are supported. Rows with missing values are dropped."),
            ui.p("The generator learns covariates conditional on treatment, p(X|Z). "
                 "The outcome mechanism is chosen in step 3, not fitted to the source outcomes."),
        ),

        ui.row(
            ui.column(
                8,
                ui.input_file("csv_file", "Upload Your CSV", accept=[".csv"], multiple=False)
            ),
            ui.column(
                4,
                ui.input_select("example_dataset", "Or load example:",
                              choices={"": "None", "lalonde": "LaLonde NSW", "pneumonia": "Pneumonia vaccine (simulated)"}),
                ui.input_action_button("load_example", "Load Example", class_="btn-secondary")
            )
        ),
        ui.output_ui("example_note"),
        ui.output_ui("csv_controls"),
        ui.output_ui("treatment_controls"),
        ui.input_action_button("process_data", "Confirm Data", class_="btn-primary"),
        ui.output_text_verbatim("data_summary")
    )),
    ui.nav_panel('2 · Train', ui.tags.div(
        {"class": "step-panel", "id": "step2"},
        ui.h3("Step 2: Learn Data Structure"),
        ui.p(ui.HTML("Train a Conditional VAE to learn \\((p(X\\mid Z))\\), capturing how covariates shift across treatment groups (observed confounding).")),

        ui.tags.details(
            ui.tags.summary("Training and holdout"),
            ui.p("Each batch produces one gradient update. Smaller batches give more updates per epoch; "
                 "a full-data batch gives one. Check the loss curve and held-out diagnostics before relying on the generator."),
            ui.p("Holdout rows are excluded from training and used to check covariate fidelity in step 4."),
        ),

        ui.row(
            ui.column(3, ui.input_slider("epochs", "Training Epochs",
                                         min=1, max=400, value=150, step=1)),
            ui.column(3, ui.input_select("batch_size", "Batch Size",
                                         choices={"32": "32", "64": "64", "128": "128",
                                                  "256": "256", "0": "Full dataset"},
                                         selected="128")),
            ui.column(3, ui.input_slider("holdout_frac", "Holdout Fraction",
                                         min=0.1, max=0.5, value=0.25, step=0.05)),
            ui.column(3, ui.input_numeric("train_seed", "Training Seed", value=1, min=0, step=1))
        ),
        ui.input_action_button("learn_model", "Train CVAE Model", class_="btn-primary"),
        ui.output_text_verbatim("training_output"),
        ui.output_plot("elbo_plot")
    )),
    ui.nav_panel('3 · Generate', ui.tags.div(
        {"class": "step-panel", "id": "step3"},
        ui.h3("Step 3: Generate a Study"),
        ui.p("Create new data using the learned structure. You set the baseline effect and inject unmeasured confounding via a latent ", ui.tags.code("U"), "."),

        ui.tags.details(
            ui.tags.summary("Data-generating process"),
            ui.p("The learned mechanism draws treatment first, then covariates conditional on treatment. "
                 "The designed mechanism draws covariates first, then assigns treatment using the overlap and hidden-confounding controls."),
            ui.p("Untreated outcomes combine the standardized prognostic index, gamma × rho × U × training-outcome SD, "
                 "and noise. Treated outcomes add the per-person effect. The reported true ATE is the mean of those effects."),
        ),

        ui.tags.details(
            ui.tags.summary("Baseline outcome: f(X)"),
            ui.p("The prognostic index combines standardized covariates with equal weights to set "
                 "the untreated outcome. The weights are shown below after generation. "
                 "Effect heterogeneity is controlled separately."),
        ),

        ui.row(
            ui.column(3, ui.input_numeric("tau", "Baseline effect (τ₀)", value=-5, min=-100, max=100, step=0.5)),
            ui.column(3, ui.input_numeric("rho", "Unmeasured Confounding (ρ)", value=0.0, min=0.0, max=1.0, step=0.1)),
            ui.column(3, ui.input_numeric("n_sim", "Sample Size", value=1500, min=100, max=10000, step=100)),
            ui.column(3, ui.input_numeric("gen_seed", "Generation Seed", value=7, min=0, step=1))
        ),

        ui.tags.details(
            ui.tags.summary("Effect heterogeneity and the actual average effect"),
            ui.p("With zero heterogeneity, everyone has the same effect. Estimators can still differ in finite-sample "
                 "error or robustness; heterogeneous effects additionally let us study patient-level benefit."),
            ui.p("The effect is τ(X) = τ₀ + κ × training-outcome SD × m(X). "
                 "The modifier m(X) is centered on the training data, not recentered on each new cohort."),
            ui.p("τ₀ is the baseline effect at m(X)=0. The generated population can have a different modifier mean, "
                 "so its actual average effect may differ from τ₀. Both are reported explicitly."),
        ),

        ui.tags.details(
            ui.tags.summary("Treatment assignment and overlap"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p(ui.HTML(
                    "<strong>Learned</strong> mechanism: draw \\(Z\\), then \\(X \\mid Z\\) from "
                    "the CVAE. Confounding is inherited from your real data &mdash; realistic, "
                    "but not adjustable.")),
                ui.p(ui.HTML(
                    "<strong>Designed</strong> mechanism: draw \\(X\\) first, then "
                    "$$\\Pr(Z=1 \\mid X, U) = \\text{logistic}(\\alpha + \\lambda\\, m(X) + \\rho U).$$ "
                    "With no hidden confounding, \\(\\lambda\\)=0 is a randomised trial on realistic "
                    "covariates, 1 is real confounding with comfortable overlap, 3 piles "
                    "propensities up at 0 and 1.")),
                ui.p(ui.HTML(
                    "The designed rule changes arm-specific covariate distributions. Its "
                    "assignment probabilities are choices of this simulation, not estimates "
                    "of the original study's treatment mechanism."))
            )
        ),

        ui.row(
            ui.column(4, ui.input_select(
                "mechanism", "Treatment mechanism",
                choices={"learned": "Learned from data (not adjustable)",
                         "designed": "Designed (overlap is a dial)"},
                selected="designed")),
            ui.column(4, ui.input_slider("overlap_strength", "Overlap / confounding (λ)",
                                         min=0.0, max=4.0, value=1.0, step=0.25)),
            ui.column(4, ui.input_slider("heterogeneity", "Effect heterogeneity (κ)",
                                         min=0.0, max=3.0, value=0.0, step=0.25))
        ),
        ui.tags.p(
            {"class": "text-muted", "style": "font-size: 0.9em; margin-top: -10px;"},
            ui.HTML("The hidden confounder and potential outcomes are available only in the separate answer key.")
        ),
        ui.tags.p(
            {"class": "text-muted", "style": "font-size: 0.9em;"},
            "Change the generation seed to redraw a cohort without retraining. "
            "Reproducibility requires the same model, settings, code, and runtime."
        ),
        ui.input_action_button("generate_data", "Generate Synthetic Data", class_="btn-primary"),
        ui.output_text_verbatim("generation_output"),
        ui.output_ui("fx_weights")
    )),
    ui.nav_panel('4 · Compare & export', ui.tags.div(
        {"class": "step-panel", "id": "step4"},
        ui.h3("Step 4: Compare & Export"),
        ui.p("Compare unadjusted and adjusted estimates with the true cohort effect."),

        ui.tags.details(
            ui.tags.summary("Why estimates differ from truth"),
            ui.tags.div(
                {"class": "math-note"},
                ui.p("The observed treated-minus-control contrast can differ from the cohort-average effect because the treatment groups have different baseline outcomes and, with heterogeneity, different effects. One run shows estimation error; repeated runs are needed to measure bias."),
            )
        ),

        ui.output_ui("analysis_results"),
        ui.output_plot("comparison_plot_output", height="280px"),
        ui.output_plot("outcome_plot"),

        ui.p("The CSV contains observed data only. The answer bundle adds ground truth, aligned by row order. "
             "Keep answer-key columns out of estimator inputs."),
        ui.output_ui("download_controls"),
        ui.tags.hr(),
        ui.h4("Check covariate fidelity"),
        ui.p("Compare generated covariates with held-out source rows."),
        ui.tags.details(
            ui.tags.summary("Read the diagnostics"),
            ui.tags.ul(
                ui.tags.li(ui.strong("SMD: "), "Difference in means, in pooled standard deviations."),
                ui.tags.li(ui.strong("KS distance: "), "Largest gap between cumulative distributions; detects differences in spread and shape."),
                ui.tags.li(ui.strong("Discriminator AUC: "), "A cross-validated random forest distinguishes whole source and synthetic rows. "
                           "AUC near 0.5 does not prove equivalent distributions or a valid causal mechanism."),
            ),
            ui.p("Check all three: similar means can hide different shapes and dependencies."),
        ),
        ui.output_ui("realism_results"),
    )),
    ui.nav_panel('5 · Scenario grid', ui.tags.div(
        {"class": "step-panel", "id": "step5"},
        ui.h3("Step 5: Factorial Scenario Grid"),
        ui.p("Compare 12 combinations of overlap, effect heterogeneity, and hidden confounding, "
             "using common random draws."),
        ui.tags.details(
            ui.tags.summary("Read the grid"),
            ui.tags.ul(
                ui.tags.li("Overlap (λ): how strongly measured covariates affect assignment."),
                ui.tags.li("Heterogeneity (κ): how much the effect varies across people."),
                ui.tags.li("Hidden confounding (ρ): how strongly an unmeasured cause affects assignment and outcomes."),
            ),
            ui.p("Compare cells that differ in one setting. The % extreme column flags assignment "
                 "probabilities near 0 or 1. Each cell uses one cohort; repeated runs are needed to estimate bias."),
        ),

        ui.row(
            ui.column(4, ui.input_numeric("grid_n", "Rows per cell", value=1500,
                                          min=200, max=8000, step=100)),
            ui.column(4, ui.input_numeric("grid_seed", "Grid seed", value=101, min=0, step=1)),
            ui.column(4, ui.input_action_button("run_grid", "Run 12-cell grid",
                                                class_="btn-primary"))
        ),
        ui.output_text_verbatim("grid_status"),
        ui.output_ui("grid_results"),
        ui.output_ui("grid_download_controls"),
    )),
    ui.nav_panel('How it works', ui.tags.div(
        {"class": "cc-card"},
        ui.h3("Learn structure, choose mechanisms, check estimates"),
        ui.tags.ul(
            ui.tags.li(ui.strong("Learn: "), "A conditional VAE approximates p(X|Z) from source data."),
            ui.tags.li(ui.strong("Choose: "), "You set assignment, the outcome equation, and treatment effects."),
            ui.tags.li(ui.strong("Check: "), "Compare estimates with known truth and covariates with held-out source rows. "
                       "Covariate fidelity does not validate the chosen causal mechanism."),
        ),
        ui.tags.details(
            ui.tags.summary("Conditional VAE"),
            ui.p(ui.HTML("The encoder \\(q_\\phi(u \\mid x,z)\\) compresses covariates within treatment groups. "
                         "The decoder \\(p_\\theta(x \\mid u,z)\\) reconstructs them and generates new covariates.")),
            ui.p("Training balances reconstruction error with a penalty that keeps the latent distribution near a standard normal prior. "
                 "This latent representation is separate from the hidden confounder U added during simulation."),
            ui.p(ui.HTML("$$\\mathcal{L}(\\theta,\\phi) = \\mathbb{E}_{q_\\phi(u \\mid x,z)}[\\log p_\\theta(x \\mid u,z)] "
                         "- \\mathrm{KL}(q_\\phi(u \\mid x,z) \\Vert p(u))$$")),
        ),
        ui.tags.details(
            ui.tags.summary("Simulation equations"),
            ui.p(ui.HTML("<strong>Designed assignment:</strong> \\(\\Pr(Z=1 \\mid X,U) = "
                         "\\operatorname{logistic}(\\alpha + \\lambda m(X) + \\rho U)\\). "
                         "The learned mechanism instead draws treatment before covariates.")),
            ui.p(ui.HTML("<strong>Potential outcomes:</strong> \\(Y(0)=f(X)+\\gamma\\rho\\sigma_Y U+\\varepsilon\\), "
                         "\\(Y(1)=Y(0)+\\tau(X)\\), with \\(\\tau(X)=\\tau_0+\\kappa\\sigma_Y m(X)\\).")),
            ui.p("The cohort ATE averages individual effects. It equals τ₀ when heterogeneity is zero."),
            ui.tags.ul(
                ui.tags.li("X: measured covariates; Z: treatment; U: hidden confounder."),
                ui.tags.li("f(X): baseline outcome index; m(X): training-centered modifier; σY: training-outcome SD."),
                ui.tags.li("λ: overlap; ρ: hidden confounding; κ: heterogeneity; α: assignment intercept; γ = 0.5: hidden outcome coefficient."),
            ),
        ),
        ui.tags.details(
            ui.tags.summary("Extend the comparison"),
            ui.p("Use the observed-data export to compare other estimators, such as IPW, AIPW, or TMLE. "
                 "Use the answer key only to score them. Repeat simulations to study bias, precision, and interval coverage."),
            ui.p("Start with the causal question and identification assumptions, then choose an estimator. "
                 "No estimator can recover information that the study design does not identify."),
        ),
        ui.h4("Framework reference"),
        ui.p("Parikh et al. (2022). ",
             ui.tags.a("Validating Causal Inference Methods", href="https://arxiv.org/abs/2202.04208", target="_blank"),
             ". ICML. This app adapts the Credence framework for teaching."),
    )),
    id="advanced_step",
)

# Server Logic
def advanced_server(input, output, session):
    # Reactive values
    data_loaded = reactive.Value(None)
    data_note = reactive.Value("")
    experiment = reactive.Value(ExperimentSession())

    def report_error(exc):
        ui.notification_show(str(exc), type="error", duration=12)

    @reactive.Effect
    @reactive.event(input.outcome_col, input.treatment_col, input.treated_level)
    def invalidate_roles():
        experiment.set(ExperimentSession())

    @reactive.Effect
    @reactive.event(input.train_seed, input.holdout_frac, input.epochs, input.batch_size)
    def invalidate_training():
        old = experiment.get()
        experiment.set(ExperimentSession(confirmed=old.confirmed))

    @reactive.Effect
    @reactive.event(input.tau, input.rho, input.n_sim, input.gen_seed,
                    input.heterogeneity, input.mechanism, input.overlap_strength)
    def invalidate_generation():
        old = experiment.get()
        experiment.set(ExperimentSession(confirmed=old.confirmed, trained=old.trained))

    @reactive.Effect
    @reactive.event(input.grid_n, input.grid_seed)
    def invalidate_grid():
        old = experiment.get()
        experiment.set(ExperimentSession(confirmed=old.confirmed, trained=old.trained, run=old.run))

    # UI and headless tests share the same example loaders.
    @reactive.Effect
    @reactive.event(input.load_example)
    def _():
        name = input.example_dataset()
        if not name:
            return
        experiment.set(ExperimentSession())
        try:
            df, note = load_example(name)
            data_loaded.set(df)
            data_note.set(note)
        except Exception as e:
            data_loaded.set(None)
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
            {"class": "teaching-note",
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
            experiment.set(ExperimentSession())
            data_note.set("")
            try:
                df = pd.read_csv(file_info[0]["datapath"])
                if len(df.columns) < 3:
                    raise ValueError("Include outcome, treatment and at least one covariate.")
                data_loaded.set(df)
            except Exception as exc:
                data_loaded.set(None)
                report_error(exc)

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

    @output
    @render.ui
    def treatment_controls():
        df = data_loaded.get()
        if df is None or input.treatment_col() not in df:
            return None
        levels = treatment_levels(df[input.treatment_col()])
        if set(levels) == {0, 1}:
            choices, selected = {"1": "1 (treated); 0 (control)"}, "1"
        else:
            choices = {"": "Choose the treated level", **{str(i): str(v) for i, v in enumerate(levels)}}
            selected = ""
        return ui.input_select("treated_level", "Treatment meaning", choices=choices, selected=selected)

    @reactive.Effect
    @reactive.event(input.process_data)
    def confirm_data():
        experiment.set(ExperimentSession())
        df = data_loaded.get()
        if df is None:
            report_error(ValueError("Load a dataset first."))
            return
        try:
            levels = treatment_levels(df[input.treatment_col()])
            selection = input.treated_level()
            treated = 1 if set(levels) == {0, 1} else (levels[int(selection)] if selection else None)
            experiment.set(ExperimentSession().confirm(df, input.outcome_col(), input.treatment_col(), treated))
        except Exception as exc:
            report_error(exc)

    @output
    @render.text
    def data_summary():
        confirmed = experiment.get().confirmed
        if confirmed is None:
            return ""
        return (f"Data confirmed: {len(confirmed['data'])} complete observations "
                f"({confirmed['dropped_rows']} rows removed).\n"
                f"Treatment: {confirmed['treatment_col']} — 0 = {confirmed['treatment_mapping']['0']}, "
                f"1 = {confirmed['treatment_mapping']['1']}\nOutcome: {confirmed['outcome_col']}")

    @reactive.Effect
    @reactive.event(input.learn_model)
    def train_model():
        old = experiment.get()
        experiment.set(ExperimentSession(confirmed=old.confirmed))
        try:
            experiment.set(experiment.get().train(seed=int(input.train_seed()),
                holdout_frac=float(input.holdout_frac()), epochs=int(input.epochs()),
                batch_size=int(input.batch_size()) or None))
        except Exception as exc:
            report_error(exc)

    # Training output
    @output
    @render.text
    def training_output():
        state = experiment.get().trained
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
        state = experiment.get().trained
        if state is None:
            return None

        history = state['model'].history_
        epochs = [h['epoch'] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

        axes[0].plot(epochs, [h['neg_elbo'] for h in history],
                     color='#2855a6', linewidth=2)
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

    @reactive.Effect
    @reactive.event(input.generate_data)
    def generate_data():
        old = experiment.get()
        experiment.set(ExperimentSession(confirmed=old.confirmed, trained=old.trained, grid=old.grid))
        try:
            experiment.set(experiment.get().generate(n=int(input.n_sim()), tau0=float(input.tau()),
                rho=float(input.rho()), seed=int(input.gen_seed()), heterogeneity=float(input.heterogeneity()),
                mechanism=input.mechanism(), overlap_strength=float(input.overlap_strength())))
        except Exception as exc:
            report_error(exc)

    # Generation output
    @output
    @render.text
    def generation_output():
        df = (experiment.get().run['result']['data'] if experiment.get().run is not None else None)
        extras = experiment.get().run
        if df is None or extras is None:
            return ""
        result = extras['result']
        ov = result['overlap']
        tau_sd = float(np.std(result['tau_true']))
        effect_line = ("  effect: constant at {:.3f}".format(result['true_ate'])
                       if tau_sd == 0 else
                       "  effect: mean {:.3f}, sd {:.3f}, range [{:.3f}, {:.3f}]".format(
                           result['true_ate'], tau_sd,
                           float(np.min(result['tau_true'])),
                           float(np.max(result['tau_true']))))
        if ov['kind'] == 'unavailable':
            overlap_line = "  observed-covariate overlap: not available for the learned mechanism"
        else:
            overlap_line = (f"  assignment probability {ov['kind']}: [{ov['ps_min']:.3f}, {ov['ps_max']:.3f}]\n"
                            f"  extreme assignment probabilities (<0.05 or >0.95): {ov['pct_extreme']:.1%}")
        return (
            f"Synthetic data generated: {len(df)} observations, {df.shape[1]} observed columns.\n"
            f"  actual cohort ATE: {result['true_ate']:.3f} | baseline effect: {extras['parameters']['tau']:.3f}\n"
            f"  all {result['decoded'].shape[1]} covariates exported; answer key is separate\n"
            f"{effect_line}\n{overlap_line}\n  treated fraction: {ov['treated_frac']:.3f}")

    # The f(X) weights, shown rather than hidden. This is the observed-confounding
    # mechanism in full: if a covariate has weight 0 it cannot confound anything.
    @output
    @render.ui
    def fx_weights():
        state = experiment.get().trained
        if state is None or (experiment.get().run['result']['data'] if experiment.get().run is not None else None) is None:
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
                        "Equal weights are used here.")
            )
        )

    # The realism verdict.
    @output
    @render.ui
    def realism_results():
        extras = experiment.get().run
        state = experiment.get().trained
        if extras is None or state is None:
            return None

        report = extras['realism']
        disc = report['discriminator']
        covariates = report['covariates'].copy()

        auc = disc['auc']
        discrimination = max(auc, 1 - auc)
        colour = "#1c8b4a" if discrimination < 0.60 else ("#fd7e14" if discrimination < 0.80 else "#dc3545")
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
                    ui.tags.div({"class": "metric-label"}, "0.5 = chance for this diagnostic")
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
                {"class": "teaching-note", "style": "border-left-color: {};".format(colour)},
                ui.HTML("<strong>Verdict:</strong> {}".format(disc['interpretation']))
            ),
            ui.HTML(display.to_html(index=False, border=0,
                                    classes="table table-sm", justify="left")),
            ui.tags.p(
                {"class": "text-muted", "style": "font-size: 0.9em;"},
                ui.HTML("Sorted by |SMD|, largest first. Inspect covariates with the largest discrepancies.")
            )
        )

    # The same estimator comparison used by the guided lessons.
    @reactive.calc
    def advanced_comparison():
        run = experiment.get().run
        if run is None:
            return None
        try:
            return compare_estimates(run)
        except ValueError:
            return None

    @output
    @render.ui
    def analysis_results():
        if experiment.get().run is None:
            return None
        comparison = advanced_comparison()
        if comparison is None:
            return ui.p("Both treatment groups are needed for this comparison. Generate another cohort.")
        display = comparison[['method', 'estimate', 'error']].copy()
        display.columns = ['Method', 'Estimated effect', 'Error vs truth']
        for col in ['Estimated effect', 'Error vs truth']:
            display[col] = display[col].map(lambda v: f"{v:+.2f}")
        return ui.TagList(
            ui.h4("Compare estimates with known truth"),
            ui.p(f"True cohort ATE: {comparison['truth'].iloc[0]:+.2f} outcome units."),
            ui.HTML(display.to_html(index=False, border=0, classes='table')),
            ui.p("Adjustment uses the exported measured covariates and a linear outcome model with "
                 "treatment interactions. It matches the outcome mechanism when hidden confounding is off. These are point "
                 "estimates from one cohort, not confidence intervals or repeated-run bias."),
        )

    @output
    @render.plot
    def comparison_plot_output():
        comparison = advanced_comparison()
        if comparison is not None:
            return comparison_plot(comparison)

    # Outcome plot
    @output
    @render.plot
    def outcome_plot():
        df = (experiment.get().run['result']['data'] if experiment.get().run is not None else None)
        if df is None:
            return None

        y_col = experiment.get().run['outcome_col']
        z_col = experiment.get().run['treatment_col']

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot density by treatment group.
        for z_val, color, label in [(0, '#6c757d', 'Control'), (1, '#2855a6', 'Treated')]:
            df_sub = df[df[z_col] == z_val][y_col]
            if len(df_sub) > 1 and df_sub.std() > 0:
                df_sub.plot.density(ax=ax, color=color, alpha=0.7, label=label, linewidth=2)

        ax.set_xlabel('Outcome (Y)')
        ax.set_ylabel('Density')
        ax.set_title('Outcome Distributions by Treatment Group')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    @reactive.Effect
    @reactive.event(input.run_grid)
    def run_grid():
        try:
            experiment.set(experiment.get().run_grid(tau0=float(input.tau()),
                                                     n=int(input.grid_n()), seed=int(input.grid_seed())))
        except Exception as exc:
            report_error(exc)

    @output
    @render.text
    def grid_status():
        grid = experiment.get().grid
        if grid is None:
            return ""
        return "{} cells generated with common random draws for covariates and outcome noise. Estimates use one cohort per cell.".format(
            len(grid))

    @output
    @render.ui
    def grid_results():
        grid = experiment.get().grid
        if grid is None:
            return None

        display = grid[['overlap', 'heterogeneity', 'hidden_bias', 'true_ate',
                        'naive_error', 'gcomp_error', 'pct_extreme', 'cate_corr']].copy()
        display['pct_extreme'] = display['pct_extreme'].map(lambda v: "{:.1%}".format(v))
        display['cate_corr'] = display['cate_corr'].map(
            lambda v: "--" if pd.isna(v) else "{:.3f}".format(v))
        for col in ('true_ate', 'naive_error', 'gcomp_error'):
            display[col] = display[col].map(lambda v: "{:+.3f}".format(v))
        display.columns = ['Overlap', 'Effect', 'Hidden bias', 'True ATE',
                           'Naive error', 'g-comp error', '% extreme', 'CATE r']

        # Summarize the displayed results.
        lines = summarize_grid(grid)

        return ui.TagList(
            ui.HTML(display.to_html(index=False, border=0,
                                    classes="table table-sm", justify="left")),
            ui.tags.div(
                {"class": "teaching-note"},
                ui.tags.ul(*[ui.tags.li(line) for line in lines])
            ),
            ui.tags.p(
                {"class": "text-muted", "style": "font-size: 0.9em;"},
                ui.HTML("<strong>CATE r</strong> is the correlation between the estimated "
                        "and true per-patient effect &mdash; undefined when the effect is "
                        "constant. Average-effect comparisons remain meaningful in that case.")
            )
        )

    @output
    @render.ui
    def download_controls():
        if experiment.get().run is None:
            return None
        return ui.TagList(ui.download_button("download_data", "Download Observed Data (.csv)", class_="btn-success"),
                          ui.download_button("download_answers", "Download Data + Answer Key (.zip)", class_="btn-secondary"))

    @output
    @render.ui
    def grid_download_controls():
        if experiment.get().grid is None:
            return None
        return ui.download_button("download_grid", "Download Grid (.csv)", class_="btn-success")

    @session.download(
        filename=lambda: f"tao_rwd_scenario_grid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def download_grid():
        grid = experiment.get().grid
        if grid is None:
            yield ""
            return
        yield grid.to_csv(index=False)

    # Download handler
    @session.download(
        filename=lambda: f"tao_rwd_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def download_data():
        df = (experiment.get().run['result']['data'] if experiment.get().run is not None else None)
        if df is None:
            yield ""
            return

        # Convert to CSV and yield as string
        csv_string = df.to_csv(index=False)
        yield csv_string


    @session.download(filename=lambda: "tao_rwd_data_and_answers.zip", media_type="application/zip")
    def download_answers():
        run = experiment.get().run
        if run is not None:
            yield answer_bundle(run)
