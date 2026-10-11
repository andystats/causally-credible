"""Tao of RWD causal inference lab: guided causal experiments and an advanced workspace."""
from pathlib import Path

import pandas as pd
from shiny import App, ui, render, reactive

from advanced import advanced_ui, advanced_server, custom_css
from comparison import comparison_plot
from lessons import LESSONS, PREDICTIONS, prediction_choices, prepare_lessons, run_lesson, experiment_record
from workflow import answer_bundle


def lesson_inputs(key, lesson):
    return ui.panel_conditional(
        f"input.g_lesson === '{key}'",
        ui.tags.div(
            {"class": "lesson-intro"},
            ui.tags.p({"class": "eyebrow"}, f"LESSON {list(LESSONS).index(key) + 1} OF 3"),
            ui.h2(lesson.question),
            ui.p({"class": "lesson-change"}, lesson.change),
        ),
        ui.tags.div(
            {"class": "prediction-box"},
            ui.tags.span({"class": "step-kicker"}, "01 / PREDICT"),
            ui.input_select(f"g_prediction_{key}", lesson.prediction,
                            choices=prediction_choices(key), selected="", width="100%"),
        ),
    )


def reflection_input(key, lesson):
    return ui.panel_conditional(
        f"input.g_lesson === '{key}'",
        ui.input_text_area(f"g_reflection_{key}", lesson.reflection, rows=3,
                           placeholder="Use the estimates to explain your prediction.",
                           width="100%"),
    )


guided_ui = ui.tags.div(
    {"class": "guided-layout"},
    ui.tags.aside(
        {"class": "lesson-sidebar", "aria-label": "Lesson navigation"},
        ui.input_radio_buttons("g_lesson", "Lessons",
                               choices={key: f"0{i + 1}  {lesson.title}" for i, (key, lesson) in enumerate(LESSONS.items())},
                               selected="randomized"),
        ui.output_ui("g_progress"),
        ui.tags.div(
            {"class": "study-card"},
            ui.tags.p({"class": "eyebrow"}, "THE STUDY"),
            ui.h3("Does a training program help?"),
            ui.p("Compare program versus no program."),
            ui.tags.dl(
                ui.tags.dt("Outcome"), ui.tags.dd("Performance score change (points). Higher is better."),
                ui.tags.dt("Measured characteristics"), ui.tags.dd("Baseline score, years of experience, prior training."),
                ui.tags.dt("Source"), ui.tags.dd("Simulated for teaching; not evidence about a real program."),
            ),
        ),
        ui.tags.p({"class": "muted small"}, "Use Advanced for your own data and settings."),
    ),
    ui.tags.main(
        {"class": "lesson-main", "aria-label": "Guided experiment"},
        *[lesson_inputs(key, lesson) for key, lesson in LESSONS.items()],
        ui.tags.div(
            {"class": "run-row"},
            ui.input_action_button("g_run", "Run this experiment", class_="btn-primary"),
            ui.tags.p({"class": "muted small"}, "The first run takes a little longer."),
        ),
        ui.output_ui("g_status"),
        ui.tags.section(
            {"class": "compare-card", "aria-label": "Estimate comparison"},
            ui.tags.div({"class": "section-heading"},
                        ui.tags.div(ui.tags.span({"class": "step-kicker"}, "02 / COMPARE"),
                                    ui.h3("Compare estimates"))),
            ui.output_ui("g_result_summary"),
            *[ui.panel_conditional(f"input.g_lesson === '{key}'",
                                  ui.input_checkbox(f"g_reveal_{key}", "Reveal the true effect", value=False))
              for key in LESSONS],
            ui.output_plot("g_plot", height="280px"),
            ui.output_ui("g_estimate_table"),
            ui.p({"class": "muted small"}, "Adjustment uses the three measured characteristics in a linear model with treatment interactions."),
        ),
        ui.tags.section(
            {"class": "explain-card"},
            ui.tags.span({"class": "step-kicker"}, "03 / EXPLAIN"),
            ui.h3("Explain the result"),
            ui.output_ui("g_debrief"),
            *[reflection_input(key, lesson) for key, lesson in LESSONS.items()],
            ui.p({"class": "muted small"}, "Save the experiment record to keep your notes after this session."),
            ui.output_ui("g_download_controls"),
            ui.input_action_button("g_next", "Next lesson →", class_="btn-outline-primary"),
        ),
        ui.tags.details(
            ui.tags.summary("How the simulation works"),
            ui.tags.div({"class": "ingredient-grid"},
                ui.tags.div(ui.h4("Learned"), ui.p("The CVAE learns covariate patterns from the simulated source cohort.")),
                ui.tags.div(ui.h4("Chosen"), ui.p("We choose assignment, outcomes, and the effect. Each lesson reuses 2,000 covariate and noise draws.")),
                ui.tags.div(ui.h4("Known"), ui.p("The simulator records both potential outcomes. A real study observes only one per person."))),
            ui.output_ui("g_fidelity"),
            ui.p("A good covariate match does not validate the chosen causal mechanism. One cohort shows estimation error; repeated cohorts are needed to assess bias."),
        ),
        ui.tags.details(ui.tags.summary("Completed experiments"), ui.output_ui("g_history")),
    ),
)

app_ui = ui.page_fluid(
    ui.tags.title("Tao of RWD | Causal inference lab"),
    ui.HTML(custom_css),
    ui.tags.style((Path(__file__).with_name('teaching.css')).read_text()),
    ui.tags.script("""
      $(function () {
        let previousLesson;
        $(document).on('shiny:inputchanged', function (event) {
          if (event.name !== 'g_lesson') return;
          if (previousLesson !== undefined && previousLesson !== event.value) {
            requestAnimationFrame(function () {
              const heading = document.querySelector('.lesson-main');
              if (heading) heading.scrollIntoView({block: 'start', behavior: 'instant'});
            });
          }
          previousLesson = event.value;
        });
      });
    """),
    ui.tags.header(
        {"class": "lab-header"},
        ui.tags.p({"class": "eyebrow"}, "Tao of RWD"),
        ui.h1("Causal inference lab"),
        ui.p("Predict. Compare. Explain."),
    ),
    ui.navset_tab(
        ui.nav_panel("Guided lessons", guided_ui, value="guided"),
        ui.nav_panel("Advanced workspace", advanced_ui, value="advanced"),
        id="workspace", selected="guided",
    ),
    ui.tags.footer({"class": "lab-footer"},
                   "Tao of RWD · Based on the Credence framework"),
)


def server(input, output, session):
    # Advanced uploads and settings cannot change the fixed guided experiments.
    advanced_server(input, output, session)
    trained = reactive.Value(None)
    completed = reactive.Value({})
    error = reactive.Value("")

    @reactive.calc
    def current_result():
        return completed.get().get(input.g_lesson())

    @reactive.calc
    def revealed():
        return bool(input[f"g_reveal_{input.g_lesson()}"]())

    @reactive.effect
    @reactive.event(input.g_lesson)
    def clear_error():
        error.set("")
        ui.update_action_button('g_next', label=('Return to lesson 1' if input.g_lesson() == 'hidden' else 'Next lesson →'))

    @reactive.effect
    @reactive.event(input.g_run)
    def run_experiment():
        key = input.g_lesson()
        prediction = input[f'g_prediction_{key}']()
        error.set("")
        if not prediction:
            error.set('Choose a prediction, or select “I’m not sure yet”.')
            return
        try:
            with ui.Progress(min=0, max=3) as progress:
                progress.set(0, message="Preparing your experiment", detail="Learning the source cohort once…")
                if trained.get() is None:
                    trained.set(prepare_lessons())
                progress.set(1, message="Generating the cohort", detail="Keeping common draws across lessons…")
                result = run_lesson(trained.get(), key, prediction)
                progress.set(3, message="Comparing estimates")
                completed.set({**completed.get(), key: result})
        except Exception as exc:
            error.set(f'Could not complete this experiment: {exc}')

    @reactive.effect
    @reactive.event(input.g_next)
    def next_lesson():
        keys = list(LESSONS)
        position = keys.index(input.g_lesson())
        ui.update_radio_buttons('g_lesson', selected=keys[(position + 1) % len(keys)])

    @output
    @render.ui
    def g_progress():
        count = len(completed.get())
        return ui.tags.p({"class": "progress-note", "role": "status"}, f"{count} of 3 experiments completed")

    @output
    @render.ui
    def g_status():
        if error.get():
            return ui.tags.div({"class": "status-note error-note", "role": "alert"}, error.get())
        if current_result() is not None:
            return ui.tags.div({"class": "status-note", "role": "status"},
                               "Experiment ready · 2,000 simulated people")
        return ui.tags.p({"class": "muted small", "role": "status"}, "Make a prediction, then run the experiment.")

    @output
    @render.ui
    def g_result_summary():
        result = current_result()
        if result is None:
            return ui.tags.div({"class": "empty-state"}, "Run the experiment to see estimates.")
        text = (f"True cohort effect: {result['comparison']['truth'].iloc[0]:+.2f} score points."
                if revealed() else "Compare the estimates, then reveal the true effect.")
        return ui.TagList(ui.p({"class": "truth-note"}, text),
                          ui.p({"class": "muted small"}, "One cohort; point estimates without confidence intervals."))

    @output
    @render.plot
    def g_plot():
        result = current_result()
        if result is not None:
            return comparison_plot(result['comparison'], reveal=revealed(), units='score points')

    @output
    @render.ui
    def g_estimate_table():
        result = current_result()
        if result is None:
            return None
        columns = ['method', 'estimate'] + (['error'] if revealed() else [])
        table = result['comparison'][columns].copy()
        for col in columns[1:]:
            table[col] = table[col].map(lambda value: f'{value:+.2f}')
        table.columns = ['Method', 'Effect (points)'] + (['Error vs truth'] if revealed() else [])
        return ui.HTML(table.to_html(index=False, border=0, classes='table estimate-table'))

    @output
    @render.ui
    def g_debrief():
        result = current_result()
        if result is None:
            return ui.p("Run the experiment, then explain what you found.")
        prediction = PREDICTIONS[result['prediction']]
        if not revealed():
            return ui.p(f"Your prediction: {prediction}. What do you notice? Reveal truth when you are ready to check.")
        return ui.TagList(
            ui.p(ui.strong("Your prediction: "), prediction),
            ui.p(LESSONS[result['lesson']].explanation),
        )

    @output
    @render.ui
    def g_fidelity():
        result = current_result()
        if result is None:
            return ui.p("A held-out covariate check will appear after the experiment runs.")
        report = result['run']['realism']
        return ui.tags.div({"class": "status-note"},
            ui.strong("Check the generator: "),
            f"discriminator AUC {report['discriminator']['auc']:.3f}, "
            f"largest |SMD| {report['covariates']['smd'].abs().max():.3f}, "
            f"against {report['discriminator']['n_real']} held-out source rows. ",
            report['discriminator']['interpretation'],
        )

    @output
    @render.ui
    def g_history():
        rows = []
        for key, lesson in LESSONS.items():
            result = completed.get().get(key)
            if result:
                comparison = result['comparison']
                rows.append({'Lesson': lesson.title,
                             'Unadjusted': f"{comparison['estimate'].iloc[0]:+.2f}",
                             'Adjusted': f"{comparison['estimate'].iloc[1]:+.2f}"})
        if not rows:
            return ui.p("Run a lesson to start the comparison.")
        return ui.TagList(ui.p("Same source model, measured covariates, cohort size, effect and noise draws; different assignment rules."),
                          ui.HTML(pd.DataFrame(rows).to_html(index=False, border=0, classes='table')))

    @output
    @render.ui
    def g_download_controls():
        if current_result() is None:
            return None
        buttons = [ui.download_button('g_record', 'Save experiment record', class_='btn-outline-primary'),
                   ui.download_button('g_observed', 'Observed data (.csv)', class_='btn-outline-secondary')]
        if revealed():
            buttons.append(ui.download_button('g_answers', 'Data + answer key (.zip)', class_='btn-outline-secondary'))
        return ui.tags.div({"class": "download-row"}, *buttons)

    @session.download(filename=lambda: f"tao_rwd_{input.g_lesson()}_reflection.json", media_type='application/json')
    def g_record():
        result = current_result()
        if result is not None:
            yield experiment_record(result, input[f'g_reflection_{input.g_lesson()}'](), revealed())

    @session.download(filename=lambda: f"tao_rwd_{input.g_lesson()}_observed.csv")
    def g_observed():
        if current_result() is not None:
            yield current_result()['run']['result']['data'].to_csv(index=False)

    @session.download(filename=lambda: f"tao_rwd_{input.g_lesson()}_answers.zip", media_type='application/zip')
    def g_answers():
        if current_result() is not None and revealed():
            yield answer_bundle(current_result()['run'])


app = App(app_ui, server)
