"""GPExplore as a Python library: single runs, experiments and exports without the GUI.

    import gpexplore as gpx

    result = gpx.run("Pagie-1", selection_method="ROIDS", generations=30, seed=1)
    print(result.best_expression, result.best_validation)
    result.save_plot("pagie.png")

Everything saved in the GUI can be used by name: custom problems and data sets, custom
operators (e.g. selection_method="my_selection") and the switched-on custom functions.
use_problem, use_operator and use_function make further ones available for one session (e.g.
in an experiment script exported from the app's Experiments tab).
Parameter names are those of the GUI settings (see gpexplore.parameters()).
"""

import math
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import config
import experiments
import gp
import operators
import plots
import problems
import synthesis

_NOT_PARAMETERS = {"target_function", "show_advanced", "expert_mode", "exp_runs", "exp_workers", "exp_seed",
                   "plot_adjust", "inter_gen_pause", *config.WINDOW_KEYS}


# --------------------------------------------------------------------------
# Problems and parameters
# --------------------------------------------------------------------------

def problem_names():
    """Names of all problems: built-in, custom and imported data sets."""
    return [problems.title(p) for p in problems.REGISTRY.values()]


def parameters():
    """Parameter names with their default values."""
    return {k: v for k, v in config.DEFAULTS.items() if k not in _NOT_PARAMETERS}


def gui_settings():
    """The parameters last used in the GUI (settings.json), e.g. as `base` of run() or Experiment."""
    saved = dict(config.DEFAULTS)
    current = config.snapshot()
    config.load_settings()
    saved.update(config.snapshot())
    for key, value in current.items():   # leave the module state as it was
        setattr(config, key, value)
    return {k: v for k, v in saved.items() if k not in _NOT_PARAMETERS}


def use_problem(entry):
    """Make a custom problem available for this session: a dict as in custom_problems.json
    (formula, Python program, logic expression or data set). Returns its name."""
    return problems.title(problems.register(entry))


def use_operator(kind, code):
    """Make a custom operator available for this session (kind: "fitness", "selection",
    "crossover", "mutation", "initialization", "bloat" or "control"). Returns its name."""
    name = operators.check(kind, code)
    operators.REGISTRY[kind][name] = code
    return name


def use_function(area, code):
    """Make a custom function available (and switched on) for this session (area: "sr",
    "ps" or "logic"). Returns its name."""
    name = operators.check_function(area, code)[0]
    operators.FUNCTIONS[area][name] = code
    operators.ENABLED[name] = True
    return name


def find_problem(name):
    """A problem by its name as shown in the GUI (e.g. "Koza-1", "Pagie-1", "Small or Large",
    the name of an imported data set) or by its expression."""
    if name in problems.REGISTRY:
        return problems.REGISTRY[name]
    for p in problems.REGISTRY.values():
        if problems.title(p) == name:
            return p
    for p in problems.REGISTRY.values():
        if p.name == name:
            return p
    raise KeyError(f"Unknown problem {name!r}. Available: {', '.join(problem_names())}")


def _data(problem, train, test, noise, outliers, seed):
    """Training and validation/test data; reproducible with a seed."""
    if problem.kind != "sr":
        try:
            tr, te = problem.generate(train, test, seed=seed)
        except TypeError:
            tr, te = problem.generate(train, test)
        return [i for i, _ in tr], [o for _, o in tr], [i for i, _ in te], [o for _, o in te]
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        return problems.make_data(problem, train, test, noise, outliers)
    finally:
        np.random.set_state(state)


def settings(problem, base=None, data_seed=0, functions=None, **params):
    """Complete settings of a run: the defaults (or `base`, e.g. gui_settings()), the problem's
    data and building blocks as when it is selected in the GUI, then `params`. `functions`:
    names of the custom functions to use (default: all switched-on ones)."""
    p = find_problem(problem)
    unknown = set(params) - set(parameters())
    if unknown:
        raise ValueError(f"Unknown parameter(s) {', '.join(sorted(unknown))}; see gpexplore.parameters()")
    s = dict(config.DEFAULTS)
    if base:
        s.update({k: v for k, v in base.items() if k in config.DEFAULTS})
    for flag in synthesis.GROUP_FLAGS:          # building blocks of the problem kind
        s[flag] = p.kind == "ps"
    for flag in synthesis.GATE_FLAGS:
        s[flag] = p.kind == "logic" and flag in getattr(p, "gates", ())
    s.update(params)
    # data the problem fixes itself (as in the app, whatever the parameters say)
    if p.kind == "ps" or getattr(p, "dataset", False):
        s["training_set_size"], s["validation_set_size"] = p.train_size, p.test_size
    elif p.kind == "logic":
        s["training_set_size"] = p.train_size
    if p.kind != "sr" or getattr(p, "dataset", False):
        s["noise_intensity"] = s["outlier_share"] = 0.0
    for key in operators.CHOICES:               # names used before a built-in choice was renamed
        s[key] = operators.canonical(s[key])
    for key, allowed in operators.OPTIONS.items():
        if s[key] not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(map(repr, allowed))} (got {s[key]!r})")
    for kind, op in operators.KINDS.items():    # custom operators by name
        if s[op.setting] not in operators.BUILTIN[kind] and s[op.setting] not in operators.REGISTRY[kind]:
            raise ValueError(f"Unknown {op.title.lower()} {s[op.setting]!r}")
    s["target_function"] = p.expression
    (s["training_X"], s["training_y"], s["validation_X"], s["validation_y"]) = _data(
        p, int(s["training_set_size"]), int(s["validation_set_size"]), s["noise_intensity"], s["outlier_share"],
        data_seed)
    s["problem"] = p
    if p.custom:   # worker processes may not know it
        s["problem_entry"] = problems.entry_of(p)
    s["custom_operators"] = operators.selected(s)
    if functions is None:
        s["custom_functions"] = operators.selected_functions()
    else:
        unknown = [n for n in functions if not any(n in operators.FUNCTIONS[a] for a in operators.FUNCTIONS)]
        if unknown:
            raise ValueError(f"Unknown custom function(s) {', '.join(unknown)}")
        s["custom_functions"] = {area: [code for name, code in operators.FUNCTIONS[area].items() if name in functions]
                                 for area in operators.FUNCTIONS}
    s["custom_library"] = operators.library()
    return s


# --------------------------------------------------------------------------
# Single runs
# --------------------------------------------------------------------------

PLOT_VIEWS = ["error", "fit", "tree"] + list(plots.DYNAMICS_VIEWS)


class Result:
    """Outcome of one run: best individual, its history and the population dynamics."""

    def __init__(self, gp_run, run_settings):
        self.settings = run_settings
        self.snapshot = gp_run.snapshot()
        if self.snapshot.state == "error":
            raise RuntimeError(self.snapshot.error)
        snap = self.snapshot
        self.best_expression = snap.best_expression
        self.best_size, self.best_depth = snap.best_size, snap.best_depth
        self.best_training = snap.best_training[-1] if snap.best_training else math.nan
        self.best_validation = snap.best_validation[-1] if snap.best_validation else math.nan
        self.history = {"best_training": list(snap.best_training), "best_validation": list(snap.best_validation),
                        "avg_training": list(snap.avg_training), "avg_validation": list(snap.avg_validation)}

    @property
    def series(self):
        """Every recorded value per generation (the columns of the raw CSV export)."""
        return experiments.run_series({"train": self.history["best_training"],
                                       "valid": self.history["best_validation"],
                                       "avg_train": self.history["avg_training"],
                                       "avg_valid": self.history["avg_validation"],
                                       "dynamics": self.snapshot.dynamics})

    def save_plot(self, path, view="error", size=(700, 450)):
        """Save a plot as PNG, PDF or SVG (by the file extension). view: "error" (curves of the
        best and average individual), "fit" (problem view), "tree" or a dynamics view
        ("fitness", "size", "diversity", "selection", "variation", "blocks")."""
        if view not in PLOT_VIEWS:
            raise ValueError(f"view must be one of {', '.join(PLOT_VIEWS)}")
        s, snap, problem = self.settings, self.snapshot, self.settings["problem"]
        fmt = os.path.splitext(path)[1].lstrip(".").lower() or "png"
        with plots.output_format(fmt):
            if view == "error":
                data = plots.render_error_plot(snap, s["generations"], True, size,
                                               holdout={"sr": "validation", "ps": "test"}.get(problem.kind),
                                               percent=problem.kind != "sr")
            elif view == "fit":
                data = plots.render_fit_plot(snap, problem, s["training_X"], s["training_y"], True, size,
                                             test_size=len(s["validation_y"]), test_y=s["validation_y"])
            elif view == "tree":
                data = plots.render_tree(snap.best_nodes)[0]
            else:
                data = plots.render_dynamics(view, snap, s["generations"], size, problem.kind)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def __repr__(self):
        return (f"<Result {problems.title(self.settings['problem'])}: {self.best_expression} "
                f"(training {self.best_training:.4g}, validation {self.best_validation:.4g})>")


def run(problem, base=None, data_seed=0, functions=None, **params):
    """One GP run, e.g. run("Koza-1", selection_method="Lexicase", generations=50, seed=1)."""
    s = settings(problem, base, data_seed, functions, **params)
    gp_run = gp.GPRun(s, headless=True)
    gp_run._run()
    return Result(gp_run, s)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def _print_progress(done, total, seconds):
    """Default progress of Experiment.run: one line on stderr, updated in place."""
    print(f"\r{done} / {total} runs  ·  {seconds:.0f} s", end="\n" if done == total else "", file=sys.stderr,
          flush=True)


class Experiment:
    """Repeated runs of several configurations on one or more problems, run in parallel.
    Scripts that call Experiment.run() need `if __name__ == "__main__":` (worker processes)."""

    def __init__(self, problems_, runs=30, seed=1, workers=None, base=None):
        names = [problems_] if isinstance(problems_, str) else list(problems_)
        self.problems = [find_problem(n) for n in names]
        self.runs, self.seed, self.base = int(runs), int(seed), base
        self.workers = workers or max(1, experiments.cpu_cores() - 1)
        self.configs = []      # (label, params, problem it was made for, custom functions)
        self.results = {}      # problem expression -> per configuration a list of run results

    def add(self, label, problem=None, functions=None, **params):
        """A configuration: the parameters that differ from the defaults (or `base`).
        `problem`: the problem the configuration was made for (as in the app, where a configuration
        is added on the selected problem). On it, all parameters apply; on other problems the
        building blocks of program synthesis and Boolean logic are set for that problem.
        `functions`: names of the custom functions to use (default: all switched-on ones)."""
        unknown = set(params) - set(parameters())
        if unknown:
            raise ValueError(f"Unknown parameter(s) {', '.join(sorted(unknown))}; see gpexplore.parameters()")
        origin = find_problem(problem) if problem is not None else None
        self.configs.append((label, params, origin, functions))
        return self

    def _settings(self, p, params, origin, functions):
        """Settings of a configuration on problem p (the app's rules for other problems)."""
        if origin is None or origin.expression == p.expression:
            return settings(p.expression, self.base, self.seed, functions, **params)
        own = settings(origin.expression, self.base, self.seed, functions, **params)
        params = dict(params)
        if p.kind == "ps" or origin.kind == "ps":
            for flag in synthesis.GROUP_FLAGS:
                params.pop(flag, None)
        else:
            params.update({flag: own[flag] for flag in synthesis.GROUP_FLAGS})
        if p.kind == "logic":
            for flag in synthesis.GATE_FLAGS:
                params.pop(flag, None)
        else:
            params.update({flag: own[flag] for flag in synthesis.GATE_FLAGS})
        return settings(p.expression, self.base, self.seed, functions, **params)

    @property
    def labels(self):
        return [label for label, *_ in self.configs]

    def run(self, progress=True):
        """Run r of every configuration uses seed `seed + r`; all configurations of a problem
        get the same data. `progress`: True for a line on stderr, False for none, or a function
        progress(done, total, seconds) called at the start and after every finished run."""
        if not self.configs:
            raise ValueError("Add at least one configuration first.")
        report = progress if callable(progress) else (_print_progress if progress else None)
        jobs = []
        for p in self.problems:
            per_config = [self._settings(p, params, origin, functions) for _, params, origin, functions in self.configs]
            for r in range(self.runs):
                for ci, s in enumerate(per_config):
                    jobs.append((p.expression, ci, r, (experiments.job_settings(s), p.expression, self.seed + r)))
        self.results = {p.expression: [[None] * self.runs for _ in self.configs] for p in self.problems}
        started, done = time.time(), 0
        if report:
            report(0, len(jobs), 0.0)
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(experiments.run_one, payload): (key, ci, r) for key, ci, r, payload in jobs}
            for future in as_completed(futures):
                key, ci, r = futures[future]
                self.results[key][ci][r] = future.result()
                done += 1
                if report:
                    report(done, len(jobs), time.time() - started)
        return self

    def groups(self, problem):
        key = find_problem(problem).expression
        return [[r for r in runs if r is not None] for runs in self.results.get(key, [])]

    def analysis(self, problem):
        """Statistics of one problem (summary per configuration, pairwise tests), as in the GUI."""
        p = find_problem(problem)
        return experiments.analyse(self.groups(p.expression), self.labels, p.kind)

    def _sets(self):
        return [(problems.title(p), p.kind, self.groups(p.expression)) for p in self.problems
                if self.results.get(p.expression)]

    def summary(self):
        """The results as text: per problem median, IQR and successes per configuration and the
        pairwise tests (Mann-Whitney U, Fisher, Holm-corrected, α = 0.05)."""
        return "\n".join(text for _, text, *_ in self._overview())

    def _overview(self):
        lines = []
        for p in self.problems:
            if not self.results.get(p.expression):
                continue
            a = self.analysis(p.expression)
            metric, higher = experiments.primary_metric(p.kind)
            lines.append(("text", f"{problems.title(p)}  ({problems.KIND_TITLES[p.kind]}; {metric}, "
                                  f"{'higher' if higher else 'lower'} is better)"))

            def fmt(v):
                if v == math.inf:
                    return "inf"
                if not math.isfinite(v):
                    return "–"
                return f"{v:.4g}" if p.kind == "sr" else f"{v:.1f} %"
            for row in a.summary:
                text = f"    {row['label']:<22} median {fmt(row['median']):<10} " \
                       f"IQR {(fmt(row['q1']) + ' – ' + fmt(row['q3'])):<22}"
                if p.kind != "sr":
                    text += f" success {row['successes']}/{row['n']}"
                if row["seconds"] == row["seconds"]:   # not NaN
                    text += f"  {row['seconds']:.1f} s per run"
                lines.append(("mono", text))
            for pair in a.pairs:
                text = f"    {pair['a']} vs. {pair['b']}: p = {pair['p_holm']:.3g}, Â12 = {pair['a12']:.2f}"
                if p.kind != "sr":
                    text += f", Fisher p = {pair['fisher_p_holm']:.3g}"
                lines.append(("mono", f"{text}  →  {pair['verdict']}",
                              plots.style.ACCENT if pair["significant"] else plots.style.INK))
            if a.omnibus is not None:
                lines.append(("mono", f"    Kruskal-Wallis: H = {a.omnibus['H']:.2f}, p = {a.omnibus['p']:.3g}"))
        return lines

    def to_csv(self, path, raw=False):
        """All results as CSV: aggregated over the runs per generation, or raw (every value of every run)."""
        data = (experiments.to_csv_raw if raw else experiments.to_csv_aggregated)(self._sets(), self.labels)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def to_zip(self, path):
        """Every plot of every problem as PNG (300 dpi) and PDF, in one ZIP file."""
        generations = max(settings_value(self.base, params, "generations") for _, params, *_ in self.configs)
        with open(path, "wb") as f:
            f.write(plots.experiment_plots_zip(self._sets(), self.labels, generations))
        return path

    def to_pdf(self, path):
        """All plots of all problems with headings and explanations, plus an overview page."""
        labels = self.labels
        generations = max(settings_value(self.base, params, "generations") for _, params, *_ in self.configs)
        lines = [("h1", "GPExplore experiment"),
                 ("text", f"{time.strftime('%Y-%m-%d %H:%M')}  ·  {self.runs} runs per configuration  ·  seeds "
                          f"{self.seed}–{self.seed + self.runs - 1}  ·  {len(self.problems)} problem(s)",
                  plots.style.MUTED),
                 ("h2", "Configurations")]
        for i, (label, params, *_) in enumerate(self.configs):
            described = ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"
            lines.append(("swatch", f"{label:<22} {described}", plots.style.CATEGORICAL[i % 8]))
        lines.append(("h2", "Results  (pairwise: Mann-Whitney U and, for program synthesis / logic, Fisher's "
                            "exact test; Holm-corrected; α = 0.05)"))
        lines += self._overview()
        sections = []
        for name, kind, groups in self._sets():
            sub = f"{problems.KIND_TITLES[kind]}  ·  {self.runs} runs per configuration  ·  configurations: " \
                  + ", ".join(labels)
            sections.append((name, sub, plots.experiment_report_plots(kind, groups, labels, generations)))
        with open(path, "wb") as f:
            f.write(plots.render_experiment_report(lines, sections))
        return path


def settings_value(base, params, key):
    """A parameter of a configuration: its own value, else `base`, else the default."""
    if key in params:
        return params[key]
    if base and key in base:
        return base[key]
    return config.DEFAULTS[key]


__all__ = ["run", "Result", "Experiment", "settings", "parameters", "gui_settings", "problem_names",
           "find_problem", "PLOT_VIEWS"]
