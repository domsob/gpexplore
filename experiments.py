"""Repeated runs and their statistical analysis (Experiments tab).

Runs are executed in separate processes (one GP run per job, no display), so
several runs can use several CPU cores at the same time. Every run gets its own
seed; run r of every configuration uses the same seed, so the configurations
see the same random starts.
"""

import io
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def cpu_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# --------------------------------------------------------------------------
# One run (executed in a worker process)
# --------------------------------------------------------------------------

def run_one(job):
    """job = (settings without "problem", problem key, seed). Returns a plain dict."""
    settings, key, seed = job
    import gp          # imported in the worker process
    import problems
    settings = dict(settings)
    settings["problem"] = problems.get(key)
    settings["seed"] = seed
    started = time.time()
    run = gp.GPRun(settings, headless=True)
    run._run()
    snap = run.snapshot()
    kind = settings["problem"].kind
    final_train = snap.best_training[-1] if snap.best_training else math.nan
    final_valid = snap.best_validation[-1] if snap.best_validation else math.nan
    if kind == "logic":
        success = final_train >= 100.0 - 1e-9
    elif kind == "ps":
        success = final_valid >= 100.0 - 1e-9
    else:
        success = None
    return {
        "seed": seed, "seconds": time.time() - started, "state": snap.state, "error": snap.error,
        "train": list(snap.best_training), "valid": list(snap.best_validation),
        "final_train": final_train, "final_valid": final_valid, "success": success,
        "size": snap.best_size, "expression": snap.best_expression,
        "avg_train": list(snap.avg_training), "avg_valid": list(snap.avg_validation),
        "dynamics": list(snap.dynamics or []),
        "gen_best": [(rec["expression"], rec["size"], rec["depth"]) for rec in run._records],
    }


def job_settings(settings):
    """Settings that can be sent to a worker process (the problem goes by its key)."""
    return {k: v for k, v in settings.items() if k != "problem"}


# --------------------------------------------------------------------------
# Per-generation series of a run (plots and CSV export)
# --------------------------------------------------------------------------

QUANTILES = ("p10", "p25", "median", "p75", "p90")
SERIES_COLUMNS = (
    ["best_training", "best_validation", "avg_training", "avg_validation"]
    + [f"fitness_{q}" for q in QUANTILES]
    + [f"size_{q}" for q in QUANTILES] + ["size_max"]
    + [f"depth_{q}" for q in QUANTILES] + ["depth_max"]
    + ["genotypic_diversity", "phenotypic_diversity", "selected_at_least_once", "most_selected_share"]
    + [f"{op}_{effect}" for op in ("crossover", "mutation") for effect in ("better", "equal", "worse")]
)
CARRIED = ("best_training", "best_validation")   # the best found so far stays when a run ends early


def _f(v):
    return math.nan if v is None else float(v)


def _effect_share(counts, k):
    total = sum(counts) if counts else 0
    return 100.0 * counts[k] / total if total else math.nan


def run_series(r):
    """All per-generation values of one run as {column: list} (computed once per run)."""
    cached = r.get("_series")
    if cached is not None:
        return cached
    dyn = r.get("dynamics") or []
    s = {"best_training": [_f(v) for v in r["train"]], "best_validation": [_f(v) for v in r["valid"]],
         "avg_training": [_f(v) for v in r.get("avg_train", [])],
         "avg_validation": [_f(v) for v in r.get("avg_valid", [])]}
    for key in ("fitness", "size", "depth"):
        for i, q in enumerate(QUANTILES):
            s[f"{key}_{q}"] = [_f(d[key][i]) for d in dyn]
    for key in ("size", "depth"):
        s[f"{key}_max"] = [_f(d[key][5]) for d in dyn]
    s["genotypic_diversity"] = [_f(d["genotypic"]) for d in dyn]
    s["phenotypic_diversity"] = [_f(d["phenotypic"]) for d in dyn]
    s["selected_at_least_once"] = [_f(d["parents"]) for d in dyn]
    s["most_selected_share"] = [_f(d["top_parent"]) for d in dyn]
    for op, name in (("cx", "crossover"), ("mut", "mutation")):
        for k, effect in enumerate(("better", "equal", "worse")):
            s[f"{name}_{effect}"] = [_effect_share((d["variation"] or {}).get(op), k) for d in dyn]
    names = set()
    for d in dyn:
        names.update(d["blocks"])
    for name in names:
        s[f"block_{name}"] = [100.0 * d["blocks"].get(name, 0.0) for d in dyn]
    r["_series"] = s
    return s


def block_names(groups):
    names = set()
    for g in groups:
        for r in g:
            names.update(c[6:] for c in run_series(r) if c.startswith("block_"))
    return sorted(names)


def generations_of(groups):
    return max((len(r["train"]) for g in groups for r in g), default=0)


def series_matrix(results, column, length):
    """Runs x generations; a run that ended early keeps its best value, other values are missing."""
    rows = []
    for r in results:
        s = run_series(r)
        values = s.get(column)
        if values is None:   # building block never used in this run
            values = [0.0] * len(s["fitness_median"]) if column.startswith("block_") else []
        values = list(values)[:length]
        if not values:
            continue
        pad = values[-1] if column in CARRIED else math.nan
        rows.append(values + [pad] * (length - len(values)))
    return np.array(rows, dtype=float) if rows else np.empty((0, length))


def success_column(kind):
    return "best_training" if kind == "logic" else "best_validation"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def primary_metric(kind):
    """(name, higher is better) of the value compared between configurations."""
    if kind == "sr":
        return "validation error", False
    if kind == "ps":
        return "test cases solved (%)", True
    return "truth-table rows correct (%)", True


EXACT = 1e-10   # regression errors below this are exact solutions (floating-point noise)


def final_values(results, kind):
    key = "final_train" if kind == "logic" else "final_valid"
    values = np.array([r[key] for r in results], dtype=float)
    if kind == "sr":
        values[np.abs(values) < EXACT] = 0.0   # so that exact solutions tie in the rank tests
    return values


Z95 = float(stats.norm.ppf(0.975))


def wilson(successes, n, z=Z95):
    """95 % Wilson score interval of a success rate."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def a12(x, y, higher_is_better):
    """Vargha-Delaney A12: probability that a run of x beats a run of y (ties count half)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not len(x) or not len(y):
        return math.nan
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    better = gt if higher_is_better else (len(x) * len(y) - gt - eq)
    return (better + 0.5 * eq) / (len(x) * len(y))


def a12_magnitude(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def holm(pvalues):
    """Holm-Bonferroni adjusted p values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


@dataclass
class Analysis:
    metric: str
    higher_is_better: bool
    summary: list = field(default_factory=list)    # one dict per configuration
    pairs: list = field(default_factory=list)      # one dict per pair of configurations
    omnibus: dict = None                           # Kruskal-Wallis for three or more configurations


_BIG = np.finfo(float).max


def _quantiles(v, qs):
    """Percentiles that also work with infinite errors (diverged runs)."""
    if not len(v):
        return [math.nan] * len(qs)
    out = np.percentile(np.where(np.isinf(v), _BIG, v), qs)
    return [math.inf if q >= _BIG / 2 else float(q) for q in out]


def analyse(groups, labels, kind):
    """groups: list (per configuration) of run result dicts.

    Final values: validation error (regression; an infinite error counts as the worst value,
    it is kept in the rank tests), % test cases solved (program synthesis) or % truth-table
    rows correct (logic). Pairwise Mann-Whitney U tests (two-sided) with Vargha-Delaney A12;
    for program synthesis and logic additionally Fisher's exact test on the success rates.
    All pairwise p values of the table form one family for the Holm correction, so a pair
    may count as significant if either of its tests is significant. Runs that failed with an
    error (no value) are left out of the rank tests and count as unsuccessful."""
    metric, higher = primary_metric(kind)
    values = [final_values(g, kind) for g in groups]
    usable = [v[~np.isnan(v)] for v in values]
    out = Analysis(metric, higher)
    for label, g, ok in zip(labels, groups, usable):
        q1, median, q3 = _quantiles(ok, [25, 50, 75])
        finite = ok[np.isfinite(ok)]
        row = {"label": label, "n": len(g), "median": median, "q1": q1, "q3": q3,
               "mean": (math.inf if len(finite) < len(ok) else float(ok.mean())) if len(ok) else math.nan,
               "sd": float(ok.std(ddof=1)) if len(ok) > 1 and len(finite) == len(ok) else math.nan,
               "seconds": float(np.mean([r["seconds"] for r in g])) if g else math.nan}
        if kind != "sr":
            n_success = sum(1 for r in g if r["success"])
            row["successes"] = n_success
            row["success_ci"] = wilson(n_success, len(g))
        out.summary.append(row)

    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            x, y = usable[i], usable[j]
            if len(x) < 2 or len(y) < 2:
                p = math.nan
            elif np.all(x == x[0]) and np.all(y == x[0]):
                p = 1.0   # all values equal: no difference (scipy would return nan)
            else:
                p = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
            pair = {"a": labels[i], "b": labels[j], "p": p, "a12": a12(x, y, higher)}
            if kind != "sr":
                si, sj = out.summary[i]["successes"], out.summary[j]["successes"]
                ni, nj = len(groups[i]), len(groups[j])
                pair["fisher_p"] = float(stats.fisher_exact([[si, ni - si], [sj, nj - sj]]).pvalue) \
                    if ni and nj else math.nan
                pair["rate_a"], pair["rate_b"] = (si / ni if ni else math.nan), (sj / nj if nj else math.nan)
            pairs.append(pair)

    if pairs:
        raw = [pp["p"] for pp in pairs] + ([pp["fisher_p"] for pp in pairs] if kind != "sr" else [])
        adjusted = holm([v if math.isfinite(v) else 1.0 for v in raw])
        for k, pair in enumerate(pairs):
            pair["p_holm"] = float(adjusted[k])
            if kind != "sr":
                pair["fisher_p_holm"] = float(adjusted[len(pairs) + k])
            pair["significant"], pair["winner"], pair["verdict"] = _verdict(pair, kind)
    out.pairs = pairs

    if len(groups) >= 3 and all(len(v) >= 2 for v in usable):
        flat = np.concatenate(usable)
        if np.all(flat == flat[0]):
            out.omnibus = {"H": 0.0, "p": 1.0}
        else:
            h, p = stats.kruskal(*usable)
            out.omnibus = {"H": float(h), "p": float(p)}
    return out


ALPHA = 0.05


def _verdict(pair, kind):
    """(significant, winner or None, text) of one pairwise comparison at ALPHA (Holm-adjusted)."""
    winners = []
    if pair["p_holm"] < ALPHA and pair["a12"] != 0.5:
        winners.append(pair["a"] if pair["a12"] > 0.5 else pair["b"])
    if kind != "sr" and pair["fisher_p_holm"] < ALPHA and pair["rate_a"] != pair["rate_b"]:
        winners.append(pair["a"] if pair["rate_a"] > pair["rate_b"] else pair["b"])
    if not winners:
        return False, None, "no significant difference"
    if len(set(winners)) > 1:
        return True, None, "tests disagree"
    return True, winners[0], f"{winners[0]} better"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else f"{float(v):.10g}"
    return v


def to_csv_raw(sets, labels):
    """Everything recorded: one row per problem, configuration, run and generation.
    sets: list of (problem name, kind, groups)."""
    import csv
    blocks = sorted({b for _, _, groups in sets for b in block_names(groups)})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["problem", "configuration", "run", "seed", "generation"] + SERIES_COLUMNS
               + [f"block_{b}" for b in blocks]
               + ["best_size", "best_depth", "best_expression", "run_success", "run_seconds", "run_state",
                  "run_error"])
    for problem, _, groups in sets:
        for label, g in zip(labels, groups):
            for k, r in enumerate(g, 1):
                s = run_series(r)
                best = r.get("gen_best") or []
                success = "" if r["success"] is None else int(bool(r["success"]))
                for gi in range(len(r["train"])):
                    def at(column):
                        values = s.get(column)
                        if values is None and column.startswith("block_"):
                            return 0.0
                        return values[gi] if values is not None and gi < len(values) else math.nan
                    expr, size, depth = best[gi] if gi < len(best) else ("", "", "")
                    w.writerow([problem, label, k, r["seed"], gi + 1] + [_cell(at(c)) for c in SERIES_COLUMNS]
                               + [_cell(at(f"block_{b}")) for b in blocks]
                               + [size, depth, expr, success, f"{r['seconds']:.3f}", r["state"], r["error"]])
    return buf.getvalue().encode("utf-8")


def to_csv_aggregated(sets, labels):
    """Per problem, configuration, metric and generation: statistics over the runs (long format).
    sets: list of (problem name, kind, groups)."""
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["problem", "configuration", "metric", "generation", "runs", "median", "q1", "q3", "mean", "sd",
                "min", "max", "ci_low", "ci_high"])
    for problem, kind, groups in sets:
        length = generations_of(groups)
        columns = SERIES_COLUMNS + [f"block_{b}" for b in block_names(groups)]
        for label, g in zip(labels, groups):
            for column in columns:
                m = series_matrix(g, column, length)
                for gi in range(length):
                    v = m[:, gi] if len(m) else np.empty(0)
                    v = v[np.isfinite(v)]
                    if not len(v):
                        w.writerow([problem, label, column, gi + 1, 0] + [""] * 9)
                        continue
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    w.writerow([problem, label, column, gi + 1, len(v), _cell(med), _cell(q1), _cell(q3),
                                _cell(v.mean()), _cell(v.std(ddof=1)) if len(v) > 1 else "", _cell(v.min()),
                                _cell(v.max()), "", ""])
            if kind != "sr" and g:
                # share of runs that have found a perfect program by this generation (95 % Wilson CI)
                m = series_matrix(g, success_column(kind), length)
                for gi in range(length):
                    col = m[:, gi]
                    n = int(np.isfinite(col).sum())
                    k = int(np.sum(col >= 100.0 - 1e-9))
                    lo, hi = wilson(k, n)
                    rate = k / n if n else math.nan
                    w.writerow([problem, label, "success_rate", gi + 1, n, "", "", "", _cell(100 * rate), "", "",
                                "", _cell(100 * lo), _cell(100 * hi)])
    return buf.getvalue().encode("utf-8")
