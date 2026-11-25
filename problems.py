"""Benchmark problems with one input (x) or two inputs (x, y).

Target functions are plain Python expressions. User-defined expressions are
validated with `parse_expression` (an AST whitelist), so only arithmetic,
a fixed set of math functions and the variables x / y are allowed.
"""

import ast
import json
import math
import os
from dataclasses import dataclass

import numpy as np

import datasets
import synthesis

DEFAULT_DOMAIN = (-1.0, 1.0)

# Custom problems are stored next to the app so they survive restarts.
CUSTOM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_problems.json")

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
VARIABLES = ("x", "y")


@dataclass(frozen=True)
class Problem:
    expression: str
    inputs: int                 # 1 or 2
    name: str = ""              # e.g. "Nguyen-10"
    limits: tuple = None        # 1D only: (y_ticks, (y_min, y_max), legend_position)
    custom: bool = False
    domain: tuple = DEFAULT_DOMAIN  # training inputs are drawn uniformly from this interval
    extrapolate: bool = True        # validate slightly outside the training domain
    kind: str = "sr"                # "sr" symbolic regression, "ps" program synthesis

    @property
    def validation_domain(self):
        """5% wider than the training domain on each side (a little extrapolation),
        unless the target is not defined there (e.g. sqrt(x) on [0, 1])."""
        lo, hi = self.domain
        pad = 0.05 * (hi - lo) if self.extrapolate else 0.0
        return lo - pad, hi + pad

    @property
    def label(self):
        return self.expression if not self.name else f"{self.expression}  ({self.name})"


BUILTIN = [
    # Names as in McDermott et al., "Genetic Programming Needs Better Benchmarks" (GECCO 2012).
    Problem("x**4 + x**3 + x**2 + x", 1, name="Koza-1", limits=(np.linspace(0, 4, 5), (-0.6, 4.2), "upper left")),
    Problem("x**5 - 2*x**3 + x", 1, name="Koza-2", limits=(np.linspace(-0.3, 0.3, 7), (-0.35, 0.35), "lower right")),
    Problem("sin(x**2) * cos(x) - 1", 1, name="Nguyen-5",
            limits=(np.linspace(-1.0, -0.6, 5), (-1.02, -0.5), "upper center")),
    Problem("sin(x) + sin(x + x**2)", 1, name="Nguyen-6",
            limits=(np.linspace(-1.0, 1.7, 5), (-1.02, 1.95), "upper left")),
    Problem("sqrt(abs(x))", 1, name="Step-Wise", limits=(np.linspace(0, 1.0, 6), (-0.01, 1.1), "upper center")),
    # Two-input benchmarks (Uy et al. 2011, "Nguyen" suite), both defined on U[-1, 1]^2
    Problem("2*sin(x)*cos(y)", 2, name="Nguyen-10"),
    Problem("x**4 - x**3 + y**2/2 - y", 2, name="Nguyen-12"),
    # Pagie & Hogeweg (1997), defined on [-5, 5]^2
    Problem("1/(1 + x**-4) + 1/(1 + y**-4)", 2, name="Pagie-1", domain=(-5.0, 5.0)),
]

# Program synthesis problems in PSB1 order (Helmuth & Spector 2015), then Boolean problems
# from small to large.
_by_name = {p.expression: p for p in synthesis.PSB1 + synthesis.LOGIC}
BUILTIN += [_by_name[n] for n in ("Small or Large", "Compare String Lengths", "Count Odds",
                                  "Even-3-Parity", "Even-4-Parity", "Even-5-Parity", "Majority-5",
                                  "Multiplexer-6")]
KIND_ORDER = ("sr", "ps", "logic")
KIND_TITLES = {"sr": "Symbolic regression", "ps": "Program synthesis", "logic": "Boolean logic"}


def ordered():
    """Problems for the dropdown: by kind, built-in problems first (in the order above),
    then custom problems of that kind in the order they were added."""
    problems_ = list(REGISTRY.values())
    return {kind: [p for p in problems_ if p.kind == kind and not p.custom]
            + [p for p in problems_ if p.kind == kind and p.custom] for kind in KIND_ORDER}

# All selectable problems by expression / name; custom problems are appended at runtime.
REGISTRY = {p.expression: p for p in BUILTIN}


def get(expression):
    return REGISTRY[expression]


def title(problem):
    """Short unique name for headings, file names and exports: Koza-1 etc. for the regression
    benchmarks, otherwise the problem itself (a data set: the name it was imported under;
    the `name` of other problems is their source, e.g. PSB1 or "dataset")."""
    if getattr(problem, "dataset", False):
        return problem.expression
    return problem.name if problem.kind == "sr" and problem.name else problem.expression


# --------------------------------------------------------------------------
# Expression handling
# --------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


class ExpressionError(ValueError):
    pass


def parse_expression(text, domain=DEFAULT_DOMAIN):
    """Validate a user expression. Returns (normalised expression, number of inputs)."""
    text = (text or "").strip().replace("^", "**")
    if not text:
        raise ExpressionError("Enter a function of x (and optionally y).")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ExpressionError("Syntax error. Use Python syntax, e.g. x**2 + sin(y).") from None

    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"Not allowed: {type(node).__name__}. Only + - * / ** and functions.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ExpressionError("Only numeric constants are allowed.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ExpressionError("Allowed functions: " + ", ".join(FUNCTIONS) + ".")
            if len(node.args) != 1 or node.keywords:
                raise ExpressionError(f"{node.func.id}() takes exactly one argument.")
        if isinstance(node, ast.Name):
            if node.id in VARIABLES:
                used.add(node.id)
            elif node.id not in CONSTANTS and node.id not in FUNCTIONS:
                raise ExpressionError(f"Unknown name '{node.id}'. Use x, y, pi, e.")
    if "x" not in used:
        raise ExpressionError("The function must use x" + (" (y alone is not enough)." if used else "."))
    inputs = 2 if "y" in used else 1

    lo, hi = float(domain[0]), float(domain[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ExpressionError("The input range needs two numbers with from < to.")
    if not _finite_on(text, inputs, (lo, hi), extrapolate=False):
        raise ExpressionError(f"The function must be finite on [{lo:g}, {hi:g}]" + ("²." if inputs == 2 else "."))
    return text, inputs


def _finite_on(expression, inputs, domain, extrapolate):
    probe = Problem(expression, inputs, domain=tuple(domain), extrapolate=extrapolate)
    try:
        values = evaluate(expression, prediction_points(probe, plot_margin=False))
    except Exception:  # e.g. overflow inside numpy
        return False
    return bool(np.all(np.isfinite(values)))


def make_problem(expression, inputs, domain, custom=True, name=""):
    domain = (float(domain[0]), float(domain[1]))
    return Problem(expression, inputs, name=name, custom=custom, domain=domain,
                   extrapolate=_finite_on(expression, inputs, domain, extrapolate=True))


def evaluate(expression, X):
    """Evaluate a target expression on inputs X with shape (n, inputs)."""
    X = np.asarray(X, dtype=float)
    names = {"__builtins__": {}}
    local = {**FUNCTIONS, **CONSTANTS, "x": X[:, 0]}
    if X.shape[1] > 1:
        local["y"] = X[:, 1]
    with np.errstate(all="ignore"):
        values = eval(expression, names, local)
    return np.broadcast_to(np.asarray(values, dtype=float), (len(X),)).copy()


# --------------------------------------------------------------------------
# Custom problems (persisted as JSON)
# --------------------------------------------------------------------------

def add_custom(expression, inputs, domain=DEFAULT_DOMAIN, name=""):
    if expression not in REGISTRY or REGISTRY[expression].custom:
        REGISTRY[expression] = make_problem(expression, inputs, domain, name=name)
        save_custom()
    return REGISTRY[expression]


def remove_custom(expression):
    problem = REGISTRY.get(expression)
    if problem is not None and problem.custom:
        del REGISTRY[expression]
        save_custom()
    return problem


def restore_custom(problem):
    REGISTRY[problem.expression] = problem
    save_custom()


def add_custom_program(problem):
    """Register a user-defined program synthesis problem (from synthesis.parse_program)."""
    REGISTRY[problem.expression] = problem
    save_custom()
    return problem


def save_custom():
    data = []
    for p in REGISTRY.values():
        if not p.custom:
            continue
        if getattr(p, "dataset", False):
            data.append({"kind": "dataset", "spec": p.spec.to_dict()})
        elif p.kind == "ps":
            data.append({"kind": "ps", "code": p.code, "inputs": p.spec.to_dict()})
        elif p.kind == "logic":
            data.append({"kind": "logic", "code": p.code, "name": getattr(p, "display_name", "")})
        else:
            data.append({"expression": p.expression, "domain": list(p.domain), "name": p.name})
    try:
        with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # read-only install: custom problems then only live for this session


def load_custom():
    try:
        with open(CUSTOM_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    for entry in data if isinstance(data, list) else []:
        if isinstance(entry, dict) and entry.get("kind") == "dataset":
            try:
                problem = datasets.build(datasets.DatasetSpec.from_dict(entry["spec"]))
            except (KeyError, TypeError, ValueError):
                continue
            REGISTRY.setdefault(problem.expression, problem)
            continue
        if isinstance(entry, dict) and entry.get("kind") == "logic":
            try:
                problem = synthesis.parse_logic(entry["code"], entry.get("name", ""))
            except (KeyError, TypeError, ValueError):
                continue
            REGISTRY.setdefault(problem.expression, problem)
            continue
        if isinstance(entry, dict) and entry.get("kind") == "ps":
            try:
                spec = synthesis.InputSpec.from_dict(entry.get("inputs"))
                if "int_range" in entry:  # files written by an earlier version
                    spec.int_range = tuple(entry["int_range"])
                problem = synthesis.parse_program(entry["code"], spec)
            except (KeyError, TypeError, ValueError):
                continue
            REGISTRY.setdefault(problem.expression, problem)
            continue
        try:
            domain = tuple(float(v) for v in entry.get("domain", DEFAULT_DOMAIN))
            expression, inputs = parse_expression(entry["expression"], domain)
        except (KeyError, TypeError, ValueError):
            continue  # skip broken entries instead of failing on start
        REGISTRY.setdefault(expression, make_problem(expression, inputs, domain, name=entry.get("name", "")))


# --------------------------------------------------------------------------
# Data sets
# --------------------------------------------------------------------------

def make_training_set(problem, size, noise, outlier_share=0.0):
    """Uniform inputs, uniform noise in [-noise, noise]. Optionally a share of the
    points becomes outliers: zero-mean Gaussian noise with four times the standard
    deviation of the clean targets (as in the ROIDS experiments, Geiger et al. 2026)."""
    X = np.random.uniform(*problem.domain, (size, problem.inputs))
    if problem.inputs == 1:
        X = X[np.argsort(X[:, 0])]
    clean = evaluate(problem.expression, X)
    y = clean + np.random.uniform(-noise, noise, size)
    n_out = int(round(outlier_share * size))
    if n_out:
        idx = np.random.choice(size, size=n_out, replace=False)
        y[idx] += np.random.normal(0.0, 4.0 * max(float(np.std(clean)), 1e-3), n_out)
    return [tuple(row) for row in X.tolist()], y.tolist()


def make_validation_set(problem, size):
    if problem.inputs == 1:
        return [(v,) for v in np.linspace(*problem.validation_domain, size).tolist()]
    side = max(2, int(round(math.sqrt(size))))
    g = np.linspace(*problem.validation_domain, side)
    return [tuple(row) for row in np.array(np.meshgrid(g, g)).reshape(2, -1).T.tolist()]


def make_data(problem, train_size, test_size, noise=0.0, outlier_share=0.0):
    """Training and validation/test data for either kind of problem:
    (train_X, train_y, test_X, test_y) with inputs as tuples."""
    if getattr(problem, "dataset", False) and problem.kind == "sr":
        (train_X, train_y), (test_X, test_y) = problem.train, problem.test
        return list(train_X), list(train_y), list(test_X), list(test_y)
    if problem.kind != "sr":
        train, test = problem.generate(train_size, test_size)
        return ([i for i, _ in train], [o for _, o in train], [i for i, _ in test], [o for _, o in test])
    train_X, train_y = make_training_set(problem, train_size, noise, outlier_share)
    val_X = make_validation_set(problem, test_size)
    return train_X, train_y, val_X, evaluate(problem.expression, val_X).tolist()


# Grids on which the best individual is drawn.
CURVE_POINTS = 241
SURFACE_POINTS = 27


def prediction_axis(problem, plot_margin=True):
    lo, hi = problem.validation_domain
    if problem.inputs == 1:
        pad = 0.05 * (hi - lo) if plot_margin and problem.extrapolate else 0.0
        return np.linspace(lo - pad, hi + pad, CURVE_POINTS)
    return np.linspace(lo, hi, SURFACE_POINTS)


def surface_grid(problem):
    axis = prediction_axis(problem)
    return np.meshgrid(axis, axis)


def prediction_points(problem, plot_margin=True):
    if problem.inputs == 1:
        return prediction_axis(problem, plot_margin)[:, None]
    gx, gy = surface_grid(problem)
    return np.column_stack([gx.ravel(), gy.ravel()])


def value_range(problem, pad=0.08):
    """Axis range of the target on the validation domain (for plot adjustment)."""
    pts = prediction_points(problem)
    if problem.inputs == 1:
        lo, hi = problem.validation_domain
        pts = pts[(pts[:, 0] >= lo) & (pts[:, 0] <= hi)]
    v = evaluate(problem.expression, pts)
    v = v[np.isfinite(v)]
    lo, hi = (float(np.min(v)), float(np.max(v))) if v.size else (-1.0, 1.0)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span


load_custom()
