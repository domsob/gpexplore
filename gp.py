"""Genetic programming engine for symbolic regression (DEAP based).

A run lives in its own background thread and can be paused, resumed,
advanced generation by generation and stopped. The UI never touches the
DEAP objects directly: it polls `GPRun.snapshot()` which returns a plain,
consistent copy of the run's history.
"""

import itertools
import math
import operator
import random
import re
from collections import Counter
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from math import sqrt

import numpy as np
from deap import algorithms, base, creator, gp, tools

import config
import operators
import problems
import synthesis
from synthesis import Bool, Int, Num, Str, Vec


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def protectedDiv(left, right):
    if right != 0:
        return left / right
    return 1.0


def analyticQuotient(left, right):
    return left / sqrt(1 + right * right)


# Symbolic regression primitives on floats: (config flag, callable, arity, name).
PRIMITIVES = [
    ("add_func", operator.add, 2, "add"),
    ("sub_func", operator.sub, 2, "sub"),
    ("mul_func", operator.mul, 2, "mul"),
    ("protectedDiv_func", protectedDiv, 2, "protectedDiv"),
    ("analyticQuotient_func", analyticQuotient, 2, "analyticQuotient"),
    ("neg_func", operator.neg, 1, "neg"),
    ("cos_func", math.cos, 1, "cos"),
    ("sin_func", math.sin, 1, "sin"),
]
for _flag, _fn, _arity, _name in PRIMITIVES:
    synthesis.register_numeric(_name, _fn)

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def _eager(block):
    """Plain function for gp.compile (symbolic regression has no loops)."""
    if block.kind == "strict":
        return block.fn
    if block.kind == "if":
        default = synthesis.DEFAULTS[block.ret]
        return lambda c, a: a if c else default
    if block.kind == "if_else":
        return lambda c, a, b: a if c else b
    return None


def problem_signature(problem):
    """(argument names, argument types, return type) of the evolved program."""
    if problem.kind != "sr":   # program synthesis and Boolean logic
        return problem.arg_names, problem.arg_types, problem.ret
    names = problem.arg_names if getattr(problem, "dataset", False) else problems.VARIABLES[:problem.inputs]
    return names, (Num,) * problem.inputs, Num


def get_pset(settings):
    """Strongly typed primitive set for symbolic regression and program synthesis.

    Building blocks whose argument or return types cannot occur in the problem are
    left out (e.g. vector loops for a problem without vectors), and every type in
    use gets terminals, so trees can always be completed."""
    problem = settings["problem"]
    names, arg_types, ret = problem_signature(problem)
    pset = gp.PrimitiveSetTyped("MAIN", list(arg_types), ret)
    stamp = time.time_ns()   # DEAP needs unique names for ephemeral constants

    blocks = synthesis.enabled_blocks(settings, problem.kind) + operators.function_blocks(settings, problem.kind)
    available = set(arg_types) | {ret}
    if problem.kind == "ps":
        available.add(Int)                       # integer constants are always there
        if problem.str_constants:
            available.add(Str)
    if any(Bool in b.args or b.ret is Bool for b in blocks):
        available.add(Bool)                      # true / false constants
    chosen = [b for b in blocks if b.ret in available and all(t in available for t in b.args)]
    if problem.kind == "sr":
        chosen = [b for b in chosen if _eager(b) is not None]

    if Num in available:
        for flag, fn, arity, name in PRIMITIVES:
            if settings[flag]:
                pset.addPrimitive(fn, [Num] * arity, Num, name=name)
        for name, arity, fn in operators.numeric_functions(settings):   # custom functions
            pset.addPrimitive(fn, [Num] * arity, Num, name=name)
            synthesis.register_numeric(name, fn)
        if settings["int101_erc"]:
            lo, hi = sorted((int(settings.get("erc_lo", -1)), int(settings.get("erc_hi", 1))))
            pset.addEphemeralConstant(f"erc_{stamp}", partial(random.randint, lo, hi), Num)
    for b in chosen:
        pset.addPrimitive(_eager(b) or (lambda *a: None), list(b.args), b.ret, name=b.name)
    if Int in available and problem.kind == "ps":
        lo, hi = problem.erc_range
        pset.addEphemeralConstant(f"erc_int_{stamp}", partial(random.randint, lo, hi), Int)
        for c in problem.int_constants:
            pset.addTerminal(c, Int)
    if Num in available and problem.kind == "ps":
        for c in getattr(problem, "float_constants", ()):
            pset.addTerminal(c, Num)
    if Bool in available and problem.kind != "logic":   # Boolean problems use only their inputs
        pset.addTerminal(True, Bool)
        pset.addTerminal(False, Bool)
    if Str in available:
        for c in ("",) + tuple(problem.str_constants):
            pset.addTerminal(c, Str)
    if synthesis.uses_loops(chosen):
        pset.addTerminal(synthesis.ELEM, Int, name=synthesis.ELEM)
    pset.renameArguments(**{f"ARG{i}": n for i, n in enumerate(names)})
    return pset


SWITCH_FLAGS = ([flag for flag, *_ in PRIMITIVES] + ["int101_erc"] + synthesis.GROUP_FLAGS
                + synthesis.GATE_FLAGS)


def applicable_flags(settings):
    """Switches of the function tab that can have an effect on the selected problem
    (whether they are on or not): builds the primitive set with every switch on."""
    trial = dict(settings)
    for flag in SWITCH_FLAGS:
        trial[flag] = True
    pset = get_pset(trial)
    names = {p.name for prims in pset.primitives.values() for p in prims}
    result = {flag for flag, _, _, name in PRIMITIVES if name in names}
    if Num in pset.terminals:
        result.add("int101_erc")
    for flag, _, _, blocks in synthesis.GROUPS:
        if any(b.name in names for b in blocks):
            result.add(flag)
    for flag, gate in synthesis.LOGIC_GATES:
        if gate.name in names:
            result.add(flag)
    return result


def _generate(pset, min_, max_, condition, type_=None):
    """gp.generate, but falls back to a terminal when a type has no primitives
    (and to a primitive when it has no terminals) instead of raising."""
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, t = stack.pop()
        prims, terms = pset.primitives.get(t, []), pset.terminals.get(t, [])
        if terms and (condition(height, depth) or not prims):
            term = random.choice(terms)
            if type(term) is gp.MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            prim = random.choice(prims)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def gen_full(pset, min_, max_, type_=None):
    return _generate(pset, min_, max_, lambda height, depth: depth == height, type_)


def gen_grow(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)
    return _generate(pset, min_, max_, condition, type_)


def gen_half_and_half(pset, min_, max_, type_=None):
    return random.choice((gen_grow, gen_full))(pset, min_, max_, type_)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

# (key, label) of all selection methods, in UI order.
CROSSOVERS = ["One-point", "Leaf-biased (90/10)"]
MUTATIONS = ["Uniform (subtree)", "Node replacement", "Shrink", "Ephemeral constants", "Mixed"]
BLOAT_CONTROLS = ["Depth limit", "Depth + size limit", "Lexicographic parsimony"]
FITNESS_FUNCTIONS = ["MSE", "MAE", "RMSE", "MSE + linear scaling"]
FITNESS_TITLES = {"MSE": "Mean squared error", "MAE": "Mean absolute error", "RMSE": "Root mean squared error",
                  "MSE + linear scaling": "Mean squared error (linearly scaled)"}

# Order: the standard method, the lexicase family (plain and ε, then with random
# and informed down-sampling, then ROIDS), finally the baselines.
SELECTION_METHODS = [
    ("Tournament", "Tournament"),
    ("Lexicase", "Lexicase (Spector)"),
    ("Epsilon-Lexicase", "ε-Lexicase (La Cava et al.)"),
    ("DS Lexicase", "Down-sampled lexicase (Hernandez et al.)"),
    ("DS Epsilon-Lexicase", "Down-sampled ε-Lexicase (Geiger et al.)"),
    ("IDS Lexicase", "Informed down-sampled lexicase (Boldi et al.)"),
    ("IDS Epsilon-Lexicase", "Informed down-sampled ε-Lexicase (Geiger et al.)"),
    ("ROIDS", "ROIDS (Geiger et al.)"),
    ("Random", "Random"),
    ("Best", "Best"),
    ("Worst", "Worst"),
]
EPSILON_METHODS = {"Epsilon-Lexicase", "DS Epsilon-Lexicase", "IDS Epsilon-Lexicase"}
LEXICASE_METHODS = EPSILON_METHODS | {"Lexicase", "DS Lexicase", "IDS Lexicase"}
DOWNSAMPLED_METHODS = {"DS Epsilon-Lexicase", "IDS Epsilon-Lexicase", "DS Lexicase", "IDS Lexicase", "ROIDS"}
INFORMED_METHODS = {"IDS Epsilon-Lexicase", "IDS Lexicase", "ROIDS"}


def epsilon_lexicase(errors, k, cases, rng, epsilon=True):
    """ε-lexicase selection (La Cava et al. 2016) on the given training cases.

    errors: (individuals x cases) matrix of squared errors.
    For every selection event the cases are shuffled; on each case all
    candidates with e_t <= e*_t + ε_t survive, where e*_t is the best error
    among the remaining candidates and ε_t = median(|e_t - median(e_t)|),
    the median absolute deviation over the remaining candidates
    (as in Geiger et al. 2024, Eq. 3).
    """
    errors = errors[:, cases]
    n, m = errors.shape
    chosen = np.empty(k, dtype=int)
    everyone = np.arange(n)
    for i in range(k):
        candidates = everyone
        for case in rng.permutation(m):
            column = errors[candidates, case]
            if epsilon:
                eps = np.median(np.abs(column - np.median(column)))
                candidates = candidates[column <= column.min() + eps]
            else:  # standard lexicase (Spector 2012): only the elite on this case survive
                candidates = candidates[column == column.min()]
            if len(candidates) == 1:
                break
        chosen[i] = candidates[rng.integers(len(candidates))]
    return chosen


def farthest_first_traversal(distances, size, rng):
    """Down-sample construction of informed down-sampling (Boldi et al. 2024, Alg. 1).

    Starts with a random case and repeatedly adds the case whose minimum
    distance to the cases already chosen is largest (ties broken randomly).
    Once only zero distances are left, the remaining cases are random.
    """
    n = distances.shape[0]
    size = min(size, n)
    first = int(rng.integers(n))
    chosen = [first]
    min_dist = distances[first].astype(float)
    min_dist[first] = -np.inf
    while len(chosen) < size:
        best = np.flatnonzero(min_dist == min_dist.max())
        nxt = int(best[rng.integers(len(best))])
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, distances[nxt])
        min_dist[chosen] = -np.inf
    return np.array(sorted(chosen))


def case_distances(errors):
    """Euclidean distance between training cases, based on the errors of
    the sampled parents on each case (Geiger et al. 2024 use Euclidean instead
    of Hamming distance for the continuous errors of symbolic regression)."""
    cols = np.minimum(errors, 1e12).T          # cases x sampled parents
    sq = np.sum(cols ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * cols @ cols.T
    return np.sqrt(np.maximum(d2, 0.0))


def solve_distances(errors):
    """Hamming distance between the solve vectors of the cases (error == 0 counts
    as solved), as in informed down-sampled lexicase (Boldi et al. 2024)."""
    solved = (errors == 0).T.astype(float)        # cases x sampled parents
    return solved @ (1 - solved).T + (1 - solved) @ solved.T


def tournament_on_cases(errors, k, cases, tourn_size, rng):
    """Tournament selection on the MSE over the given cases (used by ROIDS)."""
    fitness = errors[:, cases].mean(axis=1)
    contestants = rng.integers(len(fitness), size=(k, tourn_size))
    return contestants[np.arange(k), np.argmin(fitness[contestants], axis=1)]


class Selector:
    """Selects parents; keeps the state informed down-sampling needs."""

    def __init__(self, settings):
        self.method = settings["selection_method"]
        self.tourn_size = settings["tourn_size"]
        self.ds_rate = settings["ds_rate"]
        self.parent_rate = settings["parent_sampling_rate"]
        self.interval = settings["distance_interval"]
        self.outlier_rate = settings["outlier_rate"]
        self.parsimony = settings.get("bloat_control") == "Lexicographic parsimony"
        seed = settings.get("seed") or None
        self.rng = np.random.default_rng(seed)
        self.last_selected = []
        self.distances = None
        self.kept = None              # ROIDS: case indices that are not excluded as outliers
        self.outliers = None          # ROIDS: excluded case indices (for the plot)
        self.last_downsample = None
        self.selections = 0
        self.custom = operators.from_settings(settings, "selection")
        self.py_rng = random.Random(random.getrandbits(64))   # for custom selection (seeded through `random`)

    def __call__(self, population, k):
        chosen = self._select(population, k)
        self.last_selected = chosen
        return chosen

    def _tournament(self, population, k):
        """Tournament; with lexicographic parsimony pressure (Luke & Panait 2002)
        ties in fitness are won by the smaller tree."""
        if not self.parsimony:
            return tools.selTournament(population, k, tournsize=self.tourn_size)
        picks = self.rng.integers(len(population), size=(k, self.tourn_size))
        return [min((population[i] for i in row), key=lambda ind: (ind.fitness.values[0], len(ind)))
                for row in picks]

    def _custom(self, population, k):
        name, fn = self.custom
        fitness = [float(ind.fitness.values[0]) for ind in population]
        errors = [[float(e) for e in ind.errors] for ind in population]
        sizes = [len(ind) for ind in population]
        chosen = operators.call("selection", name, fn, fitness, errors, sizes, k, self.py_rng)
        try:
            chosen = [int(i) for i in chosen]
        except (TypeError, ValueError):
            operators.fail("selection", name, "must return a list of indices.")
        if len(chosen) != k or any(not 0 <= i < len(population) for i in chosen):
            operators.fail("selection", name, f"must return {k} indices between 0 and {len(population) - 1} "
                                              f"(got {len(chosen)}).")
        self.last_downsample = None
        return [population[i] for i in chosen]

    def _select(self, population, k):
        if self.custom is not None:
            return self._custom(population, k)
        if self.method == "ROIDS":
            return self._roids(population, k)
        if self.method not in LEXICASE_METHODS:
            simple = {
                "Tournament": self._tournament,
                "Random": tools.selRandom,
                "Best": tools.selBest,
                "Worst": tools.selWorst,
            }[self.method]
            return simple(population, k)

        errors = np.array([ind.errors for ind in population])
        n_cases = errors.shape[1]
        ds_size = max(1, int(round(self.ds_rate * n_cases)))

        epsilon = self.method in EPSILON_METHODS
        if self.method in ("Epsilon-Lexicase", "Lexicase"):
            cases = np.arange(n_cases)
            self.last_downsample = None
        elif self.method in ("DS Epsilon-Lexicase", "DS Lexicase"):
            cases = np.sort(self.rng.choice(n_cases, size=ds_size, replace=False))
            self.last_downsample = cases
        else:  # informed down-sampling
            if self.selections % self.interval == 0 or self.distances is None:
                n_parents = min(len(population), max(2, int(round(self.parent_rate * len(population)))))
                sample = self.rng.choice(len(population), size=n_parents, replace=False)
                # In the paper only these parents are evaluated on *all* cases.
                # GPExplore evaluates everyone on everything (for the plots), so
                # their full error vectors are already available.
                self.distances = (case_distances if epsilon else solve_distances)(errors[sample])
            cases = farthest_first_traversal(self.distances, ds_size, self.rng)
            self.last_downsample = cases

        self.selections += 1
        chosen = epsilon_lexicase(errors, k, cases, self.rng, epsilon=epsilon)
        return [population[i] for i in chosen]

    def _roids(self, population, k):
        """Robust outlier-aware informed down-sampling (Geiger et al. 2026, Alg. 1)
        followed by tournament selection on the down-sample.

        Every k generations: sample s·|P| parents, take each case's mean error over
        them, drop the γ·|T| cases with the highest mean error as outliers and
        compute case distances on the rest. Every generation: farthest-first
        traversal over the remaining cases builds a down-sample of size d·|T|.
        """
        errors = np.array([ind.errors for ind in population])
        n_cases = errors.shape[1]
        if self.selections % self.interval == 0 or self.distances is None:
            n_parents = min(len(population), max(2, int(round(self.parent_rate * len(population)))))
            sample = errors[self.rng.choice(len(population), size=n_parents, replace=False)]
            n_out = min(n_cases - 1, int(round(self.outlier_rate * n_cases)))
            order = np.argsort(sample.mean(axis=0))          # ascending mean error per case
            self.kept = np.sort(order[:n_cases - n_out])
            self.outliers = np.sort(order[n_cases - n_out:])
            self.distances = case_distances(sample[:, self.kept])
        ds_size = max(1, int(round(self.ds_rate * n_cases)))
        cases = self.kept[farthest_first_traversal(self.distances, ds_size, self.rng)]
        self.last_downsample = cases
        self.selections += 1
        chosen = tournament_on_cases(errors, k, cases, self.tourn_size, self.rng)
        if self.parsimony:   # ties in down-sample fitness: smaller tree wins
            fit = errors[:, cases].mean(axis=1)
            picks = self.rng.integers(len(population), size=(k, self.tourn_size))
            chosen = [min(row, key=lambda i: (fit[i], len(population[i]))) for row in picks]
        return [population[i] for i in chosen]


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

_MATH_ERRORS = (OverflowError, ValueError, ZeroDivisionError, TypeError)


def _squared_errors(func, X, y):
    out = np.empty(len(y))
    for i, (row, target) in enumerate(zip(X, y)):
        try:
            out[i] = (func(*row) - target) ** 2
        except _MATH_ERRORS:
            out[i] = math.inf
    out[~np.isfinite(out)] = math.inf
    return out


def _outputs_of(func, X):
    out = np.empty(len(X))
    for i, row in enumerate(X):
        try:
            out[i] = func(*row)
        except _MATH_ERRORS:
            out[i] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _linear_scaling(outputs, y):
    """Keijzer (2003): best a, b for y ≈ a + b·f(x) on the training data."""
    ok = np.isfinite(outputs)
    if ok.sum() < 2:
        return 0.0, 1.0
    o, t = outputs[ok], np.asarray(y, dtype=float)[ok]
    var = float(np.var(o))
    b = float(np.cov(o, t, bias=True)[0, 1] / var) if var > 1e-12 else 0.0
    a = float(t.mean() - b * o.mean())
    return a, b


def _case_errors(outputs, y, metric, scale=(0.0, 1.0)):
    """Per-case errors used by lexicase: absolute for MAE, squared otherwise."""
    a, b = scale
    diff = (a + b * outputs) - np.asarray(y, dtype=float)
    err = np.abs(diff) if metric == "MAE" else diff ** 2
    err[~np.isfinite(err)] = math.inf
    return err


def _aggregate(errors, metric):
    value = float(errors.mean())
    if metric == "RMSE":
        value = math.sqrt(value) if math.isfinite(value) else math.inf
    return value if math.isfinite(value) else math.inf


def _predict(func, problem, data=None, scale=(0.0, 1.0)):
    a, b = scale
    if getattr(problem, "dataset", False):
        # data sets: predictions on the training and on the validation rows
        def run(rows):
            out = np.empty(len(rows))
            for i, row in enumerate(rows):
                try:
                    out[i] = func(*row)
                except _MATH_ERRORS:
                    out[i] = np.nan
            out[~np.isfinite(out)] = np.nan
            return a + b * out
        return run(data[0]), run(data[1])
    pts = problems.prediction_points(problem)
    out = np.empty(len(pts))
    for i, row in enumerate(pts):
        try:
            out[i] = func(*map(float, row))
        except _MATH_ERRORS:
            out[i] = np.nan
    out[~np.isfinite(out)] = np.nan
    out = a + b * out
    return out if problem.inputs == 1 else out.reshape(problems.surface_grid(problem)[0].shape)


def _tree_nodes(individual, variables):
    nodes = []
    for node in individual:
        if node.arity > 0:
            block = synthesis.BLOCKS.get(node.name)
            nodes.append((block.symbol if block else node.name, node.arity, "function"))
        else:
            label = node.format()
            kind = "variable" if label in variables else "constant"
            nodes.append((label, 0, kind))
    return nodes


_TOKEN = re.compile(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[(),]|[^\s(),]+""")


def tree_nodes_from_string(expression, variables):
    """Prefix node list (as for the tree view) from an individual's string form,
    e.g. for the population browser (only strings are stored per generation)."""
    tokens = [t for t in _TOKEN.findall(expression) if t.strip()]
    nodes, pos = [], 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1
            index = len(nodes)
            nodes.append(None)
            arity = 0
            while tokens[pos] != ")":
                parse()
                arity += 1
                if tokens[pos] == ",":
                    pos += 1
            pos += 1
            block = synthesis.BLOCKS.get(tok)
            nodes[index] = (block.symbol if block else tok, arity, "function")
        else:
            nodes.append((tok, 0, "variable" if tok in variables else "constant"))

    parse()
    return nodes


def _finite_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.inf


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

@dataclass
class Snapshot:
    state: str                     # "running" | "paused" | "finished" | "stopped" | "error"
    generations: int               # planned number of generations
    avg_training: list = field(default_factory=list)
    avg_validation: list = field(default_factory=list)
    best_training: list = field(default_factory=list)
    best_validation: list = field(default_factory=list)
    best_expression: str = ""
    best_size: int = 0
    best_depth: int = 0
    best_prediction: np.ndarray = None
    best_nodes: list = None        # prefix list of (label, arity, kind) for the tree view
    downsample: np.ndarray = None  # training cases used for the latest selection (DS / IDS / ROIDS)
    outliers: np.ndarray = None    # ROIDS: training cases currently excluded as outliers
    best_solved: np.ndarray = None # program synthesis: training cases solved by the best individual
    best_test_solved: int = 0      # program synthesis: test cases solved by the best individual
    case_share: np.ndarray = None  # program synthesis: share of the population solving each training case
    best_outputs: list = None      # program synthesis: outputs of the best program on the training cases
    best_errors: np.ndarray = None # program synthesis: errors of the best program on the training cases
    best_scale: tuple = None       # linear scaling (a, b) of the best individual, if used
    dynamics: list = None          # per generation: population statistics (dynamics views)
    best_uid: int = None           # id of the best individual in the genealogy (ancestry view)
    population: list = None        # the population of the shown generation (population browser)
    variables: tuple = ()          # input names (for trees built from strings)
    error: str = ""

    @property
    def generation(self):
        return len(self.best_training)


class GPRun:
    """One evolutionary run in a background thread.

    Control flow between generations:
      * running  -> waits `config.inter_gen_pause` seconds (read live)
      * paused   -> waits until resumed, stepped or stopped
      * step()   -> computes exactly one more generation, then pauses again
    """

    def __init__(self, settings, start_paused=False, headless=False):
        self.settings = dict(settings)
        self.headless = headless   # experiments: no display, no pauses
        self._cond = threading.Condition()
        self._paused = start_paused
        self._steps = 0
        self._displayed = -1   # last generation the UI has shown (see ack)
        self._stopped = False
        self._state = "paused" if start_paused else "running"
        self._error = ""
        self._history = {
            "avg_training": [],
            "avg_validation": [],
            "best_training": [],
            "best_validation": [],
        }
        # One record per generation (best individual, plot data), so the UI can
        # show any earlier generation again (generation slider).
        self._records = []
        # Genealogy (GUI only): id -> (generation, operator, parent ids, fitness, size, expression)
        self._genealogy = {}
        self._thread = threading.Thread(target=self._run, name="gp-run", daemon=True)

    # ---- control (called from the UI thread) -----------------------------

    def start(self):
        self._thread.start()

    def pause(self):
        with self._cond:
            if self._state == "running":
                self._paused = True
                self._state = "paused"
            self._cond.notify_all()

    def resume(self):
        with self._cond:
            if self._state == "paused":
                self._paused = False
                self._state = "running"
            self._cond.notify_all()

    def step(self):
        with self._cond:
            if self._state == "paused":
                self._steps += 1
            self._cond.notify_all()

    def ack(self, generation):
        """Called by the UI once it has displayed `generation` (1-based count)."""
        with self._cond:
            if generation > self._displayed:
                self._displayed = generation
                self._cond.notify_all()

    def stop(self):
        with self._cond:
            self._stopped = True
            if self._state in ("running", "paused"):
                self._state = "stopped"
            self._cond.notify_all()

    def snapshot(self, generation=None):
        """State of the run, either now or as it was after `generation` (1-based)."""
        with self._cond:
            n = len(self._records)
            g = n if generation is None else max(0, min(int(generation), n))
            rec = self._records[g - 1] if g else {}
            return Snapshot(
                state=self._state,
                generations=self.settings["generations"],
                avg_training=list(self._history["avg_training"][:g]),
                avg_validation=list(self._history["avg_validation"][:g]),
                best_training=list(self._history["best_training"][:g]),
                best_validation=list(self._history["best_validation"][:g]),
                best_expression=rec.get("expression", ""),
                best_size=rec.get("size", 0),
                best_depth=rec.get("depth", 0),
                best_prediction=rec.get("prediction"),
                best_nodes=rec.get("nodes"),
                downsample=rec.get("downsample"),
                outliers=rec.get("outliers"),
                best_solved=rec.get("solved"),
                best_test_solved=rec.get("test_solved", 0),
                case_share=rec.get("share"),
                best_outputs=rec.get("outputs"),
                best_errors=rec.get("errors"),
                best_scale=rec.get("scale"),
                best_uid=rec.get("uid"),
                dynamics=[r["dynamics"] for r in self._records[:g]],
                population=rec.get("population"),
                variables=tuple(getattr(self, "_variables", ())),
                error=self._error,
            )

    def ancestry(self, uid, depth=8):
        """The ancestors of an individual over the last `depth` generations: {id: record}, where a
        record is (generation, operator, parent ids, fitness, size, expression). Operators: "init",
        "copy" (selected unchanged), "elite", "cx" (crossover), "mut" (mutation), "cx+mut"."""
        with self._cond:
            if uid not in self._genealogy:
                return {}
            oldest = self._genealogy[uid][0] - depth
            nodes, frontier = {}, [uid]
            while frontier:
                nxt = []
                for u in frontier:
                    if u in nodes or u not in self._genealogy:
                        continue
                    rec = self._genealogy[u]
                    nodes[u] = rec
                    if rec[0] > oldest:
                        nxt.extend(rec[2])
                frontier = nxt
            return nodes

    @property
    def computed(self):
        """Number of generations computed so far."""
        with self._cond:
            return len(self._records)

    # ---- worker thread ---------------------------------------------------

    ACK_TIMEOUT = 3.0   # never wait longer than this for the display

    def _wait_for_turn(self):
        """Block until the next generation may be computed. False = stop.

        Waits for the inter-generation pause *and* until the UI has displayed the
        current generation. Without the latter a pause of 0 starves the UI: the
        CPU-bound GP thread holds Python's GIL, rendering a frame then takes seconds
        and the animation appears frozen. Rendering now happens while GP waits."""
        if self.headless:
            return not self._stopped
        started = time.monotonic()
        with self._cond:
            while True:
                if self._stopped:
                    return False
                if self._steps > 0:
                    self._steps -= 1
                    return True
                if self._paused:
                    self._cond.wait()
                    continue
                now = time.monotonic()
                recorded = len(self._history["best_training"])
                shown = self._displayed >= recorded or now - started >= self.ACK_TIMEOUT
                remaining = started + config.inter_gen_pause - now
                if remaining <= 0 and shown:
                    return True
                deadline = started + (config.inter_gen_pause if shown else max(config.inter_gen_pause, self.ACK_TIMEOUT))
                self._cond.wait(max(0.001, deadline - now) if shown else
                                max(0.001, min(started + self.ACK_TIMEOUT, deadline) - now))

    def _dynamics(self, population, train_metric, extra):
        """Population statistics of one generation for the dynamics views."""
        q = lambda v: [float(x) for x in np.percentile(v, [10, 25, 50, 75, 90])] if len(v) else [math.nan] * 5
        fit = np.asarray(train_metric, dtype=float)
        sizes = np.array([len(ind) for ind in population])
        depths = np.array([ind.height for ind in population])
        genotypes = {str(ind) for ind in population}
        phenotypes = {np.round(np.asarray(ind.errors, dtype=float), 9).tobytes() for ind in population}
        blocks = Counter()
        for ind in population:
            for node in ind:
                if node.arity > 0:
                    blk = synthesis.BLOCKS.get(node.name)
                    blocks[blk.symbol if blk else node.name] += 1
                elif node.format() in self._variables:
                    blocks["inputs"] += 1
                else:
                    blocks["constants"] += 1
        total = sum(blocks.values()) or 1
        n = len(population)
        return {
            "fitness": q(fit[np.isfinite(fit)]),
            "size": q(sizes) + [int(sizes.max())], "depth": q(depths) + [int(depths.max())],
            "genotypic": 100.0 * len(genotypes) / n, "phenotypic": 100.0 * len(phenotypes) / n,
            "blocks": {k: v / total for k, v in blocks.items()},
            "parents": extra.get("parents"), "top_parent": extra.get("top_parent"),
            "variation": extra.get("variation"),
        }

    def _record(self, population, hof, validation_fitness, downsample, outliers=None, extra=None):
        problem = self.settings["problem"]
        cases = (None, None, None)
        if problem.kind != "sr":
            # Program synthesis: the curves show % of cases solved, and the best program
            # is the one solving the most training cases (ties: lower mean error).
            best = self._best_program
            solved_matrix = np.array([ind.errors for ind in population]) == 0
            share = solved_matrix.mean(axis=0)
            avg_training = 100.0 * float(share.mean())
            avg_validation = 100.0 * float(np.mean([self._test_sample_solved(ind) for ind in population]))
            test_errors = self._test_errors(best)
            best_solved = np.asarray(best.errors) == 0
            best_training = 100.0 * float(best_solved.mean())
            best_validation = 100.0 * float(np.mean(test_errors == 0))
            prediction = None
            solved = (best_solved, int(np.sum(test_errors == 0)))
            cases = (share, self._outputs(best), np.asarray(best.errors))
        else:
            best = hof[0]
            avg_training = _finite_mean([ind.fitness.values[0] for ind in population])
            avg_validation = _finite_mean([validation_fitness(ind) for ind in population])
            best_training = best.fitness.values[0]
            best_validation = validation_fitness(best)
            prediction = _predict(self._toolbox.compile(expr=best), problem,
                                  (self.settings["training_X"], self.settings["validation_X"]),
                                  getattr(best, "scale", (0.0, 1.0)))
            solved = (None, 0)
        if problem.kind == "sr":
            train_metric = [ind.fitness.values[0] for ind in population]
            val_metric = [validation_fitness(ind) for ind in population]
        else:
            train_metric = [100.0 * float(np.mean(ind.errors == 0)) for ind in population]
            val_metric = ([100.0 * self._test_sample_solved(ind) for ind in population]
                          if problem.kind == "ps" else train_metric)
        dynamics = self._dynamics(population, train_metric, extra or {})
        browser = None if self.headless else [
            (str(ind), t, v, len(ind), ind.height, getattr(ind, "uid", None))
            for ind, t, v in zip(population, train_metric, val_metric)]
        with self._cond:
            self._history["avg_training"].append(avg_training)
            self._history["avg_validation"].append(avg_validation)
            self._history["best_training"].append(best_training)
            self._history["best_validation"].append(best_validation)
            self._records.append({
                "expression": str(best), "size": len(best), "depth": best.height,
                "prediction": prediction, "nodes": _tree_nodes(best, self._variables),
                "downsample": None if downsample is None else np.array(downsample),
                "outliers": None if outliers is None else np.array(outliers),
                "solved": solved[0], "test_solved": solved[1],
                "share": cases[0], "outputs": cases[1], "errors": cases[2],
                "scale": getattr(best, "scale", None), "dynamics": dynamics, "population": browser,
                "uid": getattr(best, "uid", None),
            })

    def _run(self):
        try:
            self._evolve()
            with self._cond:
                if self._state in ("running", "paused"):
                    self._state = "finished"
        except Exception as exc:  # surface errors in the UI instead of dying silently
            with self._cond:
                self._state = "error"
                self._error = str(exc) if isinstance(exc, operators.OperatorError) else f"{type(exc).__name__}: {exc}"

    def _evolve(self):
        s = self.settings
        problem = s["problem"]
        if s.get("seed"):   # reproducible runs (experiments use one seed per run)
            random.seed(s["seed"])
            np.random.seed(s["seed"] % (2 ** 32))
        pset = get_pset(s)
        toolbox = base.Toolbox()
        self._toolbox = toolbox
        self._variables = set(pset.arguments) | {synthesis.ELEM}

        generators = {"Ramped Half-n-Half": gen_half_and_half, "Full": gen_full, "Grow": gen_grow}
        toolbox.register("expr", generators.get(s["initialization"], gen_half_and_half), pset=pset,
                         min_=s["min_initial_tree_depth"], max_=s["max_initial_tree_depth"])
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        train_X, train_y = s["training_X"], s["training_y"]
        val_X, val_y = s["validation_X"], s["validation_y"]
        val_cache = {}

        if problem.kind != "sr":
            arg_names = pset.arguments
            ret = problem.ret
            # Population averages on the test set use a fixed sample of test cases
            # (program execution is slow in Python); the best individual is always
            # evaluated on the full test set.
            sample_idx = random.Random(0).sample(range(len(val_y)), min(200, len(val_y)))
            val_sample = [(val_X[i], val_y[i]) for i in sample_idx]
            train_cases = list(zip(train_X, train_y))
            all_test = list(zip(val_X, val_y))

            def case_errors(individual, cases):
                func = synthesis.compile_tree(individual, arg_names)
                out = np.empty(len(cases))
                for i, (inputs, want) in enumerate(cases):
                    try:
                        out[i] = synthesis.case_error(ret, func(*inputs), want)
                    except Exception:  # crash or step limit
                        out[i] = synthesis.failure_error(ret, want)
                return out

            train_cache = {}

            def evaluate(individual):
                # identical programs appear often after crossover; run each one only once
                key = str(individual)
                errors = train_cache.get(key)
                if errors is None:
                    errors = case_errors(individual, train_cases)
                    if len(train_cache) > 20_000:
                        train_cache.clear()
                    train_cache[key] = errors
                individual.errors = errors
                return (float(errors.mean()),)

            def validation_fitness(individual):
                return float(sample_errors(individual).mean())

            def sample_errors(individual):
                key = str(individual)
                if key not in val_cache:
                    val_cache[key] = case_errors(individual, val_sample)
                return val_cache[key]

            def outputs(individual):
                func = synthesis.compile_tree(individual, arg_names)
                result = []
                for inputs, _ in train_cases:
                    try:
                        result.append(repr(func(*inputs)))
                    except Exception:
                        result.append("error")
                return result

            self._test_errors = lambda individual: case_errors(individual, all_test)
            self._test_sample_solved = lambda individual: float(np.mean(sample_errors(individual) == 0))
            self._outputs = outputs
        else:
            metric = s.get("fitness_function", "MSE")
            scaled = metric == "MSE + linear scaling"
            custom_fitness = operators.from_settings(s, "fitness")

            def custom_value(outputs, targets):
                name, fn = custom_fitness
                value = operators.call("fitness", name, fn, [float(o) for o in outputs], [float(t) for t in targets])
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    operators.fail("fitness", name, f"must return a number (got {value!r}).")
                return value if math.isfinite(value) else math.inf

            def evaluate(individual):
                if custom_fitness is not None:
                    outputs = _outputs_of(toolbox.compile(expr=individual), train_X)
                    individual.errors = np.minimum(np.abs(outputs - np.asarray(train_y, dtype=float)), 1e30)
                    individual.errors[~np.isfinite(individual.errors)] = 1e30
                    return (custom_value(outputs, train_y),)
                # Per-case errors are kept on the individual for lexicase.
                outputs = _outputs_of(toolbox.compile(expr=individual), train_X)
                scale = _linear_scaling(outputs, train_y) if scaled else (0.0, 1.0)
                if scaled:
                    individual.scale = scale
                errors = _case_errors(outputs, train_y, metric, scale)
                individual.errors = np.minimum(errors, 1e30)
                return (_aggregate(errors, metric),)

            def validation_fitness(individual):
                scale = getattr(individual, "scale", (0.0, 1.0))
                key = (str(individual), scale)
                if key not in val_cache:
                    outputs = _outputs_of(toolbox.compile(expr=individual), val_X)
                    val_cache[key] = (custom_value(outputs, val_y) if custom_fitness is not None else
                                      _aggregate(_case_errors(outputs, val_y, metric, scale), metric))
                return val_cache[key]

        selector = Selector(s)
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", selector)
        custom = {kind: operators.from_settings(s, kind) for kind in ("crossover", "mutation", "initialization",
                                                                      "bloat")}
        op_rng = random.Random(random.getrandbits(64))   # handed to custom operators

        def as_tree(kind, name, value):
            """A custom operator's result as a valid individual."""
            if not isinstance(value, list):
                operators.fail(kind, name, f"must return a tree (a list of nodes), got {type(value).__name__}.")
            open_slots = 1
            for node in value:
                if open_slots <= 0 or not hasattr(node, "arity"):
                    operators.fail(kind, name, "returned an invalid tree.")
                open_slots += node.arity - 1
            if open_slots != 0 or not value:
                operators.fail(kind, name, "returned an incomplete tree.")
            return value if isinstance(value, creator.Individual) else creator.Individual(value)

        if custom["crossover"] is not None:
            def custom_mate(ind1, ind2, _name=custom["crossover"][0], _fn=custom["crossover"][1]):
                result = operators.call("crossover", _name, _fn, ind1, ind2, op_rng)
                if not isinstance(result, (tuple, list)) or len(result) != 2 or not all(
                        isinstance(r, list) for r in result):
                    operators.fail("crossover", _name, "must return two children.")
                return as_tree("crossover", _name, result[0]), as_tree("crossover", _name, result[1])
            toolbox.register("mate", custom_mate)
        elif s.get("crossover_type") == "Leaf-biased (90/10)":
            toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        else:
            toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gen_full, min_=0, max_=2)

        def random_subtree(type_=None, max_depth=2):
            return gen_grow(pset, 0, max(0, int(max_depth)), type_=type_ if type_ is not None else pset.ret)
        mutations = {
            "Uniform (subtree)": partial(gp.mutUniform, expr=toolbox.expr_mut, pset=pset),
            "Node replacement": partial(gp.mutNodeReplacement, pset=pset),
            "Shrink": gp.mutShrink,
            "Ephemeral constants": partial(gp.mutEphemeral, mode="one"),
        }
        chosen_mutation = s.get("mutation_type", "Uniform (subtree)")
        if custom["mutation"] is not None:
            def custom_mutate(ind, _name=custom["mutation"][0], _fn=custom["mutation"][1]):
                return (as_tree("mutation", _name, operators.call("mutation", _name, _fn, ind, random_subtree, op_rng)),)
            toolbox.register("mutate", custom_mutate)
        elif chosen_mutation == "Mixed":
            toolbox.register("mutate", lambda ind: random.choice(list(mutations.values()))(ind))
        else:
            toolbox.register("mutate", mutations.get(chosen_mutation, mutations["Uniform (subtree)"]))
        if custom["bloat"] is not None:
            bloat_name, bloat_fn = custom["bloat"]

            def custom_limit(func):
                """Like gp.staticLimit, with the user's rule: a rejected child is replaced by its parent."""
                def wrapper(*args, **kwargs):
                    parents = [toolbox.clone(a) for a in args]
                    children = list(func(*args, **kwargs))
                    for i, child in enumerate(children):
                        parent = parents[min(i, len(parents) - 1)]
                        keep = operators.call("bloat", bloat_name, bloat_fn, len(child), child.height, len(parent),
                                              parent.height)
                        if not keep:
                            children[i] = parent
                    return tuple(children)
                return wrapper
            toolbox.decorate("mate", custom_limit)
            toolbox.decorate("mutate", custom_limit)
        limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=s["max_tree_depth"])
        toolbox.decorate("mate", limit)
        toolbox.decorate("mutate", limit)
        if s.get("bloat_control") == "Depth + size limit":
            size_limit = gp.staticLimit(key=len, max_value=int(s.get("max_tree_size", 100)))
            toolbox.decorate("mate", size_limit)
            toolbox.decorate("mutate", size_limit)

        if custom["initialization"] is not None:
            init_name, init_fn = custom["initialization"]
            trees = operators.call(
                "initialization", init_name, init_fn,
                lambda depth: creator.Individual(gen_full(pset, int(depth), int(depth))),
                lambda depth: creator.Individual(gen_grow(pset, 0, int(depth))),
                s["population_size"], s["min_initial_tree_depth"], s["max_initial_tree_depth"], op_rng)
            if not isinstance(trees, (list, tuple)) or len(trees) != s["population_size"]:
                operators.fail("initialization", init_name, f"must return {s['population_size']} trees.")
            population = [as_tree("initialization", init_name, t) for t in trees]
        else:
            population = toolbox.population(n=s["population_size"])
        hof = tools.HallOfFame(1)
        self._best_program, best_key = None, None

        def update_best(individuals):
            """Program synthesis: most solved training cases, ties by lower mean error."""
            nonlocal best_key
            for ind in individuals:
                key = (int(np.sum(ind.errors == 0)), -ind.fitness.values[0])
                if best_key is None or key > best_key:
                    best_key, self._best_program = key, toolbox.clone(ind)

        track = not self.headless   # genealogy for the ancestry view
        uids = itertools.count(1)

        def remember(individuals, generation):
            for ind in individuals:
                if problem.kind == "sr":
                    fit = float(ind.fitness.values[0])
                else:
                    fit = 100.0 * float(np.mean(np.asarray(ind.errors) == 0))
                self._genealogy[ind.uid] = (generation, ind.op, ind.parents, fit, len(ind), str(ind))

        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)
        if track:
            for ind in population:
                ind.uid, ind.parents, ind.op = next(uids), (), "init"
            remember(population, 1)
        hof.update(population)
        if problem.kind != "sr":
            update_best(population)
        self._record(population, hof, validation_fitness, None)

        n_elite = max(0, min(int(s.get("elitism", 0)), len(population) - 1))
        cxpb, mutpb = s["crossover_probability"], s["mutation_probability"]

        for _ in range(1, s["generations"]):
            if not self._wait_for_turn():
                return
            elites = [toolbox.clone(ind) for ind in tools.selBest(population, n_elite)] if n_elite else []
            selected = toolbox.select(population, len(population) - n_elite)
            counts = Counter(id(ind) for ind in selected)
            extra = {"parents": 100.0 * len(counts) / max(1, len(population)),
                     "top_parent": 100.0 * max(counts.values()) / max(1, len(selected))}

            # variation (like DEAP's varAnd), remembering operator and parent fitness per child
            offspring = [toolbox.clone(ind) for ind in selected]
            parent_fit = [ind.fitness.values[0] for ind in selected]
            parent_ids = [getattr(ind, "uid", None) for ind in selected]
            ops = [None] * len(offspring)
            partner = [None] * len(offspring)   # genealogy: the other parent of a crossover child
            mutated = [False] * len(offspring)
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values
                    ops[i - 1] = ops[i] = "cx"
                    partner[i - 1], partner[i] = i, i - 1
            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
                    ops[i] = "mut"
                    mutated[i] = True
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            effects = {"cx": [0, 0, 0], "mut": [0, 0, 0]}   # better, equal, worse than the parent
            for child, op, before in zip(offspring, ops, parent_fit):
                if op:
                    after = child.fitness.values[0]
                    effects[op][0 if after < before - 1e-12 else 2 if after > before + 1e-12 else 1] += 1
            extra["variation"] = effects

            if track:
                generation = len(self._records) + 1
                for i, child in enumerate(offspring):
                    crossed = partner[i] is not None
                    child.op = ("cx+mut" if crossed and mutated[i] else "cx" if crossed
                                else "mut" if mutated[i] else "copy")
                    child.parents = ((parent_ids[i], parent_ids[partner[i]]) if crossed else (parent_ids[i],))
                    child.uid = next(uids)
                for elite in elites:   # copied unchanged from the previous generation
                    elite.parents, elite.op, elite.uid = (elite.uid,), "elite", next(uids)
                remember(elites + offspring, generation)

            offspring = elites + offspring
            hof.update(offspring)
            if problem.kind != "sr":
                update_best(offspring)
            population[:] = offspring
            if len(val_cache) > 50_000:
                val_cache.clear()
            self._record(population, hof, validation_fitness, selector.last_downsample, selector.outliers, extra)
