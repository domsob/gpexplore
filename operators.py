"""User-defined GP operators written in Python (Additional Parameters → Custom operator).

Each operator is one Python function with a fixed signature (see KINDS and the templates).
It is stored by its function name in custom_operators.json and appears in the matching
dropdown. A run carries the code of its selected custom operators in its settings
(`custom_operators`), so experiment workers in other processes use exactly that code.
"""

import ast
import json
import math
import os
import random
from dataclasses import dataclass

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_operators.json")


@dataclass(frozen=True)
class Kind:
    key: str
    title: str          # e.g. "Selection"
    setting: str        # config / settings key whose value names the operator
    params: tuple       # parameter names of the function
    template: str


TEMPLATES = {
    "fitness": '''def mean_abs_error(outputs: list, targets: list) -> float:
    """Fitness of one program (symbolic regression): lower is better.
    outputs: values of the program on the training cases, targets: wanted values."""
    return sum(abs(o - t) for o, t in zip(outputs, targets)) / len(targets)
''',
    "selection": '''def binary_tournament(fitness: list, errors: list, sizes: list, k: int, rng) -> list:
    """Choose k parents; return their indices into the population.
    fitness[i]: fitness of individual i (lower is better), errors[i]: its error on each
    training case, sizes[i]: its number of nodes. rng: a random.Random."""
    chosen = []
    for _ in range(k):
        a, b = rng.randrange(len(fitness)), rng.randrange(len(fitness))
        chosen.append(a if fitness[a] <= fitness[b] else b)
    return chosen
''',
    "crossover": '''def one_point(parent1, parent2, rng):
    """Combine two parents; return two children.
    A tree is a list of nodes in prefix order (node.arity, node.name, node.ret = type);
    tree.searchSubtree(i) is the slice of the subtree that starts at position i.
    Swap only subtrees of the same type (node.ret)."""
    i = rng.randrange(len(parent1))
    matches = [j for j, node in enumerate(parent2) if node.ret == parent1[i].ret]
    if not matches:
        return parent1, parent2
    j = rng.choice(matches)
    s1, s2 = parent1.searchSubtree(i), parent2.searchSubtree(j)
    parent1[s1], parent2[s2] = parent2[s2], parent1[s1]
    return parent1, parent2
''',
    "mutation": '''def subtree_mutation(individual, random_subtree, rng):
    """Change one individual; return it.
    random_subtree(type, max_depth) creates a random subtree returning `type`
    (use the type of the node it replaces: individual[i].ret)."""
    i = rng.randrange(len(individual))
    s = individual.searchSubtree(i)
    individual[s] = random_subtree(individual[i].ret, 2)
    return individual
''',
    "initialization": '''def ramped_full(full, grow, size, min_depth, max_depth, rng):
    """Create the first generation: return `size` trees.
    full(depth): random tree with all leaves at exactly this depth,
    grow(depth): random tree of at most this depth."""
    depths = range(min_depth, max_depth + 1)
    return [full(depths[i % len(depths)]) for i in range(size)]
''',
    "bloat": '''def double_size_limit(child_size: int, child_depth: int, parent_size: int, parent_depth: int) -> bool:
    """Called for every child of crossover and mutation: True keeps the child,
    False keeps the parent instead. The max. tree depth limit always applies as well."""
    return child_size <= 2 * parent_size
''',
}

KINDS = {k.key: k for k in (
    Kind("fitness", "Fitness function", "fitness_function", ("outputs", "targets"), TEMPLATES["fitness"]),
    Kind("selection", "Selection", "selection_method", ("fitness", "errors", "sizes", "k", "rng"),
         TEMPLATES["selection"]),
    Kind("crossover", "Crossover", "crossover_type", ("parent1", "parent2", "rng"), TEMPLATES["crossover"]),
    Kind("mutation", "Mutation", "mutation_type", ("individual", "random_subtree", "rng"), TEMPLATES["mutation"]),
    Kind("initialization", "Initialization", "initialization",
         ("full", "grow", "size", "min_depth", "max_depth", "rng"), TEMPLATES["initialization"]),
    Kind("bloat", "Bloat control", "bloat_control", ("child_size", "child_depth", "parent_size", "parent_depth"),
         TEMPLATES["bloat"]),
)}

# names of the built-in choices per kind (a custom operator must not shadow them)
BUILTIN = {
    "fitness": ("MSE", "MAE", "RMSE", "MSE + linear scaling"),
    "selection": ("Tournament", "Lexicase", "Epsilon-Lexicase", "DS Lexicase", "DS Epsilon-Lexicase", "IDS Lexicase",
                  "IDS Epsilon-Lexicase", "ROIDS", "Random", "Best", "Worst"),
    "crossover": ("One-point", "Leaf-biased (90/10)"),
    "mutation": ("Uniform (subtree)", "Node replacement", "Shrink", "Ephemeral constants", "Mixed"),
    "initialization": ("Ramped Half-n-Half", "Full", "Grow"),
    "bloat": ("Depth limit", "Depth + size limit", "Lexicographic parsimony"),
}

REGISTRY = {key: {} for key in KINDS}   # kind -> {name: code}


class OperatorError(Exception):
    """A custom operator failed or returned something unusable (shown as the run's error)."""


# --------------------------------------------------------------------------
# Checking and compiling
# --------------------------------------------------------------------------

def check(kind, code):
    """Static check (nothing is executed): returns the function name."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        raise OperatorError(f"Syntax error in line {exc.lineno}: {exc.msg}") from None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise OperatorError("Define exactly one function (helper functions go inside it).")
    func = funcs[0]
    expected = KINDS[kind].params
    if len(func.args.args) != len(expected):
        raise OperatorError(f"The function needs {len(expected)} parameters: {', '.join(expected)}.")
    if func.name in BUILTIN[kind]:
        raise OperatorError(f"'{func.name}' is the name of a built-in {KINDS[kind].title.lower()}.")
    return func.name


_compiled = {}


def compile_operator(kind, code):
    """The function defined by `code` (cached by code)."""
    key = (kind, code)
    if key not in _compiled:
        name = check(kind, code)
        namespace = {"math": math, "random": random}
        try:
            exec(compile(code, f"<custom {kind}>", "exec"), namespace)
        except Exception as exc:  # the user's own code
            raise OperatorError(f"{type(exc).__name__}: {exc}") from None
        _compiled[key] = (name, namespace[name])
    return _compiled[key]


def is_custom(kind, value):
    return value in REGISTRY[kind]


def selected(settings):
    """Code of the custom operators chosen in `settings` (stored with the run settings)."""
    return {kind: REGISTRY[kind][settings[k.setting]] for kind, k in KINDS.items()
            if settings.get(k.setting) in REGISTRY[kind]}


def from_settings(settings, kind):
    """(name, function) of the custom operator of this kind in a run's settings, or None."""
    code = (settings.get("custom_operators") or {}).get(kind)
    if code is None:
        return None
    return compile_operator(kind, code)


def call(kind, name, fn, *args):
    """Run a custom operator; errors name the operator."""
    try:
        return fn(*args)
    except OperatorError:
        raise
    except Exception as exc:
        raise OperatorError(f"Custom {KINDS[kind].title.lower()} '{name}': {type(exc).__name__}: {exc}") from None


def fail(kind, name, message):
    raise OperatorError(f"Custom {KINDS[kind].title.lower()} '{name}': {message}")


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def load():
    for key in REGISTRY:
        REGISTRY[key].clear()
    try:
        with open(FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    for item in data if isinstance(data, list) else []:
        kind, code = item.get("kind"), item.get("code", "")
        if kind in REGISTRY:
            try:
                REGISTRY[kind][check(kind, code)] = code
            except OperatorError:
                continue   # a broken entry is skipped, the others still load


def save():
    data = [{"kind": kind, "code": code} for kind, ops in REGISTRY.items() for code in ops.values()]
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add(kind, code):
    name = check(kind, code)
    REGISTRY[kind][name] = code
    save()
    return name


def delete(kind, name):
    REGISTRY[kind].pop(name, None)
    save()


# --------------------------------------------------------------------------
# Trying an operator before it is added
# --------------------------------------------------------------------------

SR_TEST_PROBLEM = "x**4 + x**3 + x**2 + x"   # Koza-1, for fitness functions on other problems


def smoke_test(kind, code, settings):
    """A short run (20 individuals, 3 generations) with the operator; returns an error text or None."""
    import gp
    import problems
    try:
        name, _ = compile_operator(kind, code)
    except OperatorError as exc:
        return str(exc)
    s = dict(settings)
    s.update(population_size=20, generations=3, seed=1, crossover_probability=0.9, mutation_probability=0.5)
    s[KINDS[kind].setting] = name
    s["custom_operators"] = {**(s.get("custom_operators") or {}), kind: code}
    if kind == "fitness" and s["problem"].kind != "sr":
        problem = problems.get(SR_TEST_PROBLEM)
        s["problem"] = problem
        s["training_X"], s["training_y"], s["validation_X"], s["validation_y"] = problems.make_data(problem, 20, 20)
    run = gp.GPRun(s, headless=True)
    run._run()
    snap = run.snapshot()
    return snap.error or None



# --------------------------------------------------------------------------
# Custom functions for the function sets (Function & Terminal Set tab)
# --------------------------------------------------------------------------

FUNCTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_functions.json")
AREAS = {"sr": "Symbolic regression", "ps": "Program synthesis", "logic": "Logic"}
FUNCTION_TEMPLATES = {
    "sr": '''def square(x: float) -> float:
    """A function for symbolic regression: all inputs and the output are floats.
    It is applied to single numbers; an error or an infinite result counts as NaN."""
    return x * x
''',
    "ps": '''def is_even(n: int) -> bool:
    """A building block for program synthesis. Allowed types: int, float, bool, str,
    list[int] (inputs only). It is used wherever its types occur in a problem."""
    return n % 2 == 0
''',
    "logic": '''def implies(a: bool, b: bool) -> bool:
    """A gate for Boolean problems: all inputs and the output are bool."""
    return (not a) or b
''',
}
FUNCTIONS = {area: {} for area in AREAS}   # area -> {name: code}
ENABLED = {}                               # name -> switched on in the function set

# names a custom function must not take (built-in functions, blocks, gates, inputs)
RESERVED = {"add", "sub", "mul", "protectedDiv", "analyticQuotient", "neg", "sin", "cos", "x", "y", "e",
            "AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR", "IF"}


def _reserved():
    import synthesis
    return RESERVED | set(synthesis.BLOCKS_BUILTIN)


def check_function(area, code):
    """Static check: (name, argument types, return type, signature text)."""
    import synthesis
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        raise OperatorError(f"Syntax error in line {exc.lineno}: {exc.msg}") from None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise OperatorError("Define exactly one function (helper functions go inside it).")
    func = funcs[0]
    if not 1 <= len(func.args.args) <= 3:
        raise OperatorError("The function needs one to three parameters.")
    if func.name in _reserved():
        raise OperatorError(f"'{func.name}' is already used by a built-in function; choose another name.")
    other = [a for a in AREAS if a != area and func.name in FUNCTIONS[a]]
    if other:
        raise OperatorError(f"'{func.name}' is already a custom {AREAS[other[0]].lower()} function.")

    def annotation(node, default):
        text = ast.unparse(node) if node is not None else ""
        if not text:
            return default
        if text not in synthesis.ANNOTATIONS:
            raise OperatorError(f"Unsupported type '{text}'. Use int, float, bool, str or list[int].")
        return synthesis.ANNOTATIONS[text]

    fixed = {"sr": synthesis.Num, "logic": synthesis.Bool}.get(area)
    args = tuple(annotation(a.annotation, fixed) for a in func.args.args)
    ret = annotation(func.returns, fixed)
    if fixed is not None and (any(t is not fixed for t in args) or ret is not fixed):
        kind = "float" if area == "sr" else "bool"
        raise OperatorError(f"All inputs and the output must be {kind} here.")
    if area == "ps" and (None in args or ret is None):
        raise OperatorError("Annotate every parameter and the return type, e.g. def f(n: int) -> bool.")
    if ret is synthesis.Vec:
        raise OperatorError("Return type list[int] is not supported; return int, float, bool or str.")
    params = ", ".join(f"{a.arg}: {synthesis.TYPE_NAMES[t]}" for a, t in zip(func.args.args, args))
    return func.name, args, ret, f"{func.name}({params}) -> {synthesis.TYPE_NAMES[ret]}"


_functions = {}


def compile_function(area, code):
    """(name, args, ret, callable) with the result converted to the declared type."""
    key = (area, code)
    if key not in _functions:
        import synthesis
        name, args, ret, _ = check_function(area, code)
        namespace = {"math": math}
        try:
            exec(compile(code, f"<custom function {name}>", "exec"), namespace)
        except Exception as exc:
            raise OperatorError(f"{type(exc).__name__}: {exc}") from None
        fn = namespace[name]
        if area == "sr":
            def call(*a, _fn=fn):
                try:
                    v = float(_fn(*a))
                except Exception:
                    return math.nan
                return v if math.isfinite(v) else math.nan
        else:
            convert = {synthesis.Int: lambda v: synthesis._clip(int(v)), synthesis.Num: float,
                       synthesis.Bool: bool, synthesis.Str: lambda v: str(v)[:synthesis.STR_LIMIT]}[ret]

            def call(*a, _fn=fn, _convert=convert):
                return _convert(_fn(*a))   # errors count as a failed program run
        _functions[key] = (name, args, ret, call)
    return _functions[key]


def selected_functions():
    """Code of the switched-on custom functions per area (stored with the run settings)."""
    return {area: [code for name, code in FUNCTIONS[area].items() if ENABLED.get(name, True)] for area in AREAS}


def numeric_functions(settings):
    """(name, arity, callable) of the run's custom symbolic regression functions."""
    out = []
    for code in (settings.get("custom_functions") or {}).get("sr", []):
        name, args, _, fn = compile_function("sr", code)
        out.append((name, len(args), fn))
    return out


def function_blocks(settings, kind):
    """Building blocks of the run's custom program synthesis / logic functions (registered for the
    interpreter)."""
    import synthesis
    if kind not in ("ps", "logic"):
        return []
    blocks = []
    for code in (settings.get("custom_functions") or {}).get(kind, []):
        name, args, ret, fn = compile_function(kind, code)
        block = synthesis.Block(name, args, ret, "strict", fn, name)
        synthesis.BLOCKS[name] = block
        blocks.append(block)
    return blocks


def test_function(area, code):
    """Calls the function with typical inputs; returns an error text or None."""
    import itertools
    import synthesis
    try:
        name, args, ret, _ = check_function(area, code)
        namespace = {"math": math}
        exec(compile(code, f"<custom function {name}>", "exec"), namespace)
        fn = namespace[name]
    except OperatorError as exc:
        return str(exc)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    rng = random.Random(0)
    samples = {synthesis.Num: lambda: rng.uniform(-3, 3), synthesis.Int: lambda: rng.randint(-100, 100),
               synthesis.Bool: lambda: rng.random() < 0.5,
               synthesis.Str: lambda: "".join(rng.choice("abcxyz 01") for _ in range(rng.randint(0, 8))),
               synthesis.Vec: lambda: tuple(rng.randint(-10, 10) for _ in range(rng.randint(0, 6)))}
    if area == "logic":
        inputs = list(itertools.product((False, True), repeat=len(args)))
    else:
        inputs = [tuple(samples[t]() for t in args) for _ in range(30)]
    expected = {synthesis.Num: (int, float), synthesis.Int: int, synthesis.Bool: (bool, int), synthesis.Str: str}[ret]
    worked, last_error = 0, None
    for values in inputs:
        try:
            out = fn(*values)
        except Exception as exc:
            last_error = exc
            continue
        if not isinstance(out, expected):
            return (f"{name}{values} returned {out!r}; expected {synthesis.TYPE_NAMES[ret]}.")
        worked += 1
    if not worked:
        return f"The function failed for every test input: {type(last_error).__name__}: {last_error}"
    return None


def load_functions():
    for area in FUNCTIONS:
        FUNCTIONS[area].clear()
    ENABLED.clear()
    try:
        with open(FUNCTIONS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    for item in data if isinstance(data, list) else []:
        area, code = item.get("area"), item.get("code", "")
        if area in FUNCTIONS:
            try:
                name = check_function(area, code)[0]
            except OperatorError:
                continue
            FUNCTIONS[area][name] = code
            ENABLED[name] = bool(item.get("enabled", True))


def save_functions():
    data = [{"area": area, "code": code, "enabled": ENABLED.get(name, True)}
            for area, items in FUNCTIONS.items() for name, code in items.items()]
    with open(FUNCTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add_function(area, code):
    name = check_function(area, code)[0]
    FUNCTIONS[area][name] = code
    ENABLED[name] = True
    save_functions()
    return name


def delete_function(area, name):
    FUNCTIONS[area].pop(name, None)
    ENABLED.pop(name, None)
    save_functions()


def set_enabled(name, flag):
    ENABLED[name] = bool(flag)
    save_functions()


load()
load_functions()
