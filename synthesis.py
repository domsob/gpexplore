"""Program synthesis: typed building blocks, a small tree interpreter, error
functions, the PSB1 problems and user-defined Python target functions.

Trees are strongly typed (DEAP PrimitiveSetTyped). The type tags below are
plain marker classes on purpose: DEAP files primitives under every super-type
of their return type, and `bool` is a subclass of `int` in Python.
"""

import ast
import math
import random
import string
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Type tags
# --------------------------------------------------------------------------

class Num:   # float (symbolic regression)
    pass


class Int:
    pass


class Bool:
    pass


class Str:
    pass


class Vec:   # list of int
    pass


TYPE_NAMES = {Num: "float", Int: "int", Bool: "bool", Str: "str", Vec: "list[int]"}
ANNOTATIONS = {"float": Num, "int": Int, "bool": Bool, "str": Str, "list[int]": Vec, "List[int]": Vec}
DEFAULTS = {Num: 0.0, Int: 0, Bool: False, Str: "", Vec: ()}


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

INT_LIMIT = 10 ** 9      # integers are clipped to keep programs fast
STR_LIMIT = 200          # maximum string length
LOOP_LIMIT = 100         # iterations of a single range loop
STEP_LIMIT = 500         # loop iterations per program execution (stops costly nested loops)


class StepLimit(Exception):
    pass


def _clip(v):
    return INT_LIMIT if v > INT_LIMIT else -INT_LIMIT if v < -INT_LIMIT else v


@dataclass(frozen=True)
class Block:
    name: str            # unique primitive name (shown in expressions)
    args: tuple
    ret: type
    kind: str            # "strict" (fn on values), "if", "if_else", "loop_vec_sum", ...
    fn: object = None
    symbol: str = ""     # label in the tree view


def _strict(name, args, ret, fn, symbol=""):
    return Block(name, tuple(args), ret, "strict", fn, symbol or name)


def _int_div(a, b):
    return 0 if b == 0 else _clip(a // b)


def _int_mod(a, b):
    return 0 if b == 0 else a % b


def _concat(a, b):
    return (a + b)[:STR_LIMIT]


# Groups of building blocks that can be switched on in the "Function & terminal set" tab.
# (config flag, label, description, blocks). Blocks whose types do not occur in a
# problem are left out automatically (e.g. vector loops for a problem without vectors).
GROUPS = [
    ("ps_if", "if(c, a)", "Returns a if c is true, otherwise the type's default (0, \"\", false).",
     [Block(f"if_{TYPE_NAMES[t].split('[')[0]}", (Bool, t), t, "if", symbol="if") for t in (Int, Num, Str, Bool)]),
    ("ps_if_else", "if_else(c, a, b)", "Returns a if c is true, otherwise b.",
     [Block(f"if_else_{TYPE_NAMES[t].split('[')[0]}", (Bool, t, t), t, "if_else", symbol="if-else")
      for t in (Int, Num, Str, Bool)]),
    ("ps_compare", "<  >  ==", "Comparisons of numbers and strings, returning a boolean.",
     [_strict("lt", (Int, Int), Bool, lambda a, b: a < b, "<"),
      _strict("gt", (Int, Int), Bool, lambda a, b: a > b, ">"),
      _strict("eq", (Int, Int), Bool, lambda a, b: a == b, "=="),
      _strict("lt_f", (Num, Num), Bool, lambda a, b: a < b, "<"),
      _strict("gt_f", (Num, Num), Bool, lambda a, b: a > b, ">"),
      _strict("eq_str", (Str, Str), Bool, lambda a, b: a == b, "==")]),
    ("ps_bool", "and  or  not", "Boolean logic.",
     [_strict("and_", (Bool, Bool), Bool, lambda a, b: a and b, "and"),
      _strict("or_", (Bool, Bool), Bool, lambda a, b: a or b, "or"),
      _strict("not_", (Bool,), Bool, lambda a: not a, "not")]),
    ("ps_int", "+  −  ×  ÷  %", "Integer arithmetic; division and modulo by 0 return 0.",
     [_strict("add_i", (Int, Int), Int, lambda a, b: _clip(a + b), "+"),
      _strict("sub_i", (Int, Int), Int, lambda a, b: _clip(a - b), "−"),
      _strict("mul_i", (Int, Int), Int, lambda a, b: _clip(a * b), "×"),
      _strict("div_i", (Int, Int), Int, _int_div, "÷"),
      _strict("mod_i", (Int, Int), Int, _int_mod, "%")]),
    ("ps_for_vec", "for e in v", "Loops over a vector; the body sees the current element as e. "
                                  "for_sum adds up the body, for_count counts where the body is true.",
     [Block("for_sum", (Vec, Int), Int, "loop_vec_sum", symbol="for-sum"),
      Block("for_count", (Vec, Bool), Int, "loop_vec_count", symbol="for-count")]),
    ("ps_for_range", "for e in range(n)", "Loop with e = 0 … n−1 (at most 100 iterations), summing the body.",
     [Block("for_range_sum", (Int, Int), Int, "loop_range_sum", symbol="for-range")]),
    ("ps_str", "len  concat", "String length and concatenation.",
     [_strict("len_str", (Str,), Int, len, "len"),
      _strict("concat", (Str, Str), Str, _concat, "++")]),
    ("ps_vec", "len  sum", "Length and sum of a vector.",
     [_strict("len_vec", (Vec,), Int, len, "len"),
      _strict("sum_vec", (Vec,), Int, lambda v: _clip(sum(v)), "sum")]),
]
GROUP_FLAGS = [g[0] for g in GROUPS]
BLOCKS = {b.name: b for g in GROUPS for b in g[3]}
LOOP_KINDS = {"loop_vec_sum", "loop_vec_count", "loop_range_sum"}
ELEM = "e"   # name of the loop variable terminal


def enabled_blocks(settings):
    return [b for flag, _, _, blocks in GROUPS if settings.get(flag) for b in blocks]


def uses_loops(blocks):
    return any(b.kind in LOOP_KINDS for b in blocks)


# --------------------------------------------------------------------------
# Interpreter (closures built once per individual)
# --------------------------------------------------------------------------

class Env:
    __slots__ = ("inputs", "elem", "steps")

    def __init__(self, inputs):
        self.inputs, self.elem, self.steps = inputs, 0, 0

    def tick(self):
        self.steps += 1
        if self.steps > STEP_LIMIT:
            raise StepLimit()


def compile_tree(individual, arg_names):
    """Turn a DEAP tree into a callable f(*inputs) with lazy if / loop semantics."""
    nodes = list(individual)
    argindex = {n: i for i, n in enumerate(arg_names)}
    pos = 0

    def build():
        nonlocal pos
        node = nodes[pos]
        pos += 1
        if node.arity == 0:
            # symbolic terminals (conv_fct is str): renamed arguments and the loop variable
            if getattr(node, "conv_fct", None) is str:
                if node.value in argindex:
                    k = argindex[node.value]
                    return lambda env: env.inputs[k]
                if node.value == ELEM:
                    return lambda env: env.elem
            value = node.value
            return lambda env: value
        block = BLOCKS.get(node.name)
        kids = [build() for _ in range(node.arity)]
        if block is None:  # numeric primitive from symbolic regression (add, sin, ...)
            fn = node_function(node.name)
            return lambda env: fn(*[k(env) for k in kids])
        return _make(block, kids)

    root = build()

    def run(*inputs):
        return root(Env(inputs))
    return run


_NUMERIC = {}


def register_numeric(name, fn):
    _NUMERIC[name] = fn


def node_function(name):
    return _NUMERIC[name]


def _make(block, kids):
    kind = block.kind
    if kind == "strict":
        fn = block.fn
        if len(kids) == 1:
            a, = kids
            return lambda env: fn(a(env))
        if len(kids) == 2:
            a, b = kids
            return lambda env: fn(a(env), b(env))
        return lambda env: fn(*[k(env) for k in kids])
    if kind == "if":
        c, a = kids
        default = DEFAULTS[block.ret]
        return lambda env: a(env) if c(env) else default
    if kind == "if_else":
        c, a, b = kids
        return lambda env: a(env) if c(env) else b(env)
    if kind in ("loop_vec_sum", "loop_vec_count", "loop_range_sum"):
        source, body = kids
        counting = kind == "loop_vec_count"
        ranged = kind == "loop_range_sum"

        def loop(env):
            items = range(max(0, min(source(env), LOOP_LIMIT))) if ranged else source(env)
            saved, total = env.elem, 0
            for item in items:
                env.tick()
                env.elem = item
                value = body(env)
                total += (1 if value else 0) if counting else value
            env.elem = saved
            return _clip(total)
        return loop
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


ERROR_CAP = 1e6


def case_error(output_type, got, want):
    """PSB1-style errors: 0/1 for booleans, absolute difference for numbers,
    Levenshtein distance for strings and vectors."""
    if output_type is Bool:
        return 0.0 if bool(got) == bool(want) else 1.0
    if output_type in (Int, Num):
        return float(min(abs(got - want), ERROR_CAP))
    return float(levenshtein(got, want))


def failure_error(output_type, want):
    """Error for a program that crashed or exceeded the step limit."""
    if output_type is Bool:
        return 1.0
    if output_type in (Int, Num):
        return ERROR_CAP
    return float(len(want) + 10)


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------

@dataclass
class SynthesisProblem:
    expression: str                  # registry key / label in the dropdown
    arg_names: tuple
    arg_types: tuple
    ret: type
    generate: object                 # generate(n_edge_max, n_random, rng) -> list of (inputs, output)
    edge_cases: object = None        # () -> list of input tuples (always first in the training set)
    int_constants: tuple = ()
    str_constants: tuple = ()
    float_constants: tuple = ()
    erc_range: tuple = (-10, 10)
    train_size: int = 100
    test_size: int = 1000
    name: str = ""
    custom: bool = False
    code: str = ""                   # source of a user-defined target function
    kind: str = "ps"
    limits: tuple = None
    spec: object = None              # InputSpec of a custom problem

    @property
    def inputs(self):
        return len(self.arg_names)

    @property
    def label(self):
        return self.expression

    def edge_count(self, train_size):
        """How many of the first training cases are fixed edge cases."""
        return min(len(self.edge_cases()), train_size) if self.edge_cases else 0

    def describe_inputs(self, inputs, limit=70):
        parts = []
        for name, value in zip(self.arg_names, inputs):
            text = repr(list(value)) if isinstance(value, tuple) else repr(value)
            parts.append(f"{name}={text}")
        text = ", ".join(parts)
        return text if len(text) <= limit else text[:limit - 1] + "…"


def _random_string(rng, lo, hi, alphabet=string.ascii_lowercase + string.digits + " "):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))


# --- Compare String Lengths (PSB1) ----------------------------------------

def _csl_target(a, b, c):
    return len(a) < len(b) < len(c)


def _csl_edges():
    small = ["", "a", "bc"]
    perms = [(a, b, c) for a in small for b in small for c in small if len({a, b, c}) == 3]
    two_empty = [("", "", "a"), ("", "a", ""), ("a", "", ""), ("", "", "xyz"), ("", "xyz", ""), ("xyz", "", "")]
    same_diff = [("a", "a", "bb"), ("a", "bb", "a"), ("bb", "a", "a"), ("bb", "bb", "a"), ("bb", "a", "bb"),
                 ("a", "bb", "bb"), ("abc", "abc", "de"), ("abc", "de", "de"), ("de", "abc", "abc")]
    return [("", "", "")] + perms + two_empty + same_diff


def _csl_random(rng):
    r = rng.random()
    if r < 0.1:
        s = _random_string(rng, 0, 49)
        return (s, s, s)
    triple = [_random_string(rng, 0, 49) for _ in range(3)]
    if r < 0.3:
        triple.sort(key=len)
    return tuple(triple)


# --- Small or Large (PSB1) -------------------------------------------------

def _sol_target(n):
    return "small" if n < 1000 else "large" if n >= 2000 else ""


def _sol_edges():
    return [(v,) for v in (-10000, 0, 980, 990, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1010, 1500,
                           1990, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2010, 10000)]


def _sol_random(rng):
    return (rng.randint(-10000, 10000),)


# --- Count Odds (PSB1) -----------------------------------------------------

def _co_target(v):
    return sum(1 for x in v if x % 2 != 0)


def _co_edges():
    singles = [(x,) for x in range(-10, 11)] + [(-1000,), (-999,), (999,), (1000,)]
    pairs = [(2, 3), (3, 2), (2, 2), (3, 3), (-1, -2), (0, -7)]
    return [((),)] + [(s,) for s in singles] + [(p,) for p in pairs]


def _co_random(rng):
    n = rng.randint(0, 50)
    r = rng.random()
    if r < 0.1:
        return (tuple(rng.randrange(-999, 1000, 2) for _ in range(n)),)
    if r < 0.2:
        return (tuple(rng.randrange(-1000, 1001, 2) for _ in range(n)),)
    return (tuple(rng.randint(-1000, 1000) for _ in range(n)),)


def _make_generator(target, edges, random_input):
    def generate(train_size, test_size, seed=None):
        rng = random.Random(seed)
        edge = list(edges())[:train_size] if edges else []
        train_inputs = edge + [random_input(rng) for _ in range(train_size - len(edge))]
        test_inputs = [random_input(rng) for _ in range(test_size)]
        return ([(i, target(*i)) for i in train_inputs], [(i, target(*i)) for i in test_inputs])
    return generate


PSB1 = [
    SynthesisProblem("Compare String Lengths", ("s1", "s2", "s3"), (Str, Str, Str), Bool,
                     _make_generator(_csl_target, _csl_edges, _csl_random), edge_cases=_csl_edges, name="PSB1",
                     train_size=100, test_size=1000, erc_range=(-10, 10)),
    SynthesisProblem("Small or Large", ("n",), (Int,), Str,
                     _make_generator(_sol_target, _sol_edges, _sol_random), edge_cases=_sol_edges, name="PSB1",
                     str_constants=("small", "large"), int_constants=(1000, 2000),
                     erc_range=(-10000, 10000), train_size=100, test_size=1000),
    SynthesisProblem("Count Odds", ("v",), (Vec,), Int,
                     _make_generator(_co_target, _co_edges, _co_random), edge_cases=_co_edges, name="PSB1",
                     int_constants=(0, 1, 2), erc_range=(-1000, 1000), train_size=200, test_size=2000),
]


# --------------------------------------------------------------------------
# User-defined Python functions
# --------------------------------------------------------------------------

ALPHABETS = {
    "lowercase": string.ascii_lowercase,
    "letters": string.ascii_letters,
    "letters + digits": string.ascii_letters + string.digits,
    "letters, digits, space": string.ascii_letters + string.digits + " ",
    "visible ASCII": "".join(chr(c) for c in range(33, 127)),
    "custom": "",
}


@dataclass
class InputSpec:
    """How random inputs of a custom program synthesis problem are generated."""
    int_range: tuple = (-100, 100)
    float_range: tuple = (-10.0, 10.0)
    str_length: tuple = (0, 20)
    alphabet: str = "lowercase"
    custom_chars: str = ""
    vec_length: tuple = (0, 20)
    vec_range: tuple = (-100, 100)

    @property
    def chars(self):
        return self.custom_chars if self.alphabet == "custom" else ALPHABETS[self.alphabet]

    def validate(self, types):
        def check_range(rng, what, integer=True, minimum=None, maximum=None):
            lo, hi = rng
            if integer and (lo != int(lo) or hi != int(hi)):
                raise ProgramError(f"{what}: use whole numbers.")
            if lo > hi or (not integer and lo == hi):
                raise ProgramError(f"{what}: 'from' must not be larger than 'to'." if integer else f"{what}: 'from' must be smaller than 'to'.")
            if minimum is not None and lo < minimum:
                raise ProgramError(f"{what}: must be at least {minimum}.")
            if maximum is not None and hi > maximum:
                raise ProgramError(f"{what}: at most {maximum}.")
        if Int in types:
            check_range(self.int_range, "int range")
        if Num in types:
            check_range(self.float_range, "float range", integer=False)
        if Str in types:
            check_range(self.str_length, "string length", minimum=0, maximum=STR_LIMIT)
            if not self.chars:
                raise ProgramError("Enter the characters strings may contain.")
        if Vec in types:
            check_range(self.vec_length, "list length", minimum=0, maximum=100)
            check_range(self.vec_range, "list values")

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data):
        spec = cls()
        for key, value in (data or {}).items():
            if hasattr(spec, key):
                setattr(spec, key, tuple(value) if isinstance(value, list) else value)
        return spec


TEMPLATE = '''def last_digit(n: int) -> int:
    return abs(n) % 10
'''


class ProgramError(ValueError):
    pass


def _annotation(node):
    text = ast.unparse(node) if node is not None else ""
    if text not in ANNOTATIONS:
        raise ProgramError(f"Unsupported type '{text or '?'}'. Use int, float, bool, str or list[int].")
    return ANNOTATIONS[text]


MAX_CONSTANTS = 10   # per type


def literal_constants(func):
    """Numbers and strings written in the body of a function (docstring and annotations
    excluded), as they become constants of the primitive set: (ints, floats, strs).
    A minus sign in front of a number belongs to the constant (-5 stays -5)."""
    body = list(func.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]   # docstring
    ints, floats, strs = set(), set(), set()
    negated = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                    and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)):
                negated.add(id(node.operand))
                value = -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
            elif isinstance(node, ast.Constant) and id(node) not in negated:
                value = node.value
            else:
                continue
            if type(value) is int:
                ints.add(value)
            elif type(value) is float and math.isfinite(value):
                floats.add(value)
            elif isinstance(value, str) and value and len(value) <= 30:
                strs.add(value)
    return (tuple(sorted(ints)[:MAX_CONSTANTS]), tuple(sorted(floats)[:MAX_CONSTANTS]),
            tuple(sorted(strs)[:MAX_CONSTANTS]))


def format_constants(ints, floats, strs):
    return ", ".join([str(c) for c in ints] + [repr(c) for c in floats] + [repr(c) for c in strs])


def check_signature(code):
    """Static check while typing (nothing is executed): returns the function name, a short
    signature text, the parameter types and the constants found in the code."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        raise ProgramError(f"Syntax error in line {exc.lineno}: {exc.msg}") from None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ProgramError("Define exactly one function, e.g. def f(n: int) -> int: …")
    func = funcs[0]
    if not 1 <= len(func.args.args) <= 3:
        raise ProgramError("The function needs one to three parameters.")
    types = [_annotation(a.annotation) for a in func.args.args]
    params = ", ".join(f"{a.arg}: {TYPE_NAMES[t]}" for a, t in zip(func.args.args, types))
    ret = _annotation(func.returns)
    if ret is Vec:
        raise ProgramError("Return type list[int] is not supported; return int, float, bool or str.")
    return func.name, f"{func.name}({params}) -> {TYPE_NAMES[ret]}", set(types), literal_constants(func)


def parse_program(code, spec=None):
    """Check a user function and build a problem from it.

    The code must define one function with annotated parameters and return
    type. Inputs are generated at random (plus a few edge cases), outputs come
    from running the function. Number and string literals in the code become
    constants of the primitive set (see literal_constants)."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        raise ProgramError(f"Syntax error in line {exc.lineno}: {exc.msg}") from None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ProgramError("Define exactly one function, e.g. def f(n: int) -> int: …")
    func = funcs[0]
    args = func.args.args
    if not 1 <= len(args) <= 3:
        raise ProgramError("The function needs one to three parameters.")
    arg_names = tuple(a.arg for a in args)
    if ELEM in arg_names:
        raise ProgramError(f"'{ELEM}' is reserved for the loop variable.")
    arg_types = tuple(_annotation(a.annotation) for a in args)
    ret = _annotation(func.returns)
    if ret is Vec:
        raise ProgramError("Return type list[int] is not supported; return int, float, bool or str.")

    namespace = {}
    try:
        exec(compile(tree, "<custom problem>", "exec"), namespace)
    except Exception as exc:  # the user's own code
        raise ProgramError(f"{type(exc).__name__}: {exc}") from None
    target = namespace[func.name]

    ints, floats, strs = literal_constants(func)
    spec = spec or InputSpec()
    spec.validate(set(arg_types))
    lo, hi = (int(v) for v in spec.int_range)
    flo, fhi = (float(v) for v in spec.float_range)
    slo, shi = (int(v) for v in spec.str_length)
    llo, lhi = (int(v) for v in spec.vec_length)
    vlo, vhi = (int(v) for v in spec.vec_range)
    chars = spec.chars

    def random_value(t, rng):
        if t is Int:
            return rng.randint(lo, hi)
        if t is Num:
            return rng.uniform(flo, fhi)
        if t is Bool:
            return rng.random() < 0.5
        if t is Str:
            return _random_string(rng, slo, shi, chars)
        return tuple(rng.randint(vlo, vhi) for _ in range(rng.randint(llo, lhi)))

    def within(values, a, b):
        return list(dict.fromkeys(v for v in values if a <= v <= b))

    str_edges = list(dict.fromkeys(["" * slo, chars[0] * max(slo, 1), chars[-1] * shi][:3]))
    vec_edges = [tuple([v] * llo) for v in within([0, 1, vlo, vhi], vlo, vhi)] + [tuple([vhi] * lhi)]
    edge_values = {Int: within([0, 1, -1, lo, hi], lo, hi) or [lo], Num: within([0.0, 1.0, -1.0, flo, fhi], flo, fhi),
                   Bool: [False, True], Str: [e for e in str_edges if slo <= len(e) <= shi] or [chars[0] * slo],
                   Vec: list(dict.fromkeys(vec_edges))}

    def edges():
        cases = []
        for k in range(max(len(edge_values[t]) for t in arg_types)):
            cases.append(tuple(edge_values[t][k % len(edge_values[t])] for t in arg_types))
        return cases

    def random_input(rng):
        return tuple(random_value(t, rng) for t in arg_types)

    def safe_target(*inputs):
        out = target(*inputs)
        expected = {Int: int, Num: (int, float), Bool: bool, Str: str}[ret]
        if not isinstance(out, expected) or (ret is Int and isinstance(out, bool)):
            raise ProgramError(f"{func.name}{inputs} returned {out!r}, expected {TYPE_NAMES[ret]}.")
        return float(out) if ret is Num else out

    def generate(train_size, test_size, seed=None):
        rng = random.Random(seed)
        result = []
        for size, use_edges in ((train_size, True), (test_size, False)):
            cases, tries = [], 0
            pool = edges() if use_edges else []
            while len(cases) < size and tries < size * 20 + 50:
                tries += 1
                inputs = pool.pop(0) if pool else random_input(rng)
                try:
                    cases.append((inputs, safe_target(*inputs)))
                except ProgramError:
                    raise
                except Exception:
                    continue  # inputs the function does not accept are skipped
            if len(cases) < size:
                raise ProgramError("The function raised errors for most generated inputs.")
            result.append(cases)
        return tuple(result)

    generate(5, 5, seed=0)  # fail early with a readable message
    return SynthesisProblem(func.name, arg_names, arg_types, ret, generate, edge_cases=edges, custom=True, code=code,
                            int_constants=ints, str_constants=strs, float_constants=floats,
                            erc_range=_erc_range(spec, set(arg_types)), train_size=100, test_size=500, spec=spec)


def _erc_range(spec, types):
    """Range of the integer random constants, derived from the integer inputs."""
    if Int in types:
        lo, hi = spec.int_range
    elif Vec in types:
        lo, hi = spec.vec_range
    else:
        return (-10, 10)
    lo, hi = max(int(lo), -1000), min(int(hi), 1000)
    return (lo, hi) if hi - lo > 20 else (-10, 10)


# --------------------------------------------------------------------------
# Boolean logic problems
# --------------------------------------------------------------------------

# Logic gates, each switchable on its own: (config flag, block).
LOGIC_GATES = [
    ("logic_and", _strict("AND", (Bool, Bool), Bool, lambda a, b: a and b, "AND")),
    ("logic_or", _strict("OR", (Bool, Bool), Bool, lambda a, b: a or b, "OR")),
    ("logic_not", _strict("NOT", (Bool,), Bool, lambda a: not a, "NOT")),
    ("logic_nand", _strict("NAND", (Bool, Bool), Bool, lambda a, b: not (a and b), "NAND")),
    ("logic_nor", _strict("NOR", (Bool, Bool), Bool, lambda a, b: not (a or b), "NOR")),
    ("logic_xor", _strict("XOR", (Bool, Bool), Bool, lambda a, b: a != b, "XOR")),
    ("logic_xnor", _strict("XNOR", (Bool, Bool), Bool, lambda a, b: a == b, "XNOR")),
    ("logic_if", Block("IF", (Bool, Bool, Bool), Bool, "if_else", symbol="IF")),
]
GATE_FLAGS = [flag for flag, _ in LOGIC_GATES]
BLOCKS_BUILTIN = set(BLOCKS) | {gate.name for _, gate in LOGIC_GATES}   # custom functions must not reuse these
GATE_DESCRIPTIONS = {
    "logic_and": "AND(a, b): true if both inputs are true.",
    "logic_or": "OR(a, b): true if at least one input is true.",
    "logic_not": "NOT(a): negation.",
    "logic_nand": "NAND(a, b): NOT(AND(a, b)).",
    "logic_nor": "NOR(a, b): NOT(OR(a, b)).",
    "logic_xor": "XOR(a, b): true if exactly one input is true.",
    "logic_xnor": "XNOR(a, b): true if both inputs are equal.",
    "logic_if": "IF(c, a, b): a if c is true, otherwise b (a 2:1 multiplexer).",
}
BLOCKS.update({b.name: b for _, b in LOGIC_GATES})

_enabled_ps_blocks = enabled_blocks


def enabled_blocks(settings, kind=None):  # noqa: F811  (extends the program synthesis version)
    """Building blocks for a problem kind: Boolean problems use only the logic gates,
    regression and program synthesis use the program synthesis blocks (never the gates)."""
    if kind == "logic":
        return [b for flag, b in LOGIC_GATES if settings.get(flag)]
    if kind in ("sr", "ps"):
        return _enabled_ps_blocks(settings)
    return _enabled_ps_blocks(settings) + [b for flag, b in LOGIC_GATES if settings.get(flag)]


MAX_LOGIC_INPUTS = 8


def truth_table(arg_names, target):
    """All 2^n input combinations (first input = most significant bit) and outputs."""
    rows = []
    n = len(arg_names)
    for i in range(2 ** n):
        inputs = tuple(bool((i >> (n - 1 - k)) & 1) for k in range(n))
        rows.append((inputs, bool(target(*inputs))))
    return rows


@dataclass
class LogicProblem(SynthesisProblem):
    """Boolean function: Bool inputs, Bool output, the full truth table is the data."""
    kind: str = "logic"
    gates: tuple = ("logic_and", "logic_or", "logic_not")   # standard gate set of the problem
    target: object = None

    def describe_inputs(self, inputs, limit=70):
        text = "  ".join(f"{n}={int(v)}" for n, v in zip(self.arg_names, inputs))
        return text if len(text) <= limit else text[:limit - 1] + "…"

    def __post_init__(self):
        self.arg_types = (Bool,) * len(self.arg_names)
        self.ret = Bool
        table = truth_table(self.arg_names, self.target)
        self.train_size = self.test_size = len(table)
        self.generate = lambda train_size=0, test_size=0, seed=None: (table, table)


def _logic(name, arg_names, target, gates, label="Boolean"):
    return LogicProblem(name, tuple(arg_names), (), Bool, None, name=label, target=target, gates=tuple(gates))


KOZA_PARITY_GATES = ("logic_and", "logic_or", "logic_nand", "logic_nor")
KOZA_MUX_GATES = ("logic_and", "logic_or", "logic_not", "logic_if")

LOGIC = [
    _logic("Even-3-Parity", ("d0", "d1", "d2"), lambda *d: sum(d) % 2 == 0, KOZA_PARITY_GATES),
    _logic("Even-4-Parity", ("d0", "d1", "d2", "d3"), lambda *d: sum(d) % 2 == 0, KOZA_PARITY_GATES),
    _logic("Even-5-Parity", ("d0", "d1", "d2", "d3", "d4"), lambda *d: sum(d) % 2 == 0, KOZA_PARITY_GATES),
    _logic("Multiplexer-6", ("a0", "a1", "d0", "d1", "d2", "d3"),
           lambda a0, a1, *d: d[2 * a1 + a0], KOZA_MUX_GATES),
    _logic("Majority-5", ("x0", "x1", "x2", "x3", "x4"), lambda *x: sum(x) >= 3,
           ("logic_and", "logic_or", "logic_not")),
]

_LOGIC_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.BinOp, ast.BitAnd,
                ast.BitOr, ast.BitXor, ast.Compare, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant,
                ast.IfExp)


def parse_logic(text, name=""):
    """A Boolean function as a Python expression over variables, e.g. (a and not b) or c.
    Allowed: and, or, not, ^ (xor), &, |, ==, !=, x if c else y, True / False / 0 / 1."""
    text = (text or "").strip()
    if not text:
        raise ProgramError("Enter a Boolean expression, e.g. (a and not b) or c.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ProgramError("Syntax error. Example: (a and not b) or (c ^ d)") from None
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, _LOGIC_NODES):
            raise ProgramError(f"Not allowed: {type(node).__name__}. Use and, or, not, ^, ==, !=.")
        if isinstance(node, ast.Constant) and node.value not in (True, False, 0, 1):
            raise ProgramError("Only True, False, 0 and 1 are allowed as constants.")
        if isinstance(node, ast.Name):
            if node.id == ELEM or node.id in ("True", "False"):
                raise ProgramError(f"'{node.id}' cannot be used as a variable name.")
            names.add(node.id)
    if not names:
        raise ProgramError("The expression needs at least one variable.")
    if len(names) > MAX_LOGIC_INPUTS:
        raise ProgramError(f"At most {MAX_LOGIC_INPUTS} variables (2^{MAX_LOGIC_INPUTS} truth-table rows).")
    arg_names = tuple(sorted(names))
    code = compile(tree, "<logic>", "eval")

    def target(*values):
        return bool(eval(code, {"__builtins__": {}}, dict(zip(arg_names, values))))

    problem = LogicProblem(name or text, arg_names, (), Bool, None, name="Boolean" if name else "custom",
                           target=target, custom=True, code=text)
    problem.display_name = name
    return problem
