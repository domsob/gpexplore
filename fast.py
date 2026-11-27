"""Optional acceleration with numba (Additional Parameters → Numba acceleration).

Two hot spots are compiled to machine code; everything else stays plain Python:

* Evaluating symbolic regression trees. Without numba every tree is compiled to a Python
  lambda and called once per training case. Here a tree becomes a small program (prefix
  order, one opcode per node) that a single compiled interpreter runs on all cases at once.
  Only the built-in functions are known to it; a tree with custom functions is evaluated
  the usual way.
* The case loop of (ε-)lexicase selection (filtering the candidates case by case).

Both give the same numbers as the Python code, so a run with the same seed is the same
run with or without numba. Without numba installed, `AVAILABLE` is False and GPExplore
uses the Python code.
"""

import math

import numpy as np

try:
    import numba
    AVAILABLE = True
except Exception:   # not installed, or not built for this Python / numpy
    numba = None
    AVAILABLE = False


def _njit(fn):
    return numba.njit(cache=True, nogil=True)(fn) if AVAILABLE else fn


# opcodes of the tree programs
ARG, CONST, ADD, SUB, MUL, PDIV, AQ, NEG, COS, SIN = range(10)
PRIMITIVE_OPS = {"add": ADD, "sub": SUB, "mul": MUL, "protectedDiv": PDIV, "analyticQuotient": AQ,
                 "neg": NEG, "cos": COS, "sin": SIN}


def tree_program(individual, arg_index):
    """(opcodes, values) of a symbolic regression tree, or None if it uses a function the
    interpreter does not know (e.g. a custom one). `arg_index` maps argument names to columns."""
    n = len(individual)
    ops = np.empty(n, dtype=np.int64)
    vals = np.zeros(n, dtype=np.float64)
    for i, node in enumerate(individual):
        if node.arity:   # a primitive
            op = PRIMITIVE_OPS.get(node.name)
            if op is None:
                return None
            ops[i] = op
            continue
        value = node.value
        if isinstance(value, str):   # an argument
            col = arg_index.get(value)
            if col is None:
                return None
            ops[i], vals[i] = ARG, col
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, int) and abs(value) > 2 ** 53:   # not exact as a float
                return None
            ops[i], vals[i] = CONST, value
        else:
            return None
    return ops, vals


@_njit
def run_raw(ops, vals, X):
    """Values of a tree program (also of a single subtree) on every row of X, and per row
    whether Python would have raised an error (math.sin / math.cos of an infinite value).
    The values are not cleaned (inf stays inf); a failed row's value is meaningless."""
    n_rows = X.shape[0]
    n = ops.shape[0]
    out = np.empty(n_rows)
    failed = np.zeros(n_rows, dtype=np.bool_)
    stack = np.empty(n)
    for r in range(n_rows):
        sp = 0
        for i in range(n - 1, -1, -1):
            op = ops[i]
            if op == ARG:
                stack[sp] = X[r, int(vals[i])]
                sp += 1
            elif op == CONST:
                stack[sp] = vals[i]
                sp += 1
            elif op == NEG:
                stack[sp - 1] = -stack[sp - 1]
            elif op == COS or op == SIN:
                a = stack[sp - 1]
                if math.isinf(a):
                    failed[r] = True
                    break
                stack[sp - 1] = math.cos(a) if op == COS else math.sin(a)
            else:
                a = stack[sp - 1]
                b = stack[sp - 2]
                sp -= 1
                if op == ADD:
                    v = a + b
                elif op == SUB:
                    v = a - b
                elif op == MUL:
                    v = a * b
                elif op == PDIV:
                    v = a / b if b != 0 else 1.0
                else:
                    v = a / math.sqrt(1 + b * b)
                stack[sp - 1] = v
        out[r] = 0.0 if failed[r] else stack[0]
    return out, failed


@_njit
def apply_raw(op, a, fa, b, fb):
    """One function node on value vectors (b unused for unary functions), as in run_raw."""
    n = a.shape[0]
    out = np.empty(n)
    failed = np.empty(n, dtype=np.bool_)
    for r in range(n):
        f = fa[r] or (fb[r] if op != NEG and op != COS and op != SIN else False)
        x = a[r]
        v = 0.0
        if not f:
            if op == NEG:
                v = -x
            elif op == COS or op == SIN:
                if math.isinf(x):
                    f = True
                else:
                    v = math.cos(x) if op == COS else math.sin(x)
            else:
                y = b[r]
                if op == ADD:
                    v = x + y
                elif op == SUB:
                    v = x - y
                elif op == MUL:
                    v = x * y
                elif op == PDIV:
                    v = x / y if y != 0 else 1.0
                else:
                    v = x / math.sqrt(1 + y * y)
        out[r] = 0.0 if f else v
        failed[r] = f
    return out, failed


@_njit
def run_program(ops, vals, X):
    """Outputs of a tree program on every row of X, as GPExplore's Python evaluation gives
    them: NaN where the result is not finite or where Python would have raised an error
    (math.sin / math.cos of an infinite value)."""
    n_rows = X.shape[0]
    n = ops.shape[0]
    out = np.empty(n_rows)
    stack = np.empty(n)
    for r in range(n_rows):
        sp = 0
        failed = False
        for i in range(n - 1, -1, -1):   # prefix order backwards: arguments come before their function
            op = ops[i]
            if op == ARG:
                stack[sp] = X[r, int(vals[i])]
                sp += 1
            elif op == CONST:
                stack[sp] = vals[i]
                sp += 1
            elif op == NEG:
                stack[sp - 1] = -stack[sp - 1]
            elif op == COS or op == SIN:
                a = stack[sp - 1]
                if math.isinf(a):   # Python's math functions raise ValueError here
                    failed = True
                    break
                stack[sp - 1] = math.cos(a) if op == COS else math.sin(a)
            else:
                a = stack[sp - 1]   # left argument
                b = stack[sp - 2]   # right argument
                sp -= 1
                if op == ADD:
                    v = a + b
                elif op == SUB:
                    v = a - b
                elif op == MUL:
                    v = a * b
                elif op == PDIV:
                    v = a / b if b != 0 else 1.0
                else:   # AQ
                    v = a / math.sqrt(1 + b * b)
                stack[sp - 1] = v
        if failed:
            out[r] = np.nan
        else:
            v = stack[0]
            out[r] = v if math.isfinite(v) else np.nan
    return out


@_njit
def lexicase_survivors(errors, candidates, order, mode, fixed):
    """The candidates left after filtering on the cases in `order` (as in gp.epsilon_lexicase).
    mode 0: lexicase (only the elite on a case), 1: ε-lexicase with ε = median absolute
    deviation, 2: ε-lexicase with the fixed ε."""
    for case in order:
        column = np.empty(candidates.shape[0])
        for j in range(candidates.shape[0]):
            column[j] = errors[candidates[j], case]
        best = column.min()
        if mode == 1:
            eps = np.median(np.abs(column - np.median(column)))
        elif mode == 2:
            eps = fixed
        else:
            eps = 0.0
        if mode == 0:
            candidates = candidates[column == best]
        else:
            candidates = candidates[column <= best + eps]
        if candidates.shape[0] == 1:
            break
    return candidates
