"""Deep-tree acceleration for symbolic regression (Additional Parameters → Deep-tree acceleration;
switched off, GPExplore evaluates as DEAP does), after Langdon & Banzhaf (2022), "Long-Term Evolution
Experiment with Genetic Programming", Artificial Life 28(2), and Langdon (2022), "Deep Genetic
Programming Trees Are Robust", ACM TELO 2(2):

* Linear evaluation on all test cases at once: a tree is a prefix list of opcodes run by one
  stack interpreter, every opcode on every training case (compiled with numba, otherwise numpy
  vectors) instead of one Python call per node and case.
* Incremental evaluation (children of at least INCREMENTAL_MIN_SIZE nodes): a child differs from its parent (the one whose root it keeps) in one
  subtree. Only that subtree is evaluated, then the nodes above it, one level after the other,
  until the values equal the parent's on every case. From there on the child computes what its
  parent computed (the functions have no side effects), so it gets the parent's outputs. Deep
  trees dissipate changes, so this point is usually reached after a few levels.

Both give exactly the outputs of the usual evaluation (with numba: of the numba evaluation;
without: of Python's), so the run is the same run. Trees with custom functions, and without numba
trees whose constant parts need Python's exact integer arithmetic, are evaluated the usual way.
"""

import math

import numpy as np

import fast

# Smaller children are evaluated whole (on all cases at once): for them, finding the changed
# subtree costs more than it saves (measured with and without numba on trees of 10 to 1,800 nodes).
INCREMENTAL_MIN_SIZE = 400
SAFE_INT = 2 ** 26      # integers up to this size compute exactly in floats (also squared)
NONCONST = object()     # marker: a subtree that depends on the inputs

UNARY = {fast.NEG, fast.COS, fast.SIN}


def same_node(a, b):
    if a is b:
        return True
    if type(a) is not type(b) or a.arity != b.arity or a.name != b.name:
        return False
    if a.arity:
        return True
    return type(a.value) is type(b.value) and a.value == b.value


def same_values(va, fa, vb, fb):
    """Bitwise the same values and the same failed rows (values of failed rows are 0)."""
    return np.array_equal(fa, fb) and np.array_equal(va.view(np.int64), vb.view(np.int64))


def outputs_of(values, failed):
    """As GPExplore's evaluation reports them: NaN for failed and non-finite results."""
    out = values.copy()
    out[failed | ~np.isfinite(values)] = np.nan
    return out


# ---- the numpy versions of fast.run_raw / fast.apply_raw (Python's float semantics) ----

def _np_apply(op, a, fa, b=None, fb=None):
    with np.errstate(all="ignore"):
        if op == fast.NEG:
            v, f = -a, fa.copy()
        elif op in (fast.COS, fast.SIN):
            f = fa | np.isinf(a)
            fn = math.cos if op == fast.COS else math.sin
            v = np.zeros_like(a)
            ok = ~f
            v[ok] = np.fromiter(map(fn, a[ok].tolist()), dtype=np.float64, count=int(ok.sum()))
        else:
            f = fa | fb
            if op == fast.ADD:
                v = a + b
            elif op == fast.SUB:
                v = a - b
            elif op == fast.MUL:
                v = a * b
            elif op == fast.PDIV:
                v = np.where(b != 0, a / np.where(b != 0, b, 1.0), 1.0)
            else:
                v = a / np.sqrt(1 + b * b)
    v = np.where(f, 0.0, v)
    return v, f


def _np_run(ops, vals, X):
    n_rows = X.shape[0]
    stack = []
    nofail = np.zeros(n_rows, dtype=bool)
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        if op == fast.ARG:
            stack.append((X[:, int(vals[i])].astype(np.float64), nofail))
        elif op == fast.CONST:
            stack.append((np.full(n_rows, vals[i]), nofail))
        elif op in UNARY:
            a, fa = stack.pop()
            stack.append(_np_apply(op, a, fa))
        else:
            a, fa = stack.pop()
            b, fb = stack.pop()
            stack.append(_np_apply(op, a, fa, b, fb))
    return stack[0]


class DeepEvaluator:
    """Outputs of symbolic regression trees on named data sets (e.g. "train", "val")."""

    def __init__(self, pset, datasets, use_numba):
        self.context = pset.context
        self.arg_index = {name: i for i, name in enumerate(pset.arguments)}
        self.X = {key: np.ascontiguousarray(X, dtype=np.float64) for key, X in datasets.items()}
        self.numba = bool(use_numba and fast.AVAILABLE)
        self.stats = {"full": 0, "same": 0, "synced": 0, "to_root": 0, "usual": 0}

    # ---- programs ----

    def program(self, nodes):
        """(ops, vals, constant value or NONCONST) of a tree or subtree, or None if it has to be
        evaluated the usual way."""
        prog = fast.tree_program(nodes, self.arg_index)
        if prog is None:
            return None
        if self.numba:
            return prog[0], prog[1], NONCONST
        const = self._python_constant(nodes)
        if const is None:
            return None
        return prog[0], prog[1], const

    def _python_constant(self, nodes):
        """Without numba the reference is Python, which computes constant parts with exact integers:
        floats give the same only while those integers stay small. Returns the subtree's constant
        value (NONCONST if it depends on inputs), or None if a constant part is too large."""
        stack = []
        for node in reversed(nodes):
            if not node.arity:
                value = node.value
                if isinstance(value, str):
                    stack.append(NONCONST)
                    continue
                if not self._safe(value):
                    return None
                stack.append(value)
                continue
            args = [stack.pop() for _ in range(node.arity)]
            if any(a is NONCONST for a in args):
                stack.append(NONCONST)
                continue
            value = self._python_value(node.name, args)
            if value is None:
                return None
            stack.append(value)
        return stack[0]

    def _python_value(self, name, args):
        try:
            value = self.context[name](*args)
        except Exception:
            return None
        return value if self._safe(value) else None

    @staticmethod
    def _safe(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return abs(value) <= SAFE_INT
        return isinstance(value, float)

    def run(self, prog, key):
        if self.numba:
            return fast.run_raw(prog[0], prog[1], self.X[key])
        return _np_run(prog[0], prog[1], self.X[key])

    def apply(self, op, args):
        if self.numba:
            if len(args) == 1:
                (a, fa), = args
                return fast.apply_raw(op, a, fa, a, fa)
            (a, fa), (b, fb) = args
            return fast.apply_raw(op, a, fa, b, fb)
        if len(args) == 1:
            (a, fa), = args
            return _np_apply(op, a, fa)
        (a, fa), (b, fb) = args
        return _np_apply(op, a, fa, b, fb)

    # ---- evaluation ----

    def outputs(self, individual, parent=None):
        """{data set: outputs} of an individual, or None if it has to be evaluated the usual way.
        With the parent it was made from (and the parent's outputs), incrementally."""
        if parent is not None and getattr(parent, "deep_out", None) is not None \
                and len(individual) >= INCREMENTAL_MIN_SIZE:
            result = self._incremental(individual, parent)
            if result is not None:
                return result
        prog = self.program(list(individual))
        if prog is None:
            self.stats["usual"] += 1
            return None
        self.stats["full"] += 1
        return {key: outputs_of(*self.run(prog, key)) for key in self.X}

    def _incremental(self, child, parent):
        m, c = list(parent), list(child)
        lm, lc = len(m), len(c)
        lim = min(lm, lc)
        p = 0
        while p < lim and same_node(m[p], c[p]):
            p += 1
        if p == lm == lc:
            self.stats["same"] += 1
            return dict(parent.deep_out)
        s = 0
        while s < lim - p and same_node(m[lm - 1 - s], c[lc - 1 - s]):
            s += 1
        diff_end = lm - s       # the parent's nodes [p, diff_end) differ from the child's
        delta = lc - lm

        # the smallest subtree of the parent that contains every difference: node p or an ancestor
        open_nodes = []         # ancestors of node p: [index, arity, child starts, children to come]
        for i in range(p):
            node = m[i]
            if open_nodes:
                open_nodes[-1][2].append(i)
            if node.arity:
                open_nodes.append([i, node.arity, [], node.arity])
                continue
            while open_nodes:   # a leaf completes its parent's child, maybe the parent, and so on
                open_nodes[-1][3] -= 1
                if open_nodes[-1][3]:
                    break
                open_nodes.pop()
        if open_nodes:
            open_nodes[-1][2].append(p)
        start, end = p, parent.searchSubtree(p).stop
        while end < diff_end:
            if not open_nodes:
                return None
            start = open_nodes.pop()[0]
            end = parent.searchSubtree(start).stop
        if child.searchSubtree(start).stop != end + delta:
            return None   # not a single replaced subtree
        # open_nodes now holds the ancestors of `start`; their last child start is on the path
        removed = self.program(m[start:end])
        inserted = self.program(c[start:end + delta])
        if removed is None or inserted is None:
            return None
        state = {}
        for key in self.X:
            vm, vc = self.run(removed, key), self.run(inserted, key)
            state[key] = None if same_values(*vm, *vc) else (vm, vc)
        const_m, const_c = removed[2], inserted[2]
        pos = end   # end of the path subtree in the parent
        for index, arity, starts, _ in reversed(open_nodes):
            if all(v is None for v in state.values()):
                break
            node = m[index]
            op = fast.PRIMITIVE_OPS.get(node.name)
            if op is None:
                return None
            # siblings: before the path child (their ends are the next starts), and after it
            path_at = len(starts) - 1
            sibling_slices = [(starts[k], starts[k + 1]) for k in range(path_at)]
            right = []
            for _ in range(arity - len(starts)):
                stop = parent.searchSubtree(pos).stop
                right.append((pos, stop))
                pos = stop
            siblings = []
            for a, b in sibling_slices + right:
                prog = self.program(m[a:b])
                if prog is None:
                    return None
                siblings.append(prog)
            if not self.numba:   # constant parts of the child or parent that grow past this level
                consts = [prog[2] for prog in siblings]
                for side in ("m", "c"):
                    own = const_m if side == "m" else const_c
                    args = consts[:path_at] + [own] + consts[path_at:]
                    value = NONCONST if any(a is NONCONST for a in args) else self._python_value(node.name, args)
                    if value is None:
                        return None
                    if side == "m":
                        const_m = value
                    else:
                        const_c = value
            for key, current in state.items():
                if current is None:
                    continue
                vm, vc = current
                values = [self.run(prog, key) for prog in siblings]
                new_m = self.apply(op, values[:path_at] + [vm] + values[path_at:])
                new_c = self.apply(op, values[:path_at] + [vc] + values[path_at:])
                state[key] = None if same_values(*new_m, *new_c) else (new_m, new_c)
        result = {}
        for key, current in state.items():
            result[key] = parent.deep_out[key] if current is None else outputs_of(*current[1])
        self.stats["to_root" if any(v is not None for v in state.values()) else "synced"] += 1
        return result
