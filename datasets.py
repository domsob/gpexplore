"""Load your own data sets as benchmark problems.

`detect(text)` guesses how a text file is structured (delimiter, header, column
types, problem kind, target column); the dialog shows the guess and every part of
it can be changed. `build(...)` turns the (edited) settings into a problem for
symbolic regression, program synthesis or Boolean logic.
"""

import csv
import io
import json
import keyword
import math
import random
import re
import shlex
from dataclasses import dataclass, field

import numpy as np

import synthesis
from synthesis import Bool, Int, Num, Str, Vec

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_ROWS = 2000            # larger data sets are subsampled (GP evaluates every case in Python)
MAX_LOGIC_INPUTS = synthesis.MAX_LOGIC_INPUTS

DELIMITERS = {"comma": ",", "semicolon": ";", "tab": "\t", "whitespace": None, "pipe": "|"}
MISSING = {"", "?", "na", "n/a", "nan", "null", "none", "-"}
TRUE = {"1", "true", "t", "yes", "y"}
FALSE = {"0", "false", "f", "no", "n"}
TYPES = {"bool": Bool, "int": Int, "float": Num, "str": Str, "list[int]": Vec}
KINDS = ("sr", "ps", "logic")


class DatasetError(ValueError):
    pass


# --------------------------------------------------------------------------
# Reading and detection
# --------------------------------------------------------------------------

def read_file(path):
    with open(path, "rb") as f:
        raw = f.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise DatasetError("The file is larger than 20 MB.")
    if b"\x00" in raw[:4096]:
        raise DatasetError("This does not look like a text file (CSV, TSV, DAT, …).")
    return raw.decode("utf-8-sig", errors="replace")


def _lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def split_rows(text, delimiter):
    """Rows of string fields. `delimiter` is a key of DELIMITERS."""
    lines = _lines(text)
    sep = DELIMITERS[delimiter]
    if sep is None:
        rows = []
        for line in lines:
            try:
                rows.append(shlex.split(line))          # respects "quoted names with spaces"
            except ValueError:
                rows.append(line.split())
        return rows
    return [[f.strip() for f in row] for row in csv.reader(lines, delimiter=sep, skipinitialspace=True)]


def detect_delimiter(text):
    sample = "\n".join(_lines(text)[:60])
    best, best_score = "whitespace", 0
    for name in ("tab", "semicolon", "comma", "pipe", "whitespace"):
        rows = split_rows(sample, name)
        if not rows:
            continue
        counts = [len(r) for r in rows]
        common = max(set(counts), key=counts.count)
        consistent = counts.count(common) / len(counts)
        if common > 1 and consistent >= 0.9:
            score = common * consistent + (0.5 if name != "whitespace" else 0)
            if score > best_score:
                best, best_score = name, score
    return best


def _is_missing(v):
    return v.strip().lower() in MISSING


def _as_int(v):
    v = v.strip()
    if re.fullmatch(r"[+-]?\d+", v):
        return int(v)
    raise ValueError(v)


def _as_list(v):
    v = v.strip()
    if not (v.startswith("[") and v.endswith("]")):
        raise ValueError(v)
    items = json.loads(v)
    if not isinstance(items, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in items):
        raise ValueError(v)
    return tuple(items)


def _as_bool(v):
    v = v.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(v)


PARSERS = {"bool": _as_bool, "int": _as_int, "float": lambda v: float(v), "list[int]": _as_list,
           "str": lambda v: v}


def column_type(values):
    """Most specific type that parses every (non-missing) value."""
    vals = [v for v in values if not _is_missing(v)]
    if not vals:
        return "str"
    for name in ("int", "float", "list[int]"):
        try:
            for v in vals:
                PARSERS[name](v)
            return name
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    if all(v.strip().lower() in TRUE | FALSE for v in vals):
        return "bool"
    return "str"


def bool_like(values):
    vals = [v.strip().lower() for v in values if not _is_missing(v)]
    return bool(vals) and all(v in TRUE | FALSE for v in vals)


def identifier(name, used):
    """Column name -> GP argument name (valid, unique, not clashing with primitives)."""
    base = re.sub(r"\W+", "_", name.strip().lower()).strip("_") or "col"
    if base[0].isdigit():
        base = "c_" + base
    reserved = set(keyword.kwlist) | set(synthesis.BLOCKS) | {"e", "x", "y", "add", "sub", "mul", "neg", "sin",
                                                              "cos", "protectedDiv", "analyticQuotient", "True",
                                                              "False", "erc"}
    if base in reserved:
        base += "_"
    out, k = base, 2
    while out in used:
        out, k = f"{base}{k}", k + 1
    used.add(out)
    return out


@dataclass
class Detection:
    delimiter: str
    header: bool
    rows: list                 # string rows without the header
    names: list                # column names (from the header or generated)
    types: list                # detected type per column
    bool_cols: list            # columns whose values are all 0/1, true/false, …
    id_cols: list              # string columns that look like names / ids (mostly unique)
    kind: str                  # suggested problem kind
    target: int                # suggested target column
    inputs: list               # suggested input columns
    raw_rows: list = field(default_factory=list)   # all rows including a possible header


def analyse(text, delimiter=None, header=None):
    """Detect the structure; `delimiter` / `header` override the guesses."""
    delimiter = delimiter or detect_delimiter(text)
    all_rows = split_rows(text, delimiter)
    if not all_rows:
        raise DatasetError("The file is empty.")
    width = max(set(len(r) for r in all_rows), key=[len(r) for r in all_rows].count)
    all_rows = [r for r in all_rows if len(r) == width]
    if width < 2:
        raise DatasetError("At least two columns are needed (inputs and a target).")
    if header is None:
        header = _guess_header(all_rows)
    rows = all_rows[1:] if header else all_rows
    if not rows:
        raise DatasetError("No data rows.")
    cols = list(zip(*rows))
    types = [column_type(c) for c in cols]
    bools = [i for i, c in enumerate(cols) if bool_like(c)]
    head = [h.strip().lower() for h in all_rows[0]] if header else [""] * width

    def id_like(i):
        """Mostly unique text that looks like names or ids (e.g. car names in Auto MPG);
        plain unique words can be real program inputs, so they are kept."""
        c = cols[i]
        if types[i] != "str" or len(set(c)) <= 0.5 * len(c):
            return False
        named = re.search(r"(^|_| )(name|id|key|label)s?($|_| )", head[i]) is not None
        return named or sum(" " in v.strip() for v in c) > 0.5 * len(c)

    ids = [i for i in range(width) if id_like(i)]
    if width - len(ids) < 2:   # never drop so much that no input is left
        ids = []
    usable = [i for i in range(width) if i not in ids]
    if len(usable) >= 2 and all(i in bools for i in usable) and len(usable) - 1 <= MAX_LOGIC_INPUTS:
        kind = "logic"
    elif all(types[i] in ("int", "float") for i in usable):
        kind = "sr"
    else:
        kind = "ps"
    numeric = [i for i in usable if types[i] in ("int", "float")]
    target = usable[-1] if kind != "sr" or not numeric else numeric[-1]
    inputs = [i for i in usable if i != target]
    if header:
        names = [h.strip() or f"col{i + 1}" for i, h in enumerate(all_rows[0])]
    else:   # generated names: inputs x1, x2, …, the suggested target y, ids name1, …
        names, k, m = [], 1, 1
        for i in range(width):
            if i == target:
                names.append("y")
            elif i in ids:
                names.append(f"name{m}")
                m += 1
            else:
                names.append(f"x{k}")
                k += 1
    return Detection(delimiter, header, rows, names, types, bools, ids, kind, target, inputs, all_rows)


def _guess_header(rows):
    """Header if a value of the first row does not fit the type of its column, or (text
    data) if the first row looks like column names that never occur below it."""
    if len(rows) < 2:
        return False
    first, rest = rows[0], rows[1:201]
    rest_types = [column_type([r[j] for r in rest]) for j in range(len(first))]
    for j, value in enumerate(first):
        if rest_types[j] != "str" and not _is_missing(value) and column_type([value]) == "str":
            return True
    looks_like_names = all(re.fullmatch(r"[A-Za-z_][\w .()\-/]*", v.strip()) for v in first)
    repeated = any(first[j] in {r[j] for r in rest} for j in range(len(first)))
    return looks_like_names and not repeated and len(rest) >= 3


# --------------------------------------------------------------------------
# Building problems
# --------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Everything needed to rebuild a data set problem (stored in custom_problems.json)."""
    name: str
    kind: str
    names: list
    types: list
    target: int
    inputs: list
    rows: list
    test_share: float = 0.3
    standardize: bool = True
    seed: int = 0
    source: str = ""

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class DatasetProblem:
    """Symbolic regression on a data set with any number of inputs."""
    expression: str
    arg_names: tuple
    train: tuple
    test: tuple
    spec: object
    name: str = "dataset"
    custom: bool = True
    kind: str = "sr"
    dataset: bool = True
    limits: tuple = None
    info: str = ""

    @property
    def inputs(self):
        return len(self.arg_names)

    @property
    def train_size(self):
        return len(self.train[1])

    @property
    def test_size(self):
        return len(self.test[1])

    @property
    def label(self):
        return self.expression

    def describe_inputs(self, inputs, limit=70):
        text = ", ".join(f"{n}={v:g}" for n, v in zip(self.arg_names, inputs))
        return text if len(text) <= limit else text[:limit - 1] + "…"


def build(spec: DatasetSpec):
    """Problem from a data set spec. Rows with missing / unparseable values in the used
    columns are skipped; large data sets are subsampled to MAX_ROWS rows."""
    if spec.kind not in KINDS:
        raise DatasetError("Unknown problem type.")
    if spec.target in spec.inputs:
        raise DatasetError("The target column cannot also be an input.")
    if not spec.inputs:
        raise DatasetError("Select at least one input column.")
    used = list(spec.inputs) + [spec.target]
    types = list(spec.types)
    if spec.kind == "sr":
        bad = [spec.names[i] for i in used if types[i] not in ("int", "float", "bool")]
        if bad:
            raise DatasetError(f"Symbolic regression needs numeric columns; not numeric: {', '.join(bad)}.")
        types = [("float" if i in used else t) for i, t in enumerate(types)]
    if spec.kind == "logic":
        if len(spec.inputs) > MAX_LOGIC_INPUTS:
            raise DatasetError(f"Boolean problems can have at most {MAX_LOGIC_INPUTS} inputs.")
        types = [("bool" if i in used else t) for i, t in enumerate(types)]
    if spec.kind == "ps" and types[spec.target] == "list[int]":
        raise DatasetError("A list[int] column cannot be the output of a program.")

    def parse(value, t):
        if spec.kind == "logic":
            return _as_bool(value)
        if spec.kind == "sr":
            try:
                return float(value)
            except ValueError:
                return 1.0 if _as_bool(value) else 0.0
        if _is_missing(value) and t != "str":
            raise ValueError("missing")
        return PARSERS[t](value)

    parsed, skipped = [], 0
    for row in spec.rows:
        try:
            values = {i: parse(row[i], types[i]) for i in used}
            if spec.kind == "sr" and not all(math.isfinite(v) for v in values.values()):
                raise ValueError("not finite")
            parsed.append(values)
        except (ValueError, TypeError, IndexError, json.JSONDecodeError):
            skipped += 1
    if len(parsed) < 4:
        raise DatasetError("Fewer than 4 usable rows with these settings.")
    rng = random.Random(spec.seed)
    sampled = False
    if len(parsed) > MAX_ROWS:
        parsed = rng.sample(parsed, MAX_ROWS)
        sampled = True

    names = set()
    arg_names = tuple(identifier(spec.names[i], names) for i in spec.inputs)
    rows = [(tuple(r[i] for i in spec.inputs), r[spec.target]) for r in parsed]
    notes = []
    if skipped:
        notes.append(f"{skipped} rows with missing values skipped")
    if sampled:
        notes.append(f"subsampled to {MAX_ROWS} rows")

    if spec.kind == "logic":
        problem = _logic_problem(spec, arg_names, rows)
    else:
        rng.shuffle(rows)
        n_test = int(round(spec.test_share * len(rows)))
        n_test = min(max(n_test, 1 if spec.test_share > 0 else 0), len(rows) - 2)
        test, train = rows[:n_test], rows[n_test:]
        if not test:
            test = train
        if spec.kind == "sr":
            problem = _sr_problem(spec, arg_names, train, test)
        else:
            problem = _ps_problem(spec, arg_names, types, train, test)
    problem.info = ", ".join(notes)
    return problem


def _sr_problem(spec, arg_names, train, test):
    X = np.array([x for x, _ in train], dtype=float)
    y = np.array([t for _, t in train], dtype=float)
    Xt = np.array([x for x, _ in test], dtype=float)
    yt = np.array([t for _, t in test], dtype=float)
    if spec.standardize:   # z-scores with the training statistics
        mx, sx = X.mean(axis=0), X.std(axis=0)
        sx[sx == 0] = 1.0
        my, sy = y.mean(), (y.std() or 1.0)
        X, Xt = (X - mx) / sx, (Xt - mx) / sx
        y, yt = (y - my) / sy, (yt - my) / sy
    as_rows = lambda A, b: ([tuple(r) for r in A.tolist()], b.tolist())
    return DatasetProblem(spec.name, arg_names, as_rows(X, y), as_rows(Xt, yt), spec)


def _ps_problem(spec, arg_names, types, train, test):
    arg_types = tuple(TYPES[types[i]] for i in spec.inputs)
    ret = TYPES[types[spec.target]]
    outputs = [o for _, o in train]
    str_constants = tuple(sorted(set(outputs)))[:10] if ret is Str and len(set(outputs)) <= 10 else ()
    ints = [v for x, _ in train for v, t in zip(x, arg_types) if t is Int]
    ints += [v for x, _ in train for v, t in zip(x, arg_types) if t is Vec for v in v]
    if ints:
        lo, hi = max(min(ints), -1000), min(max(ints), 1000)
        erc = (lo, hi) if hi - lo > 20 else (-10, 10)
    else:
        erc = (-10, 10)
    problem = synthesis.SynthesisProblem(
        spec.name, arg_names, arg_types, ret, lambda *a, **k: (train, test), name="dataset", custom=True,
        str_constants=str_constants, erc_range=erc, train_size=len(train), test_size=len(test))
    problem.dataset = True
    problem.spec = spec
    return problem


class DatasetLogicProblem(synthesis.LogicProblem):
    """Boolean problem from rows of a (possibly incomplete) truth table."""

    def __post_init__(self):
        self.arg_types = (Bool,) * len(self.arg_names)
        self.ret = Bool
        table = self.target            # the rows themselves
        self.train_size = self.test_size = len(table)
        self.generate = lambda train_size=0, test_size=0, seed=None: (table, table)
        n = len(self.arg_names)
        # truth-table index of every row (first input = most significant bit)
        self.cell_index = [sum(int(b) << (n - 1 - k) for k, b in enumerate(x)) for x, _ in table]


def _logic_problem(spec, arg_names, rows):
    # identical input rows: keep the first (a truth table has one output per input)
    seen, unique = set(), []
    for x, o in rows:
        if x not in seen:
            seen.add(x)
            unique.append((tuple(bool(v) for v in x), bool(o)))
    n = len(arg_names)
    unique.sort(key=lambda r: sum(int(b) << (n - 1 - k) for k, b in enumerate(r[0])))
    problem = DatasetLogicProblem(spec.name, arg_names, (), Bool, None, name="dataset", target=unique,
                                  custom=True, code="")
    problem.dataset = True
    problem.spec = spec
    problem.display_name = spec.name
    return problem
