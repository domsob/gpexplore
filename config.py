"""Default settings of GPExplore. The UI changes these values at runtime."""

import os

APP_NAME = "GPExplore"
APP_VERSION = "0.2"

# Problem
target_function = "x**4 + x**3 + x**2 + x"
training_set_size = 50
noise_intensity = 0.0
validation_set_size = 200
outlier_share = 0.0   # share of training points turned into outliers (for ROIDS)

# Evolution
generations = 50
population_size = 250
crossover_probability = 0.8
mutation_probability = 0.05

# Function & terminal set
add_func = True
sub_func = True
mul_func = True
protectedDiv_func = False
analyticQuotient_func = True
neg_func = True
sin_func = True
cos_func = True
int101_erc = True

# Program synthesis building blocks (synthesis.GROUPS); off by default and
# switched on automatically when a program synthesis problem is selected.
ps_if = False
ps_if_else = False
ps_compare = False
ps_bool = False
ps_int = False
ps_for_vec = False
ps_for_range = False
ps_str = False
ps_vec = False

# Logic gates for Boolean problems (synthesis.LOGIC_GATES); off by default and set to
# the problem's standard gate set when a Boolean problem is selected.
logic_and = False
logic_or = False
logic_not = False
logic_nand = False
logic_nor = False
logic_xor = False
logic_xnor = False
logic_if = False

# Additional GP parameters
selection_method = "Tournament"
tourn_size = 3
# Down-sampled / informed down-sampled ε-lexicase. Defaults follow
# Geiger et al. (2023, 2024) and Boldi et al. (2024): d = 0.1, s = 0.01, k = 10.
ds_rate = 0.1
parent_sampling_rate = 0.01
distance_interval = 10
# ROIDS (Geiger et al. 2026): share γ of cases excluded as outliers, default 0.05.
outlier_rate = 0.05
initialization = "Ramped Half-n-Half"
min_initial_tree_depth = 2
max_initial_tree_depth = 5
max_tree_depth = 17

# Advanced GP parameters (Additional Parameters tab, "Advanced")
crossover_type = "One-point"            # "One-point" | "Leaf-biased (90/10)"
mutation_type = "Uniform (subtree)"     # see gp.MUTATIONS
bloat_control = "Depth limit"           # "Depth limit" | "Depth + size limit" | "Lexicographic parsimony"
max_tree_size = 100                     # nodes (with "Depth + size limit")
elitism = 0                             # best individuals copied unchanged into the next generation
fitness_function = "MSE"                # symbolic regression: "MSE" | "MAE" | "RMSE" | "MSE + linear scaling"
erc_lo, erc_hi = -1, 1                  # symbolic regression: integer random constants
seed = 0                                # 0 = random; otherwise runs are reproducible

# Experiments tab (repeated runs)
exp_runs = 30                           # runs per configuration
exp_workers = 0                         # parallel runs; 0 = number of CPU cores - 1
exp_seed = 1                            # run r uses seed exp_seed + r

# App behaviour
plot_adjust = True
inter_gen_pause = 0.5
show_advanced = False                   # Additional Parameters: advanced settings shown
expert_mode = False                     # number fields instead of sliders

# Data sets (filled by problems.make_training_set / make_validation_set)
training_X = []   # list of input tuples: (x,) or (x, y)
training_y = []
validation_X = []
validation_y = []

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "LICENSE.txt"), encoding="utf-8") as _f:
    license_text = _f.read()

RUN_SETTING_KEYS = [
    "target_function", "generations", "population_size", "crossover_probability",
    "mutation_probability", "add_func", "sub_func", "mul_func", "protectedDiv_func",
    "analyticQuotient_func", "neg_func", "sin_func", "cos_func", "int101_erc",
    "selection_method", "tourn_size", "ds_rate", "parent_sampling_rate", "distance_interval", "outlier_rate", "initialization", "min_initial_tree_depth",
    "max_initial_tree_depth", "max_tree_depth", "crossover_type", "mutation_type", "bloat_control",
    "max_tree_size", "elitism", "fitness_function", "erc_lo", "erc_hi", "seed", "training_X", "training_y", "validation_X", "validation_y",
    "ps_if", "ps_if_else", "ps_compare", "ps_bool", "ps_int", "ps_for_vec", "ps_for_range", "ps_str", "ps_vec",
    "logic_and", "logic_or", "logic_not", "logic_nand", "logic_nor", "logic_xor", "logic_xnor", "logic_if",
]


def run_settings():
    """Freeze the current settings for a GP run."""
    import problems
    settings = {key: globals()[key] for key in RUN_SETTING_KEYS}
    settings["problem"] = problems.get(target_function)
    import operators
    settings["custom_operators"] = operators.selected(settings)   # code of the chosen custom operators
    settings["custom_functions"] = operators.selected_functions()  # code of the switched-on custom functions
    return settings


# --------------------------------------------------------------------------
# Settings kept across sessions (settings.json next to this file)
# --------------------------------------------------------------------------

SETTINGS_FILE = os.path.join(_here, "settings.json")
SAVED_KEYS = [k for k in RUN_SETTING_KEYS if k not in ("training_X", "training_y", "validation_X", "validation_y")] + [
    "training_set_size", "noise_intensity", "validation_set_size", "outlier_share", "exp_runs", "exp_workers",
    "exp_seed", "plot_adjust", "inter_gen_pause", "show_advanced", "expert_mode"]
DEFAULTS = {key: globals()[key] for key in SAVED_KEYS}   # before anything is loaded


def snapshot():
    return {key: globals()[key] for key in SAVED_KEYS}


def load_settings():
    """Values of the last session; anything unknown or of the wrong type keeps its default."""
    import json
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if key not in DEFAULTS:
            continue
        default = DEFAULTS[key]
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int):
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            value = float(value) if ok else value
        else:
            ok = isinstance(value, type(default))
        if ok:
            globals()[key] = value


def save_settings():
    import json
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshot(), f, indent=2)
    except OSError:
        pass   # read-only folder: the settings are simply not kept


def reset_settings():
    for key, value in DEFAULTS.items():
        globals()[key] = value
