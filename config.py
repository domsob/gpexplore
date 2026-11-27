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
# Double tournament (Luke & Panait 2002): the smaller of two individuals wins the size
# tournament with probability parsimony_size / 2 (1 = no size pressure, 2 = always).
parsimony_size = 1.4
double_tournament_order = "Fitness first"   # or "Size first"
# ε-lexicase with a fixed ε (DEAP selEpsilonLexicase), on the error per case
lexicase_epsilon = 0.1
# Down-sampled / informed down-sampled ε-lexicase. Defaults follow
# Geiger et al. (2023, 2024) and Boldi et al. (2024): d = 0.1, s = 0.01, k = 10.
ds_rate = 0.1
# Down-sampling for the selection methods without one of their own (tournament, lexicase, …):
# "None" | "Random" | "Informed"
downsampling = "None"
parent_sampling_rate = 0.01
distance_interval = 10
# ROIDS (Geiger et al. 2026): share γ of cases excluded as outliers, default 0.05.
outlier_rate = 0.05
initialization = "Ramped Half-n-Half"
min_initial_tree_depth = 2
max_initial_tree_depth = 5
max_tree_depth = 17

# Advanced GP parameters (Additional Parameters tab, "Advanced")
crossover_type = "One-point"            # "One-point" | "Leaf-biased"
inner_node_probability = 0.9            # leaf-biased crossover: crossover point is an inner node (Koza 1992)
mutation_type = "Uniform (subtree)"     # see gp.MUTATIONS
subtree_min_depth = 0                   # new random subtrees (uniform mutation): depth range ...
subtree_max_depth = 2
subtree_method = "Full"                 # ... and method: "Full" | "Grow" | "Ramped Half-n-Half"
ephemeral_mode = "One constant"         # ephemeral mutation: "One constant" | "All constants"
bloat_control = "Depth limit"           # "Depth limit" | "Depth + size limit" | "Lexicographic parsimony"
max_tree_size = 100                     # nodes (with "Depth + size limit")
elitism = 0                             # best individuals copied unchanged into the next generation
fitness_function = "MSE"                # symbolic regression: "MSE" | "MAE" | "RMSE" | "MSE + linear scaling"
erc_lo, erc_hi = -1, 1                  # symbolic regression: integer random constants
seed = 0                                # 0 = random; otherwise runs are reproducible
parameter_control = "None"              # change parameters during a run (operators.BUILTIN["control"] or custom)

# Experiments tab (repeated runs)
exp_runs = 30                           # runs per configuration
exp_workers = 0                         # parallel runs; 0 = number of CPU cores - 1
exp_seed = 1                            # run r uses seed exp_seed + r

# App behaviour
plot_adjust = True
inter_gen_pause = 0.5
show_advanced = False                   # Additional Parameters: advanced settings shown
expert_mode = False                     # number fields instead of sliders
# Compiled evaluation of regression trees and lexicase selection (fast.py); no effect
# without numba. Same results either way, only faster.
numba_acceleration = True
# Symbolic regression: evaluation on all cases at once and incremental evaluation of children
# (deep.py, after Langdon & Banzhaf 2022); off: as in DEAP. Same results either way, only faster.
deep_acceleration = True
# Desktop window size of the last session (logical pixels); not touched by "Reset all settings"
window_width = 1490
window_height = 1025
WINDOW_KEYS = ("window_width", "window_height")

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
    "selection_method", "tourn_size", "parsimony_size", "double_tournament_order", "lexicase_epsilon", "downsampling", "ds_rate", "parent_sampling_rate", "distance_interval", "outlier_rate", "initialization", "min_initial_tree_depth",
    "max_initial_tree_depth", "max_tree_depth", "crossover_type", "inner_node_probability", "mutation_type",
    "subtree_min_depth", "subtree_max_depth", "subtree_method", "ephemeral_mode", "bloat_control",
    "max_tree_size", "elitism", "fitness_function", "erc_lo", "erc_hi", "seed", "numba_acceleration", "deep_acceleration", "parameter_control", "training_X", "training_y", "validation_X", "validation_y",
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
    settings["custom_library"] = operators.library()   # custom operators an adaptive control may switch to
    return settings


# --------------------------------------------------------------------------
# Settings kept across sessions (settings.json next to this file)
# --------------------------------------------------------------------------

SETTINGS_FILE = os.path.join(_here, "settings.json")
SAVED_KEYS = [k for k in RUN_SETTING_KEYS if k not in ("training_X", "training_y", "validation_X", "validation_y")] + [
    "training_set_size", "noise_intensity", "validation_set_size", "outlier_share", "exp_runs", "exp_workers",
    "exp_seed", "plot_adjust", "inter_gen_pause", "show_advanced", "expert_mode", *WINDOW_KEYS]
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
        if key == "crossover_type" and value == "Leaf-biased (90/10)":   # renamed in 0.2
            value = "Leaf-biased"
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
        import operators
        if ok and key in operators.OPTIONS:
            ok = value in operators.OPTIONS[key]
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
        if key not in WINDOW_KEYS:   # the window keeps its size
            globals()[key] = value
