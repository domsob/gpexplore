# Custom code in GPExplore

GPExplore can be extended with small Python functions written directly in the app. This page lists every kind, what its inputs mean, what it must return, and when GPExplore calls it.

| Kind | Where in the app | Saved in |
|---|---|---|
| [Custom problems](#custom-problems) | Run tab → *Custom problem* | `custom_problems.json` |
| [Custom functions](#custom-functions-function-sets) | Function & Terminal Set → *Add / edit custom functions* | `custom_functions.json` |
| [Custom operators](#custom-operators) (fitness, selection, crossover, mutation, initialization, bloat control, adaptive control) | Additional Parameters → *Add / edit custom operators* | `custom_operators.json` |

**Common to all kinds**

- **Editing:** every editor starts with a working template; adapt it instead of starting from scratch.
- **Checking:** code is checked while you type (syntax, number of parameters, types) and tested before it is saved. Operators run a short GP run; functions are called with typical inputs.
- **Editing and deleting later:** the pencil next to a saved item opens it again, and the bin deletes it (with undo).
- **Name:** the function's own name (`def my_selection(...)`) is the name under which it appears in the app. A name cannot be taken twice.
- **Modules:** `math` and `random` can be used without importing them.
- **Experiments and scripts:** the code of the items selected for a run is stored with the run settings, so experiments (in parallel processes) and the Python API (`import gpexplore`) use exactly that code.
- **Safety:** the code runs unrestricted inside GPExplore. An endless loop makes the app hang, and an error stops the run with a message naming the function.

---

## Custom problems

### Formula `f(x)` or `f(x, y)`

A symbolic regression target, e.g. `x**3 - 2*x` or `sin(x) * cos(y)`, plus the input range. Training and validation points are drawn at random from that range; noise and outliers can be added in the Run tab.

### Python program (program synthesis)

One annotated function; its outputs on random inputs become the training and test cases:

```python
def last_digit(n: int) -> int:
    return abs(n) % 10
```

- **Parameters:** 1 to 3, each annotated with `int`, `float`, `bool`, `str` or `list[int]`.
- **Return type:** `int`, `float`, `bool` or `str`.
- **Inputs:** for every parameter type used, the dialog asks how inputs are drawn: integer and float ranges, string length and characters, list length and value range. A few edge cases (empty and boundary values) always start the training set. Inputs for which the function raises an error are skipped.
- **Constants:** numbers and strings written in the body (e.g. `1000`, `-5`, `2.5`, `"big"`) become constants of the primitive set; the dialog lists them next to the signature.
- **Errors per test case** (as in PSB1): 0 / 1 for booleans, the absolute difference for numbers, the Levenshtein distance for strings. A program that crashes or runs too long gets the largest error.

### Logic expression

A Boolean function of 1 to 8 variables, e.g. `(a and not b) or (c ^ d)`. Allowed are `and`, `or`, `not`, `^` (xor), `&`, `|`, `==`, `!=`, `x if c else y` and the constants `True`, `False`, `0`, `1`. The full truth table is the data.

### Data set

A CSV / TSV / … file; delimiter, header, column types, target and input columns are detected and can be changed. It becomes a regression, program synthesis or Boolean problem.

---

## Custom functions (function sets)

New building blocks for one of the three function sets. Each appears in its section with a switch, so it can be turned on and off like the built-in functions.

| Set | Signature | Example |
|---|---|---|
| Symbolic regression | 1–3 `float` inputs → `float` | `def square(x: float) -> float: return x * x` |
| Program synthesis | 1–3 inputs of type `int`, `float`, `bool`, `str`, `list[int]` → `int`, `float`, `bool` or `str` | `def is_even(n: int) -> bool: return n % 2 == 0` |
| Logic | 1–3 `bool` inputs → `bool` | `def implies(a: bool, b: bool) -> bool: return (not a) or b` |

- **When it is used:** a program synthesis function is used only when its types occur in the problem. For example, a `str` function is left out of a problem without strings.
- **Errors (symbolic regression):** the function is applied to single numbers (one data row at a time). An error or an infinite result counts as NaN, as with the protected built-in functions.
- **Errors (program synthesis and logic):** an error counts as a failed run of the program on that test case.
- **Results:** they are converted to the declared type. Integers are clipped to ±10⁹ and strings to 200 characters.
- **Test before saving:** the function is called with 30 random inputs, or the whole truth table for logic. It must work for at least one of them and must return the declared type.
- **Names:** a name of a built-in function (`add`, `sin`, `AND`, …) or of an input (`x`, `y`, `e`) cannot be used.

---

## Custom operators

All operators are selected in the dropdown of their parameter (marked *custom*). Each one is tested with a short run (20 individuals, 3 generations) before it is saved. In all of them, **lower fitness is better**: for regression it is the error, for program synthesis and logic the mean error per test case.

### Fitness function (symbolic regression)

```python
def my_fitness(outputs: list, targets: list) -> float
```

| Input | Meaning |
|---|---|
| `outputs` | values of the program on the training cases, in the order of the data |
| `targets` | the wanted values on the same cases |

- **Returns:** a number; lower is better. Infinite or NaN results count as infinitely bad.
- **When:** for every individual on the training data, and on the validation data for the validation error.
- **Per-case errors:** lexicase selection and the case views use the absolute differences `|output − target|`. Linear scaling is not applied.

### Selection

```python
def my_selection(fitness: list, errors: list, sizes: list, k: int, rng) -> list
```

| Input | Meaning |
|---|---|
| `fitness[i]` | fitness of individual *i* (lower is better) |
| `errors[i]` | its error on each training case: for regression the squared error (absolute with MAE or a custom fitness function), for program synthesis and logic 0 / 1 or the distance |
| `sizes[i]` | its number of nodes |
| `k` | how many parents to choose (population size minus elitism) |
| `rng` | a `random.Random`, seeded with the run's seed |

- **Returns:** `k` indices into the population. Repetitions are allowed: an individual chosen twice becomes a parent twice.
- **When:** once per generation, to choose all parents.
- **With down-sampling** (Additional Parameters → *Down-sampling* "Random" or "Informed"): `errors[i]` contains only the cases of this generation's down-sample, and `fitness[i]` is the fitness on these cases.

### Crossover

```python
def my_crossover(parent1, parent2, rng) -> (child1, child2)
```

- **Inputs:** two parents as trees: lists of nodes in prefix order.
  - A node has `arity` (number of children), `name` and `ret` (its type).
  - `tree.searchSubtree(i)` returns the slice of the subtree that starts at position *i*.
  - The parents are copies, so you can change and return them.
- **Types:** in program synthesis the tree is typed, so only swap subtrees with the same `ret`.
- **Returns:** two trees.
- **When:** for pairs of parents, with the crossover probability.
- **Afterwards:** the max. tree depth (and, if chosen, the size limit or a custom bloat control) is applied. A child that is too large is replaced by a parent.

### Mutation

```python
def my_mutation(individual, random_subtree, rng) -> individual
```

- **Inputs:**
  - `individual`: a tree, as for crossover.
  - `random_subtree(type)`: creates a random subtree that returns `type` (usually the `ret` of the node it replaces), with the depth range and method set under *New subtree depth* / *New subtree method* (Advanced parameters; an adaptive control can change them). `random_subtree(type, max_depth)` instead grows a subtree of at most `max_depth`.
- **Returns:** the changed tree.
- **When:** for each child, with the mutation probability; the same limits apply afterwards.

### Initialization

```python
def my_initialization(full, grow, size, min_depth, max_depth, rng) -> list
```

| Input | Meaning |
|---|---|
| `full(depth)` | a random tree whose leaves all lie at exactly this depth |
| `grow(depth)` | a random tree of at most this depth |
| `size` | population size |
| `min_depth`, `max_depth` | min. / max. initial tree depth from the settings |

- **Returns:** exactly `size` trees.
- **When:** once, for the first generation.

### Bloat control

```python
def my_bloat_control(child_size, child_depth, parent_size, parent_depth) -> bool
```

- **Inputs:** number of nodes and depth of a new child and of its parent.
- **Returns:** `True` keeps the child; `False` keeps the parent instead.
- **When:** for every child of crossover and mutation. The max. tree depth always applies as well.

### Adaptive control

Adaptive control (in the literature: *parameter control*) changes parameters *during* a run, e.g. with the generation, the diversity or the progress. Next to your own functions there are presets:

- *Linear mutation decrease*
- *Diversity-triggered mutation*
- *Stagnation boost*
- *Adaptive operator rates*
- *Dynamic depth limit*, after Silva & Almeida, *Dynamic Maximum Tree Depth*, GECCO 2003. The limit starts at the initial depth. It rises only when a deeper child is better than the best individual so far; otherwise the parent takes the child's place.

```python
def depth_schedule(state, params) -> dict
```

| `state.` | Meaning |
|---|---|
| `generation` | the generation about to be created (2 … `generations`) |
| `generations` | number of generations of the run |
| `progress` | `(generation − 1) / (generations − 1)`, from just above 0 to 1 |
| `best` | best training fitness so far (error for regression, % of training cases solved for program synthesis and logic) |
| `median` | median training fitness of the current population (same unit) |
| `genotypic_diversity` | % of different trees in the population |
| `phenotypic_diversity` | % of different error vectors in the population |
| `size`, `depth` | median tree size (nodes) and depth of the population |
| `selected_share` | % of the population that was selected at least once as a parent |
| `crossover_better`, `mutation_better` | % of crossover / mutation children better than their parent (NaN if there were none) |
| `stagnation` | generations since `best` last improved |
| `start` | dict of the values set before the run |
| `history` | list of the `state`s of the earlier generations |
| `kind` | `"sr"`, `"ps"` or `"logic"` |

`params` is a dict of the current values. Return a dict with the values to change, or `None` / `{}` to keep them all.

**Numbers**

| Parameter | Range | Type |
|---|---|---|
| `population_size` | ≥ 2 | int: size of the next generation |
| `crossover_probability`, `mutation_probability` | 0 – 1 | float |
| `tourn_size` | ≥ 1 | int |
| `parsimony_size` (double tournament) | 1 – 2 | float: the smaller of two individuals wins the size tournament with probability `parsimony_size / 2` |
| `lexicase_epsilon` (ε-lexicase with fixed ε) | ≥ 0 | float: candidates within ε of the best error on a case survive |
| `ds_rate` (down-sampling), `outlier_rate` (ROIDS) | 0 – 1 | float |
| `parent_sampling_rate` (informed down-sampling, ROIDS) | 0 – 1 | float |
| `distance_interval` (informed down-sampling, ROIDS) | ≥ 1 | int |
| `max_tree_depth` | 1 – 1000 | int |
| `max_tree_size` | ≥ 1 | int (switches the size limit on) |
| `elitism` | ≥ 0 (at most population size − 1) | int |
| `inner_node_probability` (leaf-biased crossover) | 0 – 1 | float: probability that the crossover point is an inner node |
| `subtree_min_depth`, `subtree_max_depth` (new subtrees of uniform mutation and `random_subtree`) | 0 – 90 | int; if min > max, the two are swapped |

**Operators**: the value is a name, i.e. a built-in choice as shown in the dropdown, or the name of one of your saved custom operators of that kind.

| Parameter | Built-in choices |
|---|---|
| `selection_method` | `"Tournament"`, `"Double tournament"`, `"Roulette"`, `"Stochastic universal sampling"`, `"Lexicase"`, `"Epsilon-Lexicase"`, `"Fixed Epsilon-Lexicase"`, `"DS Lexicase"`, `"DS Epsilon-Lexicase"`, `"IDS Lexicase"`, `"IDS Epsilon-Lexicase"`, `"ROIDS"`, `"Random"`, `"Best"`, `"Worst"` |
| `crossover_type` | `"One-point"`, `"Leaf-biased"` (the earlier name `"Leaf-biased (90/10)"` is still accepted) |
| `mutation_type` | `"Uniform (subtree)"`, `"Node replacement"`, `"Shrink"`, `"Ephemeral constants"`, `"Mixed"` |
| `bloat_control` | `"Depth limit"`, `"Depth + size limit"`, `"Lexicographic parsimony"` |

**Settings with fixed choices** (no custom operators):

| Parameter | Choices |
|---|---|
| `subtree_method` | `"Full"`, `"Grow"`, `"Ramped Half-n-Half"`: how new subtrees of mutations are built |
| `ephemeral_mode` | `"One constant"`, `"All constants"`: what the ephemeral mutation changes |
| `double_tournament_order` | `"Fitness first"`, `"Size first"`: which tournament of double tournament comes first |
| `downsampling` | `"None"`, `"Random"`, `"Informed"`: down-sampling for the selection methods without one of their own (all except the DS / IDS methods and ROIDS), also for custom selections |

The variation settings act whenever their operator is used, also while an adaptive control switches between operators (e.g. `subtree_max_depth` counts as soon as `mutation_type` becomes `"Uniform (subtree)"` or `"Mixed"`).

For example, `{"selection_method": "Tournament" if state.progress > 0.5 else "Lexicase"}` switches from lexicase to tournament selection halfway through the run, and `{"subtree_max_depth": 4 if state.stagnation >= 5 else 2}` lets mutation insert larger subtrees while the run stagnates.

**Not changeable**, on purpose:

- **Number of generations:** the run length defines `progress`, the plots and the comparison of runs in experiments.
- **Fitness function:** switching it would change what the error curves measure in the middle of the run.
- **Initialization:** it only creates the first generation.
- **Function set, problem, data and seed:** they define *what* is learned, not *how*.

**How changes are applied**

- **Values outside a range** are clipped. Integers are rounded.
- **Errors:** unknown names, an operator name that does not exist, values that are not numbers and other return types stop the run with a message that lists the allowed values.
- **When:** before every generation except the first, since the initial population is created with the settings. A new operator acts from the next generation on.
- **Population size:** the next generation gets the new size, chosen by the selection method from the current population.
- **Depth and size limits** apply to new children. Individuals copied unchanged from the previous generation keep their size.
- **Seeing the effect:**
  - The plot view *Parameters over generations* (Run tab: left plot; Experiments tab: right plot) shows up to five parameters that changed, each in its own panel. Operators appear as names; in the Experiments tab the operator most runs use is shown.
  - The CSV exports contain every parameter as a `param_<name>` column. In the aggregated CSV, an operator appears as the share of runs using it in each generation, e.g. `param_selection_method=Lexicase`.

Example: the max. tree depth grows during the first half of the run and shrinks again in the second half.

```python
def depth_schedule(state, params):
    if state.progress < 0.5:
        depth = 6 + (17 - 6) * state.progress / 0.5
    else:
        depth = 17 - (17 - 8) * (state.progress - 0.5) / 0.5
    return {"max_tree_depth": round(depth)}
```

---

## In Python scripts

Custom problems, data sets, operators and switched-on functions saved in the app can be used by name in scripts:

```python
import gpexplore as gpx

if __name__ == "__main__":
    result = gpx.run("my_problem", selection_method="my_selection", parameter_control="depth_schedule")   # adaptive control: parameter_control=
```
