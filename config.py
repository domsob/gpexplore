import numpy as np

gp_running = False
best_individuals_per_generation = []
best_invididuals_per_generation_fitness = []
best_individuals_per_generation_validation = []
gp_logbook = []

noise_intensity = 0.0

generations = 50
population_size = 250
crossover_probability = 0.8
mutation_probability = 0.05
target_function = "x**4 + x**3 + x**2 + x"
main_plots = None

add_func = True
sub_func = True
mul_func = True
protectedDiv_func = False
analyticQuotient_func = True
neg_func = True
sin_func = True
cos_func = True
int101_erc = True

training_set_size = 50
training_set_x1 = list(np.random.uniform(-1, 1, training_set_size))
training_set_x1.sort()
training_set_y1 = [eval(target_function) + np.random.uniform(-noise_intensity, noise_intensity) for x in training_set_x1]

validation_set_size = 200
validation_set_x1 = list(np.linspace(-1.1, 1.1, validation_set_size))

selection_method = "Tournament"
tourn_size = 3
initialization = "Ramped Half-n-Half"
min_initial_tree_depth = 2
max_initial_tree_depth = 5
max_tree_depth = 17
plot_adjust = True
inter_gen_pause = 0.5

license_text = open('LICENSE.txt', encoding='utf-8').read()
