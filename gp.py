import config
import operator
import math
from math import cos, sin, sqrt
import random
import time
import numpy

from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def protectedDiv(left, right):
    if right != 0:
        return left / right
    else:
        return 1.0

def analyticQuotient(left, right):
    return left / sqrt(1 + right * right)

def gp_loop(population, toolbox, cxpb, mutpb, ngen, val_fit, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    config.best_individuals_per_generation.append(str(halloffame[0]))
    config.best_invididuals_per_generation_fitness.append(halloffame[0].fitness.values[0])
    config.best_individuals_per_generation_validation.append(val_fit(halloffame[0]))
    config.gp_logbook = logbook

    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        time.sleep(0.3 + config.inter_gen_pause)
        if config.gp_running == False:
            break

        offspring = toolbox.select(population, len(population))

        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        config.best_individuals_per_generation.append(str(halloffame[0]))
        config.best_invididuals_per_generation_fitness.append(halloffame[0].fitness.values[0])
        config.best_individuals_per_generation_validation.append(val_fit(halloffame[0]))
        config.gp_logbook = logbook

        if verbose:
            print(logbook.stream)

    return population, logbook

def get_pset():
    pset = gp.PrimitiveSet("MAIN", 1)
    if config.add_func:
        pset.addPrimitive(operator.add, 2)
    if config.sub_func:
        pset.addPrimitive(operator.sub, 2)
    if config.mul_func:
        pset.addPrimitive(operator.mul, 2)
    if config.protectedDiv_func:
        pset.addPrimitive(protectedDiv, 2)
    if config.analyticQuotient_func:
        pset.addPrimitive(analyticQuotient, 2)
    if config.neg_func:
        pset.addPrimitive(operator.neg, 1)
    if config.cos_func:
        pset.addPrimitive(math.cos, 1)
    if config.sin_func:
        pset.addPrimitive(math.sin, 1)
    if config.int101_erc:
        pset.addEphemeralConstant(str(time.time()), partial(random.randint, -1, 1)) 
    pset.renameArguments(ARG0='x')
    return pset

def run():
    pset = None
    toolbox = None

    pset = get_pset()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    if config.initialization == "Ramped Half-n-Half":
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=config.min_initial_tree_depth, max_=config.max_initial_tree_depth)
    elif config.initialization == "Full":
        toolbox.register("expr", gp.genFull, pset=pset, min_=config.min_initial_tree_depth, max_=config.max_initial_tree_depth)
    elif config.initialization == "Grow":
        toolbox.register("expr", gp.genGrow, pset=pset, min_=config.min_initial_tree_depth, max_=config.max_initial_tree_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def calculate_validation_fitness(individual):
        points = config.validation_set_x1
        func = toolbox.compile(expr=individual)
        sqerrors = ((func(x) - eval(config.target_function))**2 for x in points)
        return math.fsum(sqerrors) / len(points)

    def evalSymbReg(individual):
        points = config.training_set_x1
        func = toolbox.compile(expr=individual)
        sqerrors = []
        for x_val, y_val in zip(config.training_set_x1, config.training_set_y1):
            sqerrors.append((func(x_val) - y_val)**2)
        return math.fsum(sqerrors) / len(points),

    toolbox.register("evaluate", evalSymbReg)
    if config.selection_method == "Tournament":
        toolbox.register("select", tools.selTournament, tournsize=config.tourn_size)
    elif config.selection_method == "Random":
        toolbox.register("select", tools.selRandom)
    elif config.selection_method == "Best":
        toolbox.register("select", tools.selBest)
    elif config.selection_method == "Worst":
        toolbox.register("select", tools.selWorst)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_depth))

    pop = toolbox.population(n=config.population_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_validation_fit = tools.Statistics(calculate_validation_fitness)
    mstats = tools.MultiStatistics(training=stats_fit, validation=stats_validation_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = gp_loop(pop, toolbox, config.crossover_probability, config.mutation_probability, (config.generations -1), calculate_validation_fitness, stats=mstats, halloffame=hof, verbose=False)
