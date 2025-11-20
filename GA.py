from typing import Tuple
import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
import random
np.random.seed(1)
random.seed(1)

from numpy.ma.core import append
from numpy.matlib import empty

budget = 10000
pop_size = 100
dim = 50

def selection(pop, fitness):
    sorted_pairs = sorted(zip(pop, fitness), key=lambda x: x[1], reverse=True)
    new_pop, _ = zip(*sorted_pairs)
    new_pop = new_pop[:pop_size] # Must always be even for crossover

    return list(new_pop)

def crossover(pop, type_crossover="index", n=3):
    new_pop = []
    while pop:
        a = random.sample(range(len(pop)),1)
        ind1 = pop.pop(a[0])
        b = random.sample(range(len(pop)),1)
        ind2 = pop.pop(b[0])
        # Adding originals to new pop
        new_pop.append(ind1)
        new_pop.append(ind2)
        new_ind1 = []
        new_ind2 = []
        if type_crossover == "index":
            indices = sorted(random.sample(range(1, len(ind1)), n))
            swap = False
            for enum, bit in enumerate(zip(ind1, ind2)):
                if enum in indices:
                    swap = not swap
                new_ind1.append(bit[1] if swap else bit[0])
                new_ind2.append(bit[0] if swap else bit[1])
        elif type_crossover == "uniform":
            pass
        else:
            raise TypeError("Crossover type must be index or uniform")
        new_pop.append(new_ind1)
        new_pop.append(new_ind2)

    return new_pop


def mutate(pop, mutation_rate):
    new_pop = []
    for ind in pop:
        new_ind = []
        for i in ind:
            if random.random() < mutation_rate:
                new_ind.append((i+1)%2)
            else:
                new_ind.append(i)
        new_pop.append(new_ind)

    return new_pop

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    pop = [[random.choice([0, 1]) for _ in range(dim)] for _ in range(pop_size)]

    fitness = []
    while problem.state.evaluations < budget:
        if fitness:
            pop = crossover(pop, "index", 3)
            pop = mutate(pop, mutation_rate=0.0075)
            pop = selection(pop, fitness)
        fitness = problem(pop)


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=dim, fid=18)
    for run in range(20):
        studentnumber1_studentnumber2_GA(F18)
        F18.reset()
    _logger.close()

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20):
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()