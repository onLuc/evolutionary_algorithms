from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
import random

from numpy.ma.core import append
from numpy.matlib import empty

np.random.seed(1)
random.seed(1)
budget = 10000
pop_size = 100
dim = 50

def selection(pop, fitness):
    # print(pop)
    # print(fitness)
    sorted_pairs = sorted(zip(pop, fitness), key=lambda x: x[1], reverse=True)
    new_pop, _ = zip(*sorted_pairs)
    new_pop = new_pop[:51]

    return new_pop

def crossover(pop):
    pass

def mutate(pop):
    mutation_rate = 0.05
    new_pop = []
    for ind in pop:
        new_ind = []
        for i in ind:
            if random.random() < mutation_rate:
                new_ind.append(i+1%2)
            else:
                new_ind.append(i)
        new_pop.append(new_ind)

    return new_pop

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    pop = [[random.choice([0, 1]) for _ in range(dim)] for _ in range(pop_size)]

    fitness = []
    while problem.state.evaluations < budget:
        if fitness:
            # print(fitness)
            pop = selection(pop, fitness)
            # pop = crossover(pop)
            pop = mutate(pop)
        # please implement the mutation, crossover, selection here
        trans_list = [[2 * i - 1 for i in ind] for ind in pop]
        fitness = problem(trans_list)


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    triggers = [
        ioh.logger.trigger.Each(1)
    ]
    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
        triggers=triggers,
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=dim, fid=18)
    # for run in range(20):

    for run in range(20):
        np.random.seed(1)
        random.seed(1)
        F18.reset()
        studentnumber1_studentnumber2_GA(F18)

    # for run in range(1):
    #     studentnumber1_studentnumber2_GA(F18)
    #     F18.reset() # it is necessary to reset the problem after each independent run
    # _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    # F23, _logger = create_problem(dimension=49, fid=23)
    # for run in range(20):
    #     studentnumber1_studentnumber2_GA(F23)
    #     F23.reset()
    # _logger.close()