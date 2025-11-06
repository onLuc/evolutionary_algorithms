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
budget = 5000
pop_size = 100
dim = 50

def mutate(pop):
    pass

def crossover(pop):
    pass

def selection(pop):
    pass

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    pop = [[random.choice([0, 1]) for _ in range(dim)] for _ in range(pop_size)]
    pop_matrix = np.array(pop, dtype=int)
    trans_matrix = 2 * pop_matrix - 1
    print(trans_matrix)
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # pop = mutate(pop)
        # pop = crossover(pop)
        # pop = selection(pop)
        # please implement the mutation, crossover, selection here

        # f = problem(trans_matrix)
        fitness_values = [problem(ind) for ind in trans_matrix]
        print(fitness_values)


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
    for run in range(1):
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    # F23, _logger = create_problem(dimension=49, fid=23)
    # for run in range(20):
    #     studentnumber1_studentnumber2_GA(F23)
    #     F23.reset()
    # _logger.close()