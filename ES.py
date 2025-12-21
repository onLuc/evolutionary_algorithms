import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import matplotlib.pyplot as plt
from sympy.multipledispatch.dispatcher import RaiseNotImplementedError

budget = 50000
dimension = 10
mu = 15
lambd = 100
lb, ub = -5, 5
seed = 42
n_angles = int(dimension * (dimension-1) / 2) # =45
np.random.seed(seed)
tau_prime = 1 / np.sqrt(2 * dimension)
tau = 1 / np.sqrt(2 * np.sqrt(dimension))
# beta = np.deg2rad(5)

def build_covariance_matrix(sigmas, alphas):
    D = np.diag(sigmas ** 2) # scaling matrix
    R = np.eye(dimension) # rotation matrix
    k = 0
    for i in range(dimension - 1):
        for j in range(i + 1, dimension):
            Ri = np.eye(dimension)
            cos_a = np.cos(alphas[k])
            sin_a = np.sin(alphas[k])
            Ri[i, i] = Ri[j, j] = cos_a
            Ri[i, j] = -sin_a
            Ri[j, i] = sin_a
            R = R @ Ri
            k += 1

    return R @ D @ R.T

def recombine(pop):
    # Global intermediate recombination
    to_mean_x = [ind["x"] for ind in pop]
    to_mean_sigmas = [ind["sigmas"] for ind in pop]
    # to_mean_alphas = [ind["alphas"] for ind in pop]
    return {"x": np.mean(to_mean_x, axis=0),
            "sigmas": np.mean(to_mean_sigmas, axis=0),
            # "alphas": np.mean(to_mean_alphas, axis=0)
            }

def recombine_n(pop, n):
    p1, p2 = np.random.choice(pop, n, replace=False)
    return {
        "x": (p1["x"] + p2["x"]) / n,
        "sigmas": (p1["sigmas"] + p2["sigmas"]) / n,
        "alphas": (p1["alphas"] + p2["alphas"]) / n
    }


# def mutation(pop, sigma, C):
#     evals, evecs = np.linalg.eigh(C)
#     A = evecs @ np.diag(np.sqrt(np.maximum(evals, 1e-10)))
#     new_pop = []
#     for _ in range(lambd):
#         z = np.random.normal(pop, 1, dimension)
#         y = A @ z
#         x = pop + sigma * y
#         x = np.clip(x, lb, ub)
#         new_pop.append({"x": x, "y": y})
#
#     return new_pop

def mutate(ind):
    pop = []
    for _ in range(lambd):
        x, sigmas = ind['x'], ind['sigmas']

        common_noise = tau_prime * np.random.normal(0, 1)
        new_sigmas = sigmas * np.exp(common_noise + tau * np.random.normal(0, 1, dimension))
        new_sigmas = np.maximum(new_sigmas, 1e-6)

        # new_alphas = alphas + np.random.normal(0, 1, n_angles)
        # new_alphas = alphas + beta * np.random.normal(0, 1, n_angles)


        # C = build_covariance_matrix(new_sigmas, new_alphas)
        # Sampling N(0, C)
        # try:
        #     vals, vecs = np.linalg.eigh(C)
        #     # Ensure no negative eigenvalues due to precision
        #     A = vecs @ np.diag(np.sqrt(np.maximum(vals, 0)))
        #     dx = A @ np.random.normal(0, 1, dimension)
        # except np.linalg.LinAlgError:
        #     print(RaiseNotImplementedError)
        #     # Fallback if matrix is not positive definite
        dx = new_sigmas * np.random.normal(0, 1, dimension)

        new_x = np.clip(x + dx, -5, 5)
        pop.append({'x': new_x,
                    'sigmas': new_sigmas})

    return pop


def evaluate(pop, problem):
    performance = []
    for ind in pop:
        f = problem(ind["x"])
        performance.append({"x": ind["x"],
                            "sigmas": ind["sigmas"],
                            "f": f})

    return performance

def select(performance):
    update = sorted(performance, key=lambda x: x['f'])[:mu] # sorts based on best performance and takes top mu individuals
    top_performer = update[0]["f"]

    return top_performer, update


def plot_top_performers(data_list, title="Top performer over time"):
    """
    Plots a list of values against their indices.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(data_list, marker='o', linestyle='-', color='b')

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best found value")
    plt.grid(True)
    plt.show()


def studentnumber1_studentnumber2_ES(problem):
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    pop = []
    for _ in range(mu):
        pop.append({
            'x': np.random.uniform(-5, 5, dimension),
            'sigmas': np.ones(dimension) * 0.2,
        })
    # problem(pop)
    top_performers = []
    best_so_far = np.inf
    successes = 0
    while problem.state.evaluations < budget:
        # print(problem.optimum)
        parent = recombine(pop) # takes all mu parents to create a single global intermediate
        # offspring = []
        # for _ in range(lambd):
        #     parent = recombine_n(pop, 2)
        #     offspring.append(mutate(parent, n_children=1)[0])
        pop = mutate(parent)
        # print(pop)
        performance = evaluate(pop, problem)
        # pop = mutation(pop, sigma, c) # creates lambda offspring by sampling normal distribution lambda times
        top_performer, pop = select(performance)
        best_so_far = min(top_performer, best_so_far)
        # print(top_y)
        top_performers.append(top_performer)

    plot_top_performers(top_performers)


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F23, _logger = create_problem(23)
    # for run in range(20):
    for run in range(1):
        studentnumber1_studentnumber2_ES(F23)
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


