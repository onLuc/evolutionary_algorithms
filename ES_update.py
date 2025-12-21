import numpy as np
from ioh import get_problem, logger, ProblemClass
import matplotlib.pyplot as plt

# Configuration
budget = 50000
dimension = 10
mu = 15
lambd = 100
lb, ub = -5, 5

# Learning rates for individual self-adaptation
tau_prime = 1 / np.sqrt(2 * dimension)
tau = 1 / np.sqrt(2 * np.sqrt(dimension))


def get_weights(mu):
    """ Calculate weights for weighted recombination: better individuals have more influence """
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    return weights / np.sum(weights)


def weighted_recombine(pop, weights):
    """ Weighted Global Intermediary Recombination """
    new_x = np.zeros(dimension)
    new_sigmas = np.zeros(dimension)
    for i in range(mu):
        new_x += weights[i] * pop[i]['x']
        new_sigmas += weights[i] * pop[i]['sigmas']
    return {"x": new_x, "sigmas": new_sigmas}


def mutate(parent, global_scale):
    # Self-adaptive mutation
    common_noise = np.random.normal(0, 1)
    # Strategy parameters mutate first
    new_sigmas = parent['sigmas'] * np.exp(tau_prime * common_noise + tau * np.random.normal(0, 1, dimension))

    # Apply global scale multiplier to control exploitation/exploration
    effective_sigmas = new_sigmas * global_scale
    effective_sigmas = np.maximum(effective_sigmas, 1e-9)  # Lower floor to allow precision

    # Mutate search variables
    new_x = parent['x'] + effective_sigmas * np.random.normal(0, 1, dimension)
    return {'x': np.clip(new_x, lb, ub), 'sigmas': new_sigmas}


def student_ES_improved(problem):
    weights = get_weights(mu)
    # Initialize mu parents
    pop = [{'x': np.random.uniform(lb, ub, dimension), 'sigmas': np.ones(dimension) * 0.5} for _ in range(mu)]

    best_so_far_history = []
    best_val = np.inf
    global_scale = 1.0

    while problem.state.evaluations < budget:
        # Sort current population to apply weights
        # (Though in (mu, lambda) they are already sorted from the previous selection)

        # 1. RECOMBINATION
        centroid = weighted_recombine(pop, weights)
        centroid_f = problem(centroid['x'])  # evaluate the "mean" individual

        # 2. MUTATION & EVALUATION
        offspring = []
        success_count = 0
        for _ in range(lambd):
            child = mutate(centroid, global_scale)
            child['f'] = problem(child['x'])
            if child['f'] < centroid_f:
                success_count += 1
            offspring.append(child)

        # 3. DYNAMIC CONTROL (1/5th Rule logic)
        # If success rate is low, we are likely 'bouncing' around a peak; shrink scale.
        ps = success_count / lambd
        if ps > 0.20:
            global_scale *= 1.1  # Expand
        else:
            global_scale *= 0.9  # Contract

        # 4. SELECTION: (mu, lambda)
        offspring.sort(key=lambda ind: ind['f'])
        pop = offspring[:mu]

        if offspring[0]['f'] < best_val:
            best_val = offspring[0]['f']
        best_so_far_history.append(best_val)

        if len(best_so_far_history) % 50 == 0:
            print(f"Eval: {problem.state.evaluations} | Best: {best_val:.4e} | Scale: {global_scale:.2e}")

    return best_so_far_history


if __name__ == "__main__":
    # Test on Katsuura (F23)
    problem = get_problem(23, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)
    results = student_ES_improved(problem)

    plt.figure(figsize=(8, 5))
    plt.semilogy(results)
    plt.title("Improved ES: Weighted Recombination + Success-based Scaling")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Log Scale)")
    plt.grid(True)
    plt.show()