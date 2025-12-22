import numpy as np
from ioh import get_problem, logger, ProblemClass
import matplotlib.pyplot as plt

# ============================================================
# Tunable parameters (algorithm / experiment / numerics / plots)
# ============================================================

# --- Experiment / IOH ---
BUDGET = 5000
DIMENSION = 10
LB, UB = -5.0, 5.0
SEED = 42

N_RUNS = 20 #20 for assignment

# --- CMA-ES population ---
LAMBDA = 40
MU = 7 

# --- CMA-ES initialization ---
SIGMA0 = 0.03

# --- Numerical stability --- (avoids issues with non-positive definite covariance)
EIGENVAL_FLOOR = 1e-20
C_JITTER = 1e-14

# --- Plotting / output ---
PLOT_SHOW_EACH_RUN = False
PRINT_RUN_STATS = True  # print overall stats after N_RUNS


# ----------------------------
# Utilities
# ----------------------------
def reflect_bounds(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Mirror-reflect x into [lb, ub] (less distorting than clip)."""
    w = ub - lb
    y = (x - lb) % (2 * w)              # map to [0, 2w)
    y = np.where(y > w, 2 * w - y, y)   # reflect second half
    return lb + y


def evaluate(pop, problem):
    """Reused shape from your code: list[dict] -> list[dict] with fitness."""
    performance = []
    for ind in pop:
        f = problem(ind["x"])  # consumes 1 evaluation
        d = dict(ind)
        d["f"] = f
        performance.append(d)
    return performance


def plot_top_performers(best_gen, best_so_far, title="CMA-ES convergence"):
    plt.figure(figsize=(9, 5))
    plt.plot(best_gen, marker="o", linestyle="-", label="Best of generation")
    plt.plot(best_so_far, marker="x", linestyle="--", label="Best so far")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.grid(True)
    plt.legend()
    plt.show()


# ----------------------------
# CMA-ES core
# ----------------------------
def compute_cma_parameters(n, mu_used):
    """Compute (weights, mueff, cc, cs, c1, cmu, damps, chiN) for CMA-ES."""
    weights = np.log(mu_used + 0.5) - np.log(np.arange(1, mu_used + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(
        1 - c1,
        2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff),
    )
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    return weights, mueff, cc, cs, c1, cmu, damps, chiN


def eigen_decomposition(C):
    """Eigendecomposition so A = B*D with C = B diag(D^2) B^T."""
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, EIGENVAL_FLOOR)
    D = np.sqrt(vals)
    B = vecs
    invsqrtC = B @ np.diag(1.0 / D) @ B.T
    return B, D, invsqrtC


def sample_offspring(mean, sigma, B, D, n_offspring, rng):
    """Sample offspring and reflect into bounds (unchanged)."""
    pop = []
    for _ in range(n_offspring):
        z = rng.normal(0.0, 1.0, size=DIMENSION)
        y = B @ (D * z)
        x = mean + sigma * y
        x = reflect_bounds(x, LB, UB)
        y = (x - mean) / sigma
        pop.append({"x": x, "y": y})
    return pop


def studentnumber1_studentnumber2_ES(problem, rng: np.random.Generator):
    """
    CMA-ES (μ/λ)-ES:
    - mean m
    - global step-size sigma
    - covariance matrix C
    - evolution paths ps, pc
    """
    n = DIMENSION

    mean = rng.uniform(LB, UB, size=n)
    sigma = SIGMA0
    C = np.eye(n)

    ps = np.zeros(n)
    pc = np.zeros(n)

    B, D, invsqrtC = eigen_decomposition(C)
    eigeneval = 0

    best_gen_history = []
    best_so_far_history = []
    best_so_far = np.inf

    mu_used_prev = None

    while problem.state.evaluations < BUDGET:
        remaining = BUDGET - problem.state.evaluations
        n_offspring = min(LAMBDA, remaining)
        if n_offspring <= 0:
            break

        mu_used = min(MU, n_offspring)
        if mu_used != mu_used_prev:
            weights, mueff, cc, cs, c1, cmu, damps, chiN = compute_cma_parameters(n, mu_used)
            mu_used_prev = mu_used

        E = sample_offspring(mean, sigma, B, D, n_offspring, rng)
        offspring = evaluate(E, problem)
        offspring_sorted = sorted(offspring, key=lambda d: d["f"])

        best_gen = offspring_sorted[0]["f"]
        best_so_far = min(best_so_far, best_gen)
        best_gen_history.append(best_gen)
        best_so_far_history.append(best_so_far)

        x_sel = np.array([d["x"] for d in offspring_sorted[:mu_used]])
        y_sel = np.array([d["y"] for d in offspring_sorted[:mu_used]])

        mean = (weights @ x_sel)
        y_w = (weights @ y_sel)

        # CSA evolution path
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)

        norm_ps = np.linalg.norm(ps)
        counteval = problem.state.evaluations
        hsig = 1.0 if (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * counteval / LAMBDA))) < (1.4 + 2 / (n + 1)) * chiN else 0.0

        # Covariance evolution path
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

        # Rank-μ covariance update
        C_mu = np.zeros((n, n))
        for i in range(mu_used):
            yi = y_sel[i]
            C_mu += weights[i] * np.outer(yi, yi)

        C = (1 - c1 - cmu) * C \
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * C_mu

        C = 0.5 * (C + C.T)
        C += C_JITTER * np.eye(n)

        # Step-size update (CSA)
        sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))

        # eigen update
        B, D, invsqrtC = eigen_decomposition(C)
        eigeneval = counteval

    if PLOT_SHOW_EACH_RUN:
        plot_top_performers(best_gen_history, best_so_far_history, title="CMA-ES on BBOB F23 (Katsuura)")

    return best_so_far

# ----------------------------
# IOH setup
# ----------------------------
def create_problem(fid: int):
    problem = get_problem(fid, dimension=DIMENSION, instance=1, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="data",
        folder_name="run",
        algorithm_name="CMA-ES",
        algorithm_info="PA part2: CMA-ES with bound reflection, weighted recombination, covariance adaptation",
    )
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    F23, _logger = create_problem(23)

    final_scores = []

    for run in range(N_RUNS):
        rng = np.random.default_rng(SEED + run)
        best = studentnumber1_studentnumber2_ES(F23, rng=rng)
        final_scores.append(best)
        F23.reset()

    _logger.close()

    if PRINT_RUN_STATS:
        scores = np.array(final_scores, dtype=float)
        print("\n=== Final best-so-far after budget (per run) ===")
        print(f"Runs:   {N_RUNS}")
        print(f"Mean:   {scores.mean():.6f}")
        print(f"Median: {np.median(scores):.6f}")
        print(f"Std:    {scores.std(ddof=1):.6f}")
        print(f"Min:    {scores.min():.6f}")
        print(f"Max:    {scores.max():.6f}")
