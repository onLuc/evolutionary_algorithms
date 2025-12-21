# GA_gpt.py
# -----------------------------------------------------------------------------
# Genetic Algorithm for IOH PBO problems F18 (LABS) and F23 (N-Queens)
# -----------------------------------------------------------------------------
# Assignment summary:
# - F18 (LABS) evaluates how correlated a binary sequence is with its shifts.
#   Higher scores mean lower autocorrelation (better sequences).
# - F23 (N-Queens) evaluates queen placements encoded as bitstrings.
#   Higher scores mean fewer conflicts between queens (closer to valid board).
#
# This script runs the GA using one common hyperparameter set on both problems.
# After tuning (done in tuning.py), we fix the best parameters here and perform
# 20 independent runs per problem, each with 5,000 function evaluations.
# The results (best_y) are printed per run and recorded by IOH's logger for later
# analysis in IOHanalyzer.
# -----------------------------------------------------------------------------

from typing import Tuple 
import numpy as np
import json, os
import ioh
from ioh import get_problem, logger, ProblemClass

# =============================
# Global / default parameters
# =============================

budget = 5_000

PARAMS = dict(
    pop_size=50,
    p_cx=0.5,             # crossover probability
    mut_per_n=1.0,        # mutation probability per bit = mut_per_n / n
    elitism=2,
    tour_k=3,
    cx_type="one_point",
    cx_k=2,               # for k-point crossover
    selection_type="tournament",  # tournament, proportional, rank, truncation
    truncation_frac=0.5,
    init_type="random",   # random, biased, complementary
    init_p=0.5,           # for biased init: probability of 1
    replacement_type="elitism",  # elitism or generational
    seed=42,
    budget=budget,        # evaluations per run
)

def set_params(**kwargs):
    """Override GA hyperparameters (used by tuning.py)."""
    PARAMS.update(kwargs)

# =============================
# IOH Problem + Logger Setup
# =============================

def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    l = logger.Analyzer(
        root="data",  # change to relative path if needed
        folder_name="run",
        algorithm_name="genetic_algorithm",
        algorithm_info="Practical assignment of the EA course",
    )
    problem.attach_logger(l)

    return problem, l

# =============================
# Core GA
# =============================

def studentnumber1_studentnumber2_GA(problem: "ioh.problem.PBO") -> None:
    rng = np.random.default_rng(int(PARAMS["seed"]))
    n = int(problem.meta_data.n_variables)
    pop_size = int(PARAMS["pop_size"])
    elitism = int(PARAMS["elitism"])
    tour_k = int(PARAMS["tour_k"])
    p_cx = float(PARAMS["p_cx"])
    p_mut = float(PARAMS["mut_per_n"]) / float(n)
    max_evals = int(PARAMS["budget"])
    selection_type = PARAMS.get("selection_type", "tournament")
    truncation_frac = float(PARAMS.get("truncation_frac", 0.5))
    init_type = PARAMS.get("init_type", "random")
    init_p = float(PARAMS.get("init_p", 0.5))
    replacement_type = PARAMS.get("replacement_type", "elitism")

    # ---- initialization
    if init_type == "biased":
        pop = (rng.random((pop_size, n)) < init_p).astype(np.uint8)
    elif init_type == "complementary":
        half = pop_size // 2
        base = rng.integers(0, 2, size=(half, n), dtype=np.uint8)
        comp = 1 - base
        if pop_size % 2 == 0:
            pop = np.vstack([base, comp])
        else:
            extra = rng.integers(0, 2, size=(1, n), dtype=np.uint8)
            pop = np.vstack([base, comp, extra])
    else:
        pop = rng.integers(0, 2, size=(pop_size, n), dtype=np.uint8)
    fit = np.empty(pop_size, dtype=float)
    for i in range(pop_size):
        fit[i] = problem(pop[i].tolist())

    def _roulette(weights: np.ndarray) -> int:
        total = float(np.sum(weights))
        if total <= 0.0:
            return int(rng.integers(0, pop_size))
        probs = weights / total
        return int(rng.choice(pop_size, p=probs))

    def select_parent() -> np.ndarray:
        if selection_type == "proportional":
            weights = np.maximum(fit, 0.0)
            return pop[_roulette(weights)]
        if selection_type == "rank":
            order = np.argsort(fit)
            ranks = np.empty(pop_size, dtype=float)
            ranks[order] = np.arange(1, pop_size + 1, dtype=float)
            return pop[_roulette(ranks)]
        if selection_type == "truncation":
            m = max(1, int(np.ceil(truncation_frac * pop_size)))
            order = np.argsort(fit)
            top_idx = order[-m:]
            return pop[int(rng.choice(top_idx))]
        # default: tournament
        idx = rng.integers(0, pop_size, size=tour_k)
        f = fit[idx]
        return pop[idx[np.argmax(f)]]

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Uniform or one-point crossover depending on PARAMS['cx_type']."""
        if rng.random() >= p_cx:
            return p1.copy()
        ctype = PARAMS.get("cx_type", "uniform")
        if ctype == "one_point":
            # choose a cut in [1, n-1]
            cut = rng.integers(1, n)
            child = np.empty_like(p1)
            child[:cut] = p1[:cut]
            child[cut:] = p2[cut:]
            return child
        if ctype == "k_point":
            k = int(PARAMS.get("cx_k", 2))
            k = max(1, min(k, n - 1))
            cuts = np.sort(rng.choice(np.arange(1, n), size=k, replace=False))
            child = np.empty_like(p1)
            last = 0
            use_p1 = True
            for c in cuts:
                if use_p1:
                    child[last:c] = p1[last:c]
                else:
                    child[last:c] = p2[last:c]
                use_p1 = not use_p1
                last = c
            if use_p1:
                child[last:] = p1[last:]
            else:
                child[last:] = p2[last:]
            return child
        else:
            # default: uniform
            mask = rng.integers(0, 2, size=n, dtype=np.uint8)
            return (p1 & (1 - mask)) | (p2 & mask)

    def mutate_bitflip(x: np.ndarray) -> np.ndarray:
        flips = rng.random(n) < p_mut
        return x ^ flips.astype(np.uint8)

    # ---- generational loop
    while problem.state.evaluations < max_evals:
        remaining = max_evals - problem.state.evaluations
        if remaining <= 0:
            break

        lambda_target = min(pop_size - elitism, remaining)
        off = np.empty((lambda_target, n), dtype=np.uint8)

        for i in range(lambda_target):
            p1, p2 = select_parent(), select_parent()
            child = crossover(p1, p2)
            child = mutate_bitflip(child)
            off[i] = child

        off_fit = np.empty(lambda_target, dtype=float)
        for i in range(lambda_target):
            off_fit[i] = problem(off[i].tolist())

        if replacement_type == "generational":
            pop = off
            fit = off_fit
            continue

        order = np.argsort(fit)
        elite_idx = order[-elitism:] if elitism > 0 else np.array([], dtype=int)
        survivors = pop[elite_idx] if elitism > 0 else np.empty((0, n), dtype=np.uint8)
        survivors_fit = fit[elite_idx] if elitism > 0 else np.empty((0,), dtype=float)

        pop = np.vstack([survivors, off])
        fit = np.concatenate([survivors_fit, off_fit])

        order = np.argsort(fit)
        keep = order[-pop_size:]
        pop, fit = pop[keep], fit[keep]

    # Done. The best-so-far is in problem.state.best/current_best and in the IOH log.

# =============================
# Evaluation Script Entry Point
# =============================

def _best_y(problem) -> float:
    try:
        return float(problem.state.best.y)
    except Exception:
        return float(problem.state.current_best.y)

if __name__ == "__main__":
    # Auto-load tuned parameters from best_params.json if available
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            cfg = json.load(f)
            best_cfg = cfg.get("best_config", {})
            print("Loaded tuned parameters from best_params.json:")
            print(best_cfg)
            for k, v in best_cfg.items():
                if k in PARAMS:
                    PARAMS[k] = v

    problems = [(18, 50), (23, 49)]
    runs = 20
    seed_base = 42

    for fid, dim in problems:
        problem, log = create_problem(dimension=dim, fid=fid)
        print(f"=== F{fid} (n={dim}) : {runs} runs, budget={PARAMS['budget']} ===")
        best_vals = []
        for r in range(runs):
            set_params(seed=seed_base + r)
            problem.reset()
            studentnumber1_studentnumber2_GA(problem)
            best_val = _best_y(problem)
            best_vals.append(best_val)
            print(f"Run {r+1:02d} best_y = {best_val:.6f}")
        # Summary line per problem
        print(f"F{fid} summary -> median={np.median(best_vals):.4f}, mean={np.mean(best_vals):.4f}, min={np.min(best_vals):.4f}, max={np.max(best_vals):.4f}\n")
        log.close()
