from typing import Tuple 
import numpy as np
import json, os
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5_000 # per run

PARAMS = dict(
    pop_size=20,          # population size
    p_cx=0.7,             # crossover probability
    mut_per_n=1.0,        # mutation probability per bit = mut_per_n / n
    elitism=3,
    tour_k=3,
    cx_type="k_point",  # uniform, k_point
    cx_k=1,               # for k-point crossover
    selection_type="truncation",  # tournament, proportional, rank, truncation
    truncation_frac=0.5,
    init_type="biased",   # random, biased
    init_p=0.4,           # for biased init
    replacement_type="elitism",  # elitism or generational
    seed=42,
    budget=budget,        # evaluations per run
)

def set_params(**kwargs):
    """Override GA hyperparameters for tuning.py"""
    PARAMS.update(kwargs)

# =============================
# IOH Problem + Logger Setup
# =============================

def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    l = logger.Analyzer(
        root="data",
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
    # Params
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

    # Initialization
    if init_type == "biased":
        pop = (rng.random((pop_size, n)) < init_p).astype(np.uint8)
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

    # Selection
    def select_parent() -> np.ndarray:
        if selection_type == "proportional":
            weights = np.maximum(fit, 0.0)
            return pop[_roulette(weights)]
        
        elif selection_type == "rank":
            order = np.argsort(fit)
            ranks = np.empty(pop_size, dtype=float)
            ranks[order] = np.arange(1, pop_size + 1, dtype=float)
            return pop[_roulette(ranks)]
        
        elif selection_type == "truncation":
            m = max(1, int(np.ceil(truncation_frac * pop_size)))
            order = np.argsort(fit)
            top_idx = order[-m:]
            return pop[int(rng.choice(top_idx))]
        
        else: # tournament
            idx = rng.integers(0, pop_size, size=tour_k)
            f = fit[idx]
            return pop[idx[np.argmax(f)]]

    # Crossover
    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if rng.random() >= p_cx:
            return p1.copy()
        ctype = PARAMS.get("cx_type", "uniform")

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
        
        else: # uniform
            mask = rng.integers(0, 2, size=n, dtype=np.uint8)
            return (p1 & (1 - mask)) | (p2 & mask)

    # Mutation
    def mutate_bitflip(x: np.ndarray) -> np.ndarray:
        flips = rng.random(n) < p_mut
        return x ^ flips.astype(np.uint8)

    # GA loop
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

# Main execution

if __name__ == "__main__":
    # Load hyperparameters from best_params.json from tuning
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            cfg = json.load(f)
            best_cfg = cfg.get("best_config", {})
            print("Loaded tuned parameters from best_params.json:")
            print(best_cfg)
            for k, v in best_cfg.items():
                if k in PARAMS:
                    PARAMS[k] = v
    else:
        print("WARNING: No best_params.json found, using default parameters.")

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
            try:
                best_val = float(problem.state.best.y)
            except Exception:
                best_val = float(problem.state.current_best.y)
            best_vals.append(best_val)
            print(f"Run {r+1:02d} best_y = {best_val:.6f}")
        print(f"F{fid} summary -> median={np.median(best_vals):.4f}, mean={np.mean(best_vals):.4f}, min={np.min(best_vals):.4f}, max={np.max(best_vals):.4f}\n")
        log.close()
