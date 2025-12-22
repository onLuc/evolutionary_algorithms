from typing import List
from pathlib import Path
import json
import random
import numpy as np
from GA import set_params, studentnumber1_studentnumber2_GA, create_problem

budget = 100_000 # 100_000 for assignment

# Tuning plan
K = 10                  # K configurations to evaluate (randomly sampled from SPACE)
RUNS_PER_PROBLEM = 5   # per configuration per problem
EVALS_PER_RUN = 1000   # budget used in each run
SEED_BASE = 42         # random seed

# Hyperparameter search space
SPACE = dict(
    pop_size=[20, 30, 50],                  # population size
    p_cx=[0.7, 0.9],                     # crossover probability
    mut_per_n=[1.0],                # mutation rate (per n)
    elitism=[3],                    # number of elite individuals
    cx_type=["k_point"],            # "uniform", "k_point"
    cx_k=[1, 2],                       # for k-point crossover
    selection_type=["truncation", "tournament"],  # "tournament", "proportional", "rank", "truncation"
    tour_k=[3, 5],                     # for tournament selection
    truncation_frac=[0.5],          # for truncation selection
    init_type=["biased"],           # "random", "biased"
    init_p=[0.4],                   # for biased initialization
    replacement_type=["elitism"],   # "elitism", "generational"
)

def _enumerate_candidates():
    keys = list(SPACE.keys())
    cand = []

    def _recurse(i, cfg):
        if i == len(keys):
            if cfg["elitism"] >= cfg["pop_size"]:
                return
            if cfg["replacement_type"] == "generational":
                if cfg["elitism"] != 0:
                    return
            cfg_out = dict(cfg)
            if cfg_out["selection_type"] != "tournament":
                cfg_out.pop("tour_k", None)
            if cfg_out["selection_type"] != "truncation":
                cfg_out.pop("truncation_frac", None)
            if cfg_out["cx_type"] != "k_point":
                cfg_out.pop("cx_k", None)
            if cfg_out["init_type"] != "biased":
                cfg_out.pop("init_p", None)
            cand.append(cfg_out)
            return

        k = keys[i]
        for v in SPACE[k]:
            cfg[k] = v
            _recurse(i + 1, cfg)
        cfg.pop(k, None)

    _recurse(0, {})
    return cand

def evaluate_config(cfg, rng):
    """Return (median_score_F18, median_score_F23) at 1,000 evaluations per run."""
    scores = []
    for fid, dim in [(18, 50), (23, 49)]:
        vals = []
        for r in range(RUNS_PER_PROBLEM):
            prob, log = create_problem(dimension=dim, fid=fid)
            set_params(**cfg, budget=EVALS_PER_RUN, seed=SEED_BASE + r)
            studentnumber1_studentnumber2_GA(prob)
            try:
                vals.append(prob.state.best.y)
            except Exception:
                vals.append(prob.state.current_best.y)
            prob.reset()
            log.close()
        scores.append(float(np.median(vals)))
    return scores[0], scores[1]

def _normalize_known_optima(s18_all, s23_all):
    # Clip to non-negative and divide by known optima
    s18_norm = np.clip(s18_all / 8.17, 0, 1)
    s23_norm = np.clip(s23_all / 7.0, 0, 1)
    return s18_norm, s23_norm

def tune_hyperparameters() -> List:
    global K, budget
    rng = np.random.default_rng(SEED_BASE)
    random.seed(SEED_BASE)

    total_evals = K * 2 * RUNS_PER_PROBLEM * EVALS_PER_RUN

    if total_evals > budget:
        raise ValueError(f"Total evaluations {total_evals} exceed budget {budget}")

    candidates = random.sample(_enumerate_candidates(), K)
    results = []
    for i, cfg in enumerate(candidates, 1):
        s18, s23 = evaluate_config(cfg, rng)
        results.append((cfg, s18, s23))
        print(f"[{i:02d}/{K}] cfg={cfg}  ->  F18={s18:.6f}, F23={s23:.6f}")

    s18_all = np.array([r[1] for r in results], dtype=float)
    s23_all = np.array([r[2] for r in results], dtype=float)

    # Normalize using known optimum values
    z18, z23 = _normalize_known_optima(s18_all, s23_all)

    combined = 0.5 * (z18 + z23)
    best_i = int(np.argmax(combined))
    best_cfg = results[best_i][0]

    out = {
        "best_config": best_cfg,
        "scores": {"F18_median": float(results[best_i][1]), "F23_median": float(results[best_i][2])},
        "plan": {
            "K": K,
            "RUNS_PER_PROBLEM": RUNS_PER_PROBLEM,
            "EVALS_PER_RUN": EVALS_PER_RUN,
            "total_evaluations": total_evals,
        },
    }

    Path("best_params.json").write_text(json.dumps(out, indent=2))

    return [best_cfg.get("pop_size"), best_cfg.get("mut_per_n"), best_cfg.get("p_cx")]

if __name__ == "__main__":
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print("\n=== Suggested hyperparameters ===")
    print("Population size:", population_size)
    print("Mutation rate:", mutation_rate)
    print("Crossover rate:", crossover_rate)
    print("Full best config written to best_params.json")
