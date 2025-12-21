# tuning.py
# -----------------------------------------------------------------------------
# Hyperparameter tuning for GA.py with a strict global budget of 100,000 evaluations.
# -----------------------------------------------------------------------------
# This script searches for the best GA hyperparameters for BOTH problems:
#   - F18 (LABS, bitstring autocorrelation task)
#   - F23 (N-Queens, bit-encoded placement task)
# Each problem has very different scales, so we normalize scores before comparing.
#
# Updated normalization:
# Based on known optimum scores:
#   - F18 (LABS, n=50) optimum = 8.17
#   - F23 (N-Queens, n=49) optimum = 7
# We normalize each problem's score relative to [0, optimum].
# -----------------------------------------------------------------------------

from typing import List
from pathlib import Path
import json
import numpy as np
from GA import set_params, studentnumber1_studentnumber2_GA, create_problem

# Exhaustive grid search budget is computed from SPACE, RUNS_PER_PROBLEM, and EVALS_PER_RUN.
budget = 0

# Tuning plan
K = 0                  # computed after building the exhaustive grid
RUNS_PER_PROBLEM = 5   # per candidate per problem
EVALS_PER_RUN = 1000    # budget used in each tuning run
SEED_BASE = 42         # base seed for reproducibility

# Hyperparameters to tune
SPACE = dict(
    pop_size=[30, 50, 70],
    p_cx=[0.5, 0.6, 0.7],
    mut_per_n=[1.0, 2.0],
    elitism=[1, 3],
    cx_type=["k_point"], # "uniform", "one_point", "k_point"
    cx_k=[1, 2, 3, 4],
    selection_type=["truncation"], # "tournament", "proportional", "rank", "truncation"
    tour_k=[3],
    truncation_frac=[0.3, 0.5],
    init_type=["biased"], # "random", "biased", "complementary"
    init_p=[0.3, 0.5, 0.7],
    replacement_type=["elitism"], # "elitism", "generational"
)

# Known optimum values for normalization
OPTIMUM_F18 = 8.17
OPTIMUM_F23 = 7.0 # n=49 means 7x7 board so optimal solution is 7

def _best_y(problem):
    try:
        return float(problem.state.best.y)
    except Exception:
        return float(problem.state.current_best.y)

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
            vals.append(_best_y(prob))
            prob.reset()
            log.close()
        scores.append(float(np.median(vals)))
    return scores[0], scores[1]

def _normalize_known_optima(s18_all, s23_all):
    # Clip to non-negative and divide by known optima
    s18_norm = np.clip(s18_all / OPTIMUM_F18, 0, 1)
    s23_norm = np.clip(s23_all / OPTIMUM_F23, 0, 1)
    return s18_norm, s23_norm

def tune_hyperparameters() -> List:
    global K, budget
    rng = np.random.default_rng(2025)
    candidates = _enumerate_candidates()
    K = len(candidates)
    total_evals = K * 2 * RUNS_PER_PROBLEM * EVALS_PER_RUN
    budget = total_evals

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

    def _convert(o):
        import numpy as _np
        if isinstance(o, (_np.integer, _np.int_)): return int(o)
        if isinstance(o, (_np.floating, _np.float64)): return float(o)
        if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_convert(v) for v in o]
        return o

    Path("best_params.json").write_text(json.dumps(_convert(out), indent=2))

    return [best_cfg.get("pop_size"), best_cfg.get("mut_per_n"), best_cfg.get("p_cx")]

if __name__ == "__main__":
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print("\n=== Suggested hyperparameters ===")
    print("Population size:", population_size)
    print("Mutation rate:", mutation_rate)
    print("Crossover rate:", crossover_rate)
    print("Full best config written to best_params.json")
