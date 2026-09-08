"""
Run every algorithm across several seeds and write the results out.

Produces three things:
  results/runs.csv    - one row per run, the summary numbers
  results/curves.csv  - one row per episode, for the learning curve plots
  a markdown table on stdout, ready to paste into the README

Every number in the README should come from here rather than from a terminal
scrollback, so that anyone can rerun this and get the same table.

    python experiments/run_all.py
    python experiments/run_all.py --quick     (fewer batches and seeds, for a smoke test)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# so that `python experiments/run_all.py` can find the algorithm files
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "algorithms"))

import a2c
import baseline
import ppo
import reinforce


RESULTS_DIR = os.path.join(REPO_ROOT, "results")

SEEDS = [0, 1, 2]
QUICK_SEEDS = [0]
NUM_BATCHES = 200
QUICK_NUM_BATCHES = 20


# (label, module, extra kwargs). the label is what shows up in the table, so it
# has to say what is actually being varied - the module's own algorithm name is
# not enough once one file can run two configurations.
CONFIGS = [
    ("REINFORCE (batch norm)", reinforce, {"normalization": "batch"}),
    ("REINFORCE (episode norm)", reinforce, {"normalization": "episode"}),
    ("REINFORCE + baseline", baseline, {"normalization": "batch"}),
    ("A2C", a2c, {"normalization": "batch"}),
    ("PPO (TD advantage)", ppo, {"normalization": "batch", "advantage_estimator": "td"}),
    ("PPO (GAE, lambda=0.95)", ppo, {"normalization": "batch", "advantage_estimator": "gae"}),
]


def run_all(seeds, num_batches):
    runs = []
    curves = []

    for label, module, kwargs in CONFIGS:
        for seed in seeds:
            print(f"running {label}, seed {seed} ... ", end="", flush=True)
            start = time.time()

            result = module.train(seed=seed, num_batches=num_batches, verbose=False, **kwargs)

            elapsed = time.time() - start
            solved = result["episodes_to_solve"]

            runs.append({
                "label": label,
                "algorithm": result["algorithm"],
                "normalization": result["normalization"],
                "advantage": kwargs.get("advantage_estimator", "mc"),
                "seed": seed,
                "episodes_to_solve": solved if solved is not None else "",
                "episodes_trained": len(result["episode_returns"]),
                "eval_final_mean": round(result["eval_final_mean"], 2),
                "eval_final_std": round(result["eval_final_std"], 2),
                "eval_best_mean": round(result["eval_best_mean"], 2),
                "eval_best_std": round(result["eval_best_std"], 2),
                "wall_seconds": round(elapsed, 1),
            })

            for episode, total_reward in enumerate(result["episode_returns"]):
                curves.append((label, seed, episode, total_reward))

            print(
                f"solved at {solved if solved is not None else 'never':>6}, "
                f"final eval {result['eval_final_mean']:6.1f}, {elapsed:.0f}s"
            )

    return runs, curves


def write_runs_csv(runs, path):
    fieldnames = list(runs[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)


def write_curves_csv(curves, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "seed", "episode", "return"])
        writer.writerows(curves)


def format_table(runs, seeds):
    lines = []
    lines.append("| Algorithm | Solved | Episodes to solve | Final eval | Best eval |")
    lines.append("| --- | --- | --- | --- | --- |")

    for label, _, _ in CONFIGS:
        rows = [r for r in runs if r["label"] == label]

        # a run that never reaches the threshold has no episodes-to-solve to
        # average, so the count of runs that solved carries information the mean
        # cannot. reporting only the mean over the solved runs would quietly
        # hide the failures.
        solved_counts = [r["episodes_to_solve"] for r in rows if r["episodes_to_solve"] != ""]

        if solved_counts:
            solve_str = f"{np.mean(solved_counts):.0f} +/- {np.std(solved_counts):.0f}"
        else:
            solve_str = "-"

        final = [r["eval_final_mean"] for r in rows]
        best = [r["eval_best_mean"] for r in rows]

        lines.append(
            f"| {label} "
            f"| {len(solved_counts)}/{len(seeds)} "
            f"| {solve_str} "
            f"| {np.mean(final):.0f} +/- {np.std(final):.0f} "
            f"| {np.mean(best):.0f} +/- {np.std(best):.0f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="run every algorithm across seeds")
    parser.add_argument("--quick", action="store_true", help="short run, one seed, for a smoke test")
    args = parser.parse_args()

    seeds = QUICK_SEEDS if args.quick else SEEDS
    num_batches = QUICK_NUM_BATCHES if args.quick else NUM_BATCHES

    os.makedirs(RESULTS_DIR, exist_ok=True)

    runs, curves = run_all(seeds, num_batches)

    runs_path = os.path.join(RESULTS_DIR, "runs.csv")
    curves_path = os.path.join(RESULTS_DIR, "curves.csv")

    write_runs_csv(runs, runs_path)
    write_curves_csv(curves, curves_path)

    print()
    print(f"wrote {runs_path}")
    print(f"wrote {curves_path}")
    print()
    print(f"CartPole-v1, {len(seeds)} seeds, {num_batches} batches of 10 episodes.")
    print("Solved = 50-episode average >= 475. Eval = 20 greedy episodes.")
    print()
    print(format_table(runs, seeds))


if __name__ == "__main__":
    main()