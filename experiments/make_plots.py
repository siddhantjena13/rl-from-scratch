"""
Turn results/curves.csv into the plots the README embeds.

    python experiments/make_plots.py

Writes:
  results/learning_curves.png        - the four algorithms
  results/ablation_advantage.png     - PPO with TD vs GAE advantages
  results/ablation_normalization.png - REINFORCE with episode vs batch normalization

Raw episode returns on CartPole are far too noisy to read, so every curve is a
50-episode rolling mean, averaged across seeds, with a shaded band showing plus
and minus one standard deviation across seeds.
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # no display on a headless machine

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

WINDOW = 50


def load_curves(path):
    # label -> seed -> list of episode returns, in episode order
    curves = defaultdict(lambda: defaultdict(list))

    with open(path) as f:
        for row in csv.DictReader(f):
            curves[row["label"]][int(row["seed"])].append(float(row["return"]))

    return curves


def rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)

    if len(values) < window:
        return values

    # cumulative sums make this one subtraction per point instead of a slice
    # and a mean per point
    csum = np.concatenate([[0.0], np.cumsum(values)])
    return (csum[window:] - csum[:-window]) / window


def mean_and_std_across_seeds(seed_to_returns, window):
    smoothed = [rolling_mean(returns, window) for returns in seed_to_returns.values()]

    # runs are all the same length here, but truncate to the shortest anyway so
    # a partial run cannot silently break the stacking
    shortest = min(len(s) for s in smoothed)
    stacked = np.stack([s[:shortest] for s in smoothed])

    return np.arange(shortest) + window, stacked.mean(axis=0), stacked.std(axis=0)


def plot_group(curves, labels, title, out_path, colors=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, label in enumerate(labels):
        if label not in curves:
            print(f"  skipping {label!r}, not in curves.csv")
            continue

        episodes, mean, std = mean_and_std_across_seeds(curves[label], WINDOW)

        color = None if colors is None else colors[i]
        line, = ax.plot(episodes, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(episodes, mean - std, mean + std, color=line.get_color(), alpha=0.15)

    ax.axhline(475, linestyle="--", linewidth=1, color="grey")
    ax.text(20, 483, "solved (475)", fontsize=8, color="grey")

    ax.set_xlabel("episode")
    ax.set_ylabel(f"return ({WINDOW}-episode rolling mean)")
    ax.set_title(title)
    ax.set_ylim(0, 520)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    print(f"wrote {out_path}")


def main():
    curves_path = os.path.join(RESULTS_DIR, "curves.csv")

    if not os.path.exists(curves_path):
        raise SystemExit("no results/curves.csv - run experiments/run_all.py first")

    curves = load_curves(curves_path)
    print(f"loaded {len(curves)} configurations from {curves_path}")

    plot_group(
        curves,
        [
            "REINFORCE (batch norm)",
            "REINFORCE + baseline",
            "A2C",
            "PPO (GAE, lambda=0.95)",
        ],
        "CartPole-v1, mean of 3 seeds, +/- 1 std",
        os.path.join(RESULTS_DIR, "learning_curves.png"),
    )

    plot_group(
        curves,
        ["PPO (TD advantage)", "PPO (GAE, lambda=0.95)"],
        "PPO: single-step TD advantage vs GAE (lambda = 0.95)",
        os.path.join(RESULTS_DIR, "ablation_advantage.png"),
        colors=["tab:red", "tab:blue"],
    )

    plot_group(
        curves,
        ["REINFORCE (episode norm)", "REINFORCE (batch norm)"],
        "REINFORCE: per-episode vs per-batch advantage normalization",
        os.path.join(RESULTS_DIR, "ablation_normalization.png"),
        colors=["tab:orange", "tab:green"],
    )


if __name__ == "__main__":
    main()