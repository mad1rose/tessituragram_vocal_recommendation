"""
Visualizations for Research Question 3: Score Spread and Internal Validity.

Generates publication-ready figures for the methodology/results section.
Best choices for RQ3: (a) spread bar (variance and range with 95% CI) to show
scores discriminate; (b) correlations bar (Pearson r with 95% CI) to show
internal validity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / 'RQ3_results.json'


def load_results() -> dict:
    with open(RESULTS_PATH, encoding='utf-8') as f:
        return json.load(f)


def fig_spread(results: dict) -> None:
    """Bar chart of mean variance and mean range with 95% CI. Primary visual for RQ3a."""
    spread = results['metrics']['spread']
    mean_var = spread['mean_variance_final_score']
    ci_var = spread['ci_95_variance']
    mean_rng = spread['mean_range_final_score']
    ci_rng = spread['ci_95_range']
    M = results['data_summary']['n_profiles']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    # Variance (own scale)
    ax1.bar([0], [mean_var], color='#2e86ab', edgecolor='#1a5276', linewidth=1.2)
    ax1.errorbar([0], [mean_var], yerr=[[mean_var - ci_var[0]], [ci_var[1] - mean_var]],
                 fmt='none', color='#333', capsize=10, capthick=2)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Variance'])
    ax1.set_ylabel('Variance of final_score', fontsize=11)
    ax1.set_title(f"(a) Variance (M = {M})")
    ax1.annotate(f'{mean_var:.4f}\n[{ci_var[0]:.4f}–{ci_var[1]:.4f}]',
                 xy=(0, mean_var + 0.002), ha='center', va='bottom', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    # Range (own scale)
    ax2.bar([0], [mean_rng], color='#27ae60', edgecolor='#196f3d', linewidth=1.2)
    ax2.errorbar([0], [mean_rng], yerr=[[mean_rng - ci_rng[0]], [ci_rng[1] - mean_rng]],
                 fmt='none', color='#333', capsize=10, capthick=2)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Range'])
    ax2.set_ylabel('Range of final_score', fontsize=11)
    ax2.set_title(f"(b) Range (M = {M})")
    ax2.annotate(f'{mean_rng:.3f}\n[{ci_rng[0]:.3f}–{ci_rng[1]:.3f}]',
                 xy=(0, mean_rng + 0.03), ha='center', va='bottom', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    fig.suptitle("RQ3: Score Spread", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ3_spread.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: RQ3_spread.png")


def fig_correlations(results: dict) -> None:
    """Bar chart of Pearson r for the three pairs with 95% CI. Primary visual for RQ3b."""
    corr = results['metrics']['correlations']
    labels = [
        r'final_score vs cosine',
        r'final_score vs avoid_pen',
        r'cosine vs fav_overlap',
    ]
    keys = [
        'r_final_score_cosine_similarity',
        'r_final_score_avoid_penalty',
        'r_cosine_similarity_favorite_overlap',
    ]
    means = [corr[k]['mean'] for k in keys]
    cis = [corr[k]['ci_95'] for k in keys]
    yerr_lo = [means[i] - cis[i][0] for i in range(3)]
    yerr_hi = [cis[i][1] - means[i] for i in range(3)]
    expected_signs = ['+', '-', '+']
    colors = ['#2e86ab' if m >= 0 else '#c0392b' for m in means]  # blue for +, red for -
    M = results['data_summary']['n_profiles']

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax.bar(x, means, color=colors, edgecolor=['#1a5276', '#922b21', '#1a5276'], linewidth=1.2)
    ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi], fmt='none', color='#333', capsize=10, capthick=2)
    ax.axhline(y=0, color='#333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_ylim(-0.9, 1.1)
    ax.set_title(f"RQ3: Internal Validity — Correlations (M = {M} profiles)", fontsize=14)
    for i, (m, ci) in enumerate(zip(means, cis)):
        y_pos = m + 0.06 if m >= 0 else m - 0.12
        va = 'bottom' if m >= 0 else 'top'
        ax.annotate(f'r = {m:.3f}\n(expected {expected_signs[i]})',
                    xy=(i, y_pos), ha='center', va=va, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ3_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: RQ3_correlations.png")


def fig_variance_distribution(results: dict) -> None:
    """Histogram of variance across runs — shows spread of spread."""
    per_run = results['per_run']
    vars_final = [r['variance_final_score'] for r in per_run]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vars_final, bins=10, color='#2e86ab', edgecolor='#1a5276', alpha=0.85)
    ax.axvline(x=results['metrics']['spread']['mean_variance_final_score'],
               color='#c0392b', linestyle='-', linewidth=2,
               label=f"Mean = {results['metrics']['spread']['mean_variance_final_score']:.4f}")
    ax.set_xlabel('Variance of final_score (per run)', fontsize=12)
    ax.set_ylabel('Number of runs', fontsize=12)
    ax.set_title(f"RQ3: Distribution of Score Variance (n = {len(per_run)} runs)", fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ3_variance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: RQ3_variance_distribution.png")


def fig_combined(results: dict) -> None:
    """Combined figure: (a) spread, (b) correlations — for paper layout."""
    spread = results['metrics']['spread']
    corr = results['metrics']['correlations']
    M = results['data_summary']['n_profiles']

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    # Top-left: variance
    mean_var = spread['mean_variance_final_score']
    ci_var = spread['ci_95_variance']
    ax1a.bar([0], [mean_var], color='#2e86ab', edgecolor='#1a5276')
    ax1a.errorbar([0], [mean_var], yerr=[[mean_var - ci_var[0]], [ci_var[1] - mean_var]],
                  fmt='none', color='#333', capsize=8)
    ax1a.set_xticks([0])
    ax1a.set_xticklabels(['Variance'])
    ax1a.set_ylabel('Value')
    ax1a.set_title(f"(a) Variance (M = {M})")
    ax1a.grid(axis='y', alpha=0.3)

    # Top-right: range
    mean_rng = spread['mean_range_final_score']
    ci_rng = spread['ci_95_range']
    ax1b.bar([0], [mean_rng], color='#27ae60', edgecolor='#196f3d')
    ax1b.errorbar([0], [mean_rng], yerr=[[mean_rng - ci_rng[0]], [ci_rng[1] - mean_rng]],
                  fmt='none', color='#333', capsize=8)
    ax1b.set_xticks([0])
    ax1b.set_xticklabels(['Range'])
    ax1b.set_ylabel('Value')
    ax1b.set_title(f"(b) Range (M = {M})")
    ax1b.grid(axis='y', alpha=0.3)

    # Bottom-left: correlations
    labels2 = [r'final vs cosine', r'final vs avoid', r'cosine vs fav']
    keys2 = ['r_final_score_cosine_similarity', 'r_final_score_avoid_penalty',
             'r_cosine_similarity_favorite_overlap']
    means2 = [corr[k]['mean'] for k in keys2]
    cis2 = [corr[k]['ci_95'] for k in keys2]
    x2 = np.arange(3)
    colors2 = ['#2e86ab', '#c0392b', '#2e86ab']
    ax2.bar(x2, means2, color=colors2, edgecolor=['#1a5276', '#922b21', '#1a5276'])
    ax2.errorbar(x2, means2,
                 yerr=[[m - c[0] for m, c in zip(means2, cis2)],
                       [c[1] - m for m, c in zip(means2, cis2)]],
                 fmt='none', color='#333', capsize=8)
    ax2.axhline(y=0, color='#333', linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)
    ax2.set_ylabel("Pearson r")
    ax2.set_ylim(-0.9, 1.1)
    ax2.set_title(f"(c) Internal validity (M = {M})")
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle("RQ3: Score Spread and Internal Validity", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ3_visualizations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: RQ3_visualizations.png")


def main() -> None:
    print("Loading RQ3 results...")
    results = load_results()
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    print("Generating visualizations...")
    fig_spread(results)
    fig_correlations(results)
    fig_variance_distribution(results)
    fig_combined(results)
    print("Done.")


if __name__ == '__main__':
    main()
