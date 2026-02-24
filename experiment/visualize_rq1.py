"""
Visualizations for Research Question 1: Self-Retrieval Accuracy.

Generates publication-ready figures for the methodology/results section.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / 'RQ1_results.json'


def load_results() -> dict:
    with open(RESULTS_PATH, encoding='utf-8') as f:
        return json.load(f)


def fig_metrics_bar(results: dict) -> None:
    """Bar chart of HR@1, HR@3, HR@5, MRR with 95% CI error bars."""
    m = results['metrics']
    labels = ['HR@1', 'HR@3', 'HR@5', 'MRR']
    values = [m['HR@1']['value'], m['HR@3']['value'], m['HR@5']['value'], m['MRR']['value']]
    ci_lo = [m['HR@1']['ci_95'][0], m['HR@3']['ci_95'][0], m['HR@5']['ci_95'][0], m['MRR']['ci_95'][0]]
    ci_hi = [m['HR@1']['ci_95'][1], m['HR@3']['ci_95'][1], m['HR@5']['ci_95'][1], m['MRR']['ci_95'][1]]
    yerr_lo = [v - lo for v, lo in zip(values, ci_lo)]
    yerr_hi = [hi - v for v, hi in zip(values, ci_hi)]
    yerr = [yerr_lo, yerr_hi]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(labels))
    bars = ax.bar(x, values, color=['#2e86ab', '#a23b72', '#f18f01', '#3b1f2b'], edgecolor='#333', linewidth=1.2)
    ax.errorbar(x, values, yerr=yerr, fmt='none', color='#333', capsize=6, capthick=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title('RQ1: Self-Retrieval Accuracy Metrics (95% CI)', fontsize=14)
    ax.axhline(y=1.0, color='#ccc', linestyle='--', alpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    for i, (v, lo, hi) in enumerate(zip(values, ci_lo, ci_hi)):
        ax.annotate(f'{v:.2f}\n[{lo:.2f}–{hi:.2f}]', xy=(i, v + 0.02), ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ1_metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ1_metrics_bar.png'}")


def fig_rank_distribution(results: dict) -> None:
    """Histogram of query song ranks."""
    ranks = [r['rank'] for r in results['per_query']]
    n = len(ranks)
    max_rank = max(ranks)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = list(range(0, max_rank + 2))
    ax.hist(ranks, bins=bins, color='#2e86ab', edgecolor='#1a5276', alpha=0.85)
    ax.set_xlabel('Rank of Query Song', fontsize=12)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_title(f'RQ1: Distribution of Query Song Ranks (n = {n})', fontsize=14)
    ax.set_xticks(range(1, max_rank + 1))
    ax.grid(axis='y', alpha=0.3)
    # Add count labels on bars
    for r in range(1, max_rank + 1):
        c = ranks.count(r)
        if c > 0:
            ax.annotate(str(c), xy=(r, c), ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ1_rank_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ1_rank_distribution.png'}")


def fig_cumulative_hit_rate(results: dict) -> None:
    """Cumulative hit rate: fraction of queries at rank <= k for k = 1,2,...,max."""
    ranks = [r['rank'] for r in results['per_query']]
    n = len(ranks)
    max_rank = max(ranks)

    cumul = []
    for k in range(1, max_rank + 1):
        cumul.append(sum(1 for r in ranks if r <= k) / n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, max_rank + 1), cumul, 'o-', color='#2e86ab', linewidth=2, markersize=8)
    ax.set_xlabel('Rank k (top-k)', fontsize=12)
    ax.set_ylabel('Fraction of queries with rank ≤ k', fontsize=12)
    ax.set_title(f'RQ1: Cumulative Hit Rate (n = {n})', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(y=1.0, color='#ccc', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ1_cumulative_hit_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ1_cumulative_hit_rate.png'}")


def fig_combined(results: dict) -> None:
    """Combined figure: metrics bar + rank distribution in one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: metrics bar
    m = results['metrics']
    labels = ['HR@1', 'HR@3', 'HR@5', 'MRR']
    values = [m['HR@1']['value'], m['HR@3']['value'], m['HR@5']['value'], m['MRR']['value']]
    ci_lo = [m[l]['ci_95'][0] for l in ['HR@1', 'HR@3', 'HR@5', 'MRR']]
    ci_hi = [m[l]['ci_95'][1] for l in ['HR@1', 'HR@3', 'HR@5', 'MRR']]
    yerr = [[v - lo for v, lo in zip(values, ci_lo)], [hi - v for v, hi in zip(values, ci_hi)]]
    x = range(len(labels))
    ax1.bar(x, values, color=['#2e86ab', '#a23b72', '#f18f01', '#3b1f2b'], edgecolor='#333', linewidth=1.2)
    ax1.errorbar(x, values, yerr=yerr, fmt='none', color='#333', capsize=6, capthick=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.15)
    ax1.set_title('(a) Accuracy Metrics (95% CI)')
    ax1.grid(axis='y', alpha=0.3)

    # Right: rank distribution
    ranks = [r['rank'] for r in results['per_query']]
    max_rank = max(ranks)
    ax2.hist(ranks, bins=range(0, max_rank + 2), color='#2e86ab', edgecolor='#1a5276', alpha=0.85)
    ax2.set_xlabel('Rank of Query Song')
    ax2.set_ylabel('Number of Queries')
    ax2.set_title(f'(b) Distribution of Ranks (n = {len(ranks)})')
    ax2.set_xticks(range(1, max_rank + 1))
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('RQ1: Self-Retrieval Accuracy', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ1_visualizations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ1_visualizations.png'}")


def main() -> None:
    print("Loading RQ1 results...")
    results = load_results()
    print("Generating visualizations...")
    fig_metrics_bar(results)
    fig_rank_distribution(results)
    fig_cumulative_hit_rate(results)
    fig_combined(results)
    print("Done.")


if __name__ == '__main__':
    main()
