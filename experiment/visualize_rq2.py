"""
Visualizations for Research Question 2: Ranking Stability (Kendall's tau).

Generates publication-ready figures for the methodology/results section.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / 'experiment_results'
RESULTS_PATH = EXP_DIR / 'RQ2_results.json'


def load_results() -> dict:
    with open(RESULTS_PATH, encoding='utf-8') as f:
        return json.load(f)


def _get_n_perturbations(results: dict) -> int:
    return results['data_summary'].get('total_perturbations') or results['data_summary'].get('number_of_perturbations', 0)


def fig_tau_bar(results: dict) -> None:
    """Bar chart of mean tau with 95% CI. Primary visual for RQ2."""
    m = results['metrics']
    mean_tau = m['mean_tau']
    ci_lo, ci_hi = m['ci_95']
    yerr = [[mean_tau - ci_lo], [ci_hi - mean_tau]]
    n = _get_n_perturbations(results)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([0], [mean_tau], color='#2e86ab', edgecolor='#1a5276', linewidth=1.2)
    ax.errorbar([0], [mean_tau], yerr=yerr, fmt='none', color='#333', capsize=12, capthick=2)
    ax.set_xticks([0])
    ax.set_xticklabels([r"Mean $\tau$"])
    ax.set_ylabel(r"Kendall's $\tau$", fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0.7, color='#888', linestyle='--', alpha=0.8, label=r'$\tau = 0.7$ (strong)')
    ax.axhline(y=0.3, color='#aaa', linestyle=':', alpha=0.6, label=r'$\tau = 0.3$ (moderate)')
    ax.set_title(f"RQ2: Ranking Stability (n = {n} perturbations)", fontsize=14)
    ax.annotate(f'{mean_tau:.3f}\n[{ci_lo:.3f}–{ci_hi:.3f}]', xy=(0, mean_tau + 0.03), ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ2_tau_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ2_tau_bar.png'}")


def fig_tau_distribution(results: dict) -> None:
    """Histogram of tau values across perturbations."""
    taus = [p['tau'] for p in results['per_perturbation']]
    n = len(taus)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(taus, bins=15, color='#2e86ab', edgecolor='#1a5276', alpha=0.85)
    ax.axvline(x=results['metrics']['mean_tau'], color='#c0392b', linestyle='-', linewidth=2, label=f"Mean = {results['metrics']['mean_tau']:.3f}")
    ax.axvline(x=0.7, color='#888', linestyle='--', alpha=0.8, label=r'$\tau = 0.7$ (strong)')
    ax.set_xlabel(r"Kendall's $\tau$", fontsize=12)
    ax.set_ylabel('Number of perturbations', fontsize=12)
    ax.set_title(f"RQ2: Distribution of Kendall's tau (n = {n} perturbations)", fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ2_tau_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ2_tau_distribution.png'}")


def fig_tau_by_type(results: dict) -> None:
    """Strip plot of tau by perturbation type. Strip plot preferred over box for small n (e.g. remove_avoid has 2)."""
    per_pert = results['per_perturbation']
    types = {}
    for p in per_pert:
        t = p['perturbation_type']
        if t not in types:
            types[t] = []
        types[t].append(p['tau'])

    labels = ['add_fav', 'remove_fav', 'add_avoid', 'remove_avoid']
    labels = [l for l in labels if l in types]
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)  # reproducible jitter
    for i, lbl in enumerate(labels):
        taus = types[lbl]
        x = rng.normal(i, 0.04, size=len(taus))
        ax.scatter(x, taus, alpha=0.7, s=40, color='#2e86ab', edgecolor='#1a5276')
        ax.annotate(f'n={len(taus)}', xy=(i, -0.08), ha='center', fontsize=9)
    ax.axhline(y=0.7, color='#888', linestyle='--', alpha=0.8, label=r'$\tau = 0.7$ (strong)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"Kendall's $\tau$", fontsize=12)
    ax.set_title("RQ2: tau by perturbation type (strip plot)", fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.15, 1.05)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ2_tau_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ2_tau_by_type.png'}")


def fig_combined(results: dict) -> None:
    """Combined figure: tau bar + distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean tau bar
    m = results['metrics']
    mean_tau = m['mean_tau']
    ci_lo, ci_hi = m['ci_95']
    ax1.bar([0], [mean_tau], color='#2e86ab', edgecolor='#1a5276')
    ax1.errorbar([0], [mean_tau], yerr=[[mean_tau - ci_lo], [ci_hi - mean_tau]], fmt='none', color='#333', capsize=10)
    ax1.set_xticks([0])
    ax1.set_xticklabels([r"Mean $\tau$"])
    ax1.set_ylabel(r"Kendall's $\tau$")
    ax1.set_ylim(-0.1, 1.1)
    ax1.axhline(y=0.7, color='#888', linestyle='--', alpha=0.7)
    ax1.set_title("(a) Mean tau with 95% CI")
    ax1.grid(axis='y', alpha=0.3)

    # Right: histogram
    taus = [p['tau'] for p in results['per_perturbation']]
    ax2.hist(taus, bins=12, color='#2e86ab', edgecolor='#1a5276', alpha=0.85)
    ax2.axvline(x=mean_tau, color='#c0392b', linestyle='-', linewidth=2, label=f'Mean = {mean_tau:.3f}')
    ax2.axvline(x=0.7, color='#888', linestyle='--', alpha=0.7)
    ax2.set_xlabel(r"Kendall's $\tau$")
    ax2.set_ylabel('Count')
    ax2.set_title(f"(b) Distribution of tau (n = {len(taus)} perturbations)")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle("RQ2: Ranking Stability Under One-Note Changes", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'RQ2_visualizations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {EXP_DIR / 'RQ2_visualizations.png'}")


def main() -> None:
    print("Loading RQ2 results...")
    results = load_results()
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    print("Generating visualizations...")
    fig_tau_bar(results)
    fig_tau_distribution(results)
    fig_tau_by_type(results)
    fig_combined(results)
    print("Done.")


if __name__ == '__main__':
    main()
