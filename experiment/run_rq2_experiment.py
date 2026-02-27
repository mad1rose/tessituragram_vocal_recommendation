"""
Research Question 2: Ranking stability under small preference changes.

When we change the user's favourite or avoid list by exactly one note
(add one or remove one), how similar is the new recommendation list to
the original? We measure similarity with Kendall's τ (tau).

Metrics: mean τ, std, 95% bootstrap CI. τ ∈ [−1, 1]; τ > 0.7 strong,
0.3–0.7 moderate, < 0.3 weak (Kendall, 1948; evaluation plan).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

# Allow running from project root or experiment folder
ROOT = Path(__file__).resolve().parent.parent
_sys_path = list(__import__('sys').path)
if str(ROOT) not in _sys_path:
    __import__('sys').path.insert(0, str(ROOT))

from src.storage import load_tessituragrams
from src.recommend import (
    filter_by_range,
    build_ideal_vector,
    score_songs,
    midi_to_note_name,
)

ALPHA = 0.5
TOP_N_FAV = 4
BOTTOM_N_AVOID = 2
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 42
MIN_CANDIDATES = 10  # Candidate set C must have ≥ 10 songs
N_BASELINES = 5  # Use 5 baselines to reduce risk of atypical baseline (eval plan §2.4)


def _derive_synthetic_profile(song: dict) -> tuple[int, int, list[int], list[int]]:
    """Same rule as RQ1: top-4 fav, bottom-2 avoid (disjoint)."""
    pr = song.get('statistics', {}).get('pitch_range', {})
    user_min = pr.get('min_midi')
    user_max = pr.get('max_midi')
    if user_min is None or user_max is None:
        raise ValueError(f"Song {song.get('filename')} has no pitch range")

    tess = song.get('tessituragram', {})
    if not tess:
        raise ValueError(f"Song {song.get('filename')} has empty tessituragram")
    total = sum(tess.values())
    if total <= 0:
        raise ValueError(f"Song {song.get('filename')} has zero total duration")

    proportions = [(int(midi), dur / total) for midi, dur in tess.items()]
    sorted_by_duration = sorted(proportions, key=lambda x: (-x[1], x[0]))
    n_pitches = len(sorted_by_duration)

    favorite_midis = [m for m, _ in sorted_by_duration[: min(TOP_N_FAV, n_pitches)]]
    avoid_candidates = (
        [m for m, _ in sorted_by_duration[-BOTTOM_N_AVOID:]]
        if n_pitches >= BOTTOM_N_AVOID else []
    )
    avoid_midis = [m for m in avoid_candidates if m not in favorite_midis]
    return user_min, user_max, favorite_midis, avoid_midis


def _compute_kendall_tau(ranking_r0: list[dict], ranking_r_new: list[dict]) -> float:
    """
    Compute Kendall's τ between two rankings of the same set of songs.
    Both rankings must contain the same filenames. Returns τ ∈ [−1, 1].
    """
    filenames = [r['filename'] for r in ranking_r0]
    assert set(filenames) == set(r['filename'] for r in ranking_r_new)

    rank_r0 = {r['filename']: r['rank'] for r in ranking_r0}
    rank_r_new = {r['filename']: r['rank'] for r in ranking_r_new}
    r0_vec = np.array([rank_r0[f] for f in filenames])
    r_new_vec = np.array([rank_r_new[f] for f in filenames])

    tau, _ = kendalltau(r0_vec, r_new_vec)
    return float(tau) if not np.isnan(tau) else 0.0


def _run_one_baseline(
    all_songs: list[dict],
    user_min: int,
    user_max: int,
    fav_midis: list[int],
    avoid_midis: list[int],
) -> tuple[list[float], list[dict]]:
    """Run perturbations for one baseline. Returns (tau_values, per_perturbation)."""
    filtered = filter_by_range(all_songs, user_min, user_max)
    ideal_vec = build_ideal_vector(user_min, user_max, fav_midis, avoid_midis)
    ranking_r0 = score_songs(
        filtered, ideal_vec, user_min, user_max,
        avoid_midis, fav_midis, alpha=ALPHA,
    )
    perturbations: list[tuple[str, int, list[int], list[int]]] = []
    midi_in_range = set(range(user_min, user_max + 1))
    for m in midi_in_range:
        if m not in fav_midis:
            perturbations.append(('add_fav', m, fav_midis + [m], avoid_midis))
    for i, m in enumerate(fav_midis):
        new_fav = fav_midis[:i] + fav_midis[i + 1:]
        perturbations.append(('remove_fav', m, new_fav, avoid_midis))
    for m in midi_in_range:
        if m not in avoid_midis and m not in fav_midis:
            perturbations.append(('add_avoid', m, fav_midis, avoid_midis + [m]))
    for i, m in enumerate(avoid_midis):
        new_avoid = avoid_midis[:i] + avoid_midis[i + 1:]
        perturbations.append(('remove_avoid', m, fav_midis, new_avoid))

    tau_values: list[float] = []
    per_pert: list[dict] = []
    for pert_type, midi_changed, new_fav, new_avoid in perturbations:
        ideal_new = build_ideal_vector(user_min, user_max, new_fav, new_avoid)
        ranking_new = score_songs(
            filtered, ideal_new, user_min, user_max,
            new_avoid, new_fav, alpha=ALPHA,
        )
        tau = _compute_kendall_tau(ranking_r0, ranking_new)
        tau_values.append(tau)
        per_pert.append({
            'perturbation_type': pert_type,
            'midi_changed': midi_changed,
            'note_changed': midi_to_note_name(midi_changed),
            'tau': round(tau, 4),
        })
    return tau_values, per_pert


def run_rq2_experiment(library_path: Path) -> dict:
    """Run the RQ2 ranking stability experiment. Uses N_BASELINES profiles."""
    random.seed(RANDOM_SEED)
    all_songs = load_tessituragrams(library_path)

    # Find N_BASELINES baseline profiles with ≥ MIN_CANDIDATES songs
    baselines: list[tuple[dict, int, int, list[int], list[int]]] = []
    for song in all_songs:
        if len(baselines) >= N_BASELINES:
            break
        try:
            user_min, user_max, fav_midis, avoid_midis = _derive_synthetic_profile(song)
        except ValueError:
            continue
        cand = filter_by_range(all_songs, user_min, user_max)
        if len(cand) >= MIN_CANDIDATES:
            baselines.append((song, user_min, user_max, fav_midis, avoid_midis))

    if not baselines:
        return {
            'experiment': 'RQ2_ranking_stability',
            'error': f'No song yielded ≥ {MIN_CANDIDATES} candidates. Library too small.',
            'data_summary': {'total_songs': len(all_songs)},
        }

    all_tau_values: list[float] = []
    all_per_perturbation: list[dict] = []
    per_baseline: list[dict] = []

    for song, user_min, user_max, fav_midis, avoid_midis in baselines:
        tau_vals, per_pert = _run_one_baseline(
            all_songs, user_min, user_max, fav_midis, avoid_midis,
        )
        all_tau_values.extend(tau_vals)
        for p in per_pert:
            p2 = dict(p)
            p2['baseline_source'] = song.get('filename', '')
            all_per_perturbation.append(p2)
        mean_b = float(np.mean(tau_vals))
        per_baseline.append({
            'source_song': song.get('filename', ''),
            'composer': song.get('composer', ''),
            'n_perturbations': len(tau_vals),
            'mean_tau': round(mean_b, 4),
        })

    n = len(all_tau_values)
    mean_tau = float(np.mean(all_tau_values)) if n else 0.0
    std_tau = float(np.std(all_tau_values, ddof=1)) if n > 1 else 0.0

    def bootstrap_mean(values: list[float]) -> tuple[float, float]:
        boot = np.array([np.mean(random.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES)])
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    tau_ci = bootstrap_mean(all_tau_values) if n else (0.0, 0.0)

    mean_across_baselines = float(np.mean([b['mean_tau'] for b in per_baseline]))
    std_across_baselines = float(np.std([b['mean_tau'] for b in per_baseline])) if len(per_baseline) > 1 else 0.0

    return {
        'experiment': 'RQ2_ranking_stability',
        'description': 'When we change favourites or avoids by one note, how similar is the new ranking to the original? (Kendall\'s τ)',
        'parameters': {
            'alpha': ALPHA,
            'top_n_favorite': TOP_N_FAV,
            'bottom_n_avoid': BOTTOM_N_AVOID,
            'min_candidates': MIN_CANDIDATES,
            'n_baselines': N_BASELINES,
            'bootstrap_samples': BOOTSTRAP_SAMPLES,
            'random_seed': RANDOM_SEED,
            'library_path': str(Path('data/tessituragrams.json')),
        },
        'baseline_profiles': per_baseline,
        'data_summary': {
            'total_songs_in_library': len(all_songs),
            'n_baselines': len(baselines),
            'total_perturbations': n,
        },
        'metrics': {
            'mean_tau': round(mean_tau, 4),
            'std_tau': round(std_tau, 4),
            'ci_95': [round(tau_ci[0], 4), round(tau_ci[1], 4)],
            'mean_tau_per_baseline': round(mean_across_baselines, 4),
            'std_tau_across_baselines': round(std_across_baselines, 4),
        },
        'interpretation': {
            'tau_gt_0.7': 'strong agreement (rankings very similar)',
            'tau_0.3_to_0.7': 'moderate agreement',
            'tau_lt_0.3': 'weak agreement',
        },
        'per_perturbation': all_per_perturbation,
    }


def main() -> None:
    library_path = ROOT / 'data' / 'tessituragrams.json'
    out_dir = ROOT / 'experiment_results'
    out_dir.mkdir(exist_ok=True)

    print("Running RQ2 Ranking Stability Experiment...")
    print(f"Library: {library_path}")

    results = run_rq2_experiment(library_path)

    out_json = out_dir / 'RQ2_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_json}")

    if 'error' in results:
        print(f"\nError: {results['error']}")
        return

    m = results['metrics']
    print("\n--- Metrics ---")
    print(f"Mean tau: {m['mean_tau']}  (std: {m['std_tau']})")
    print(f"95% CI: {m['ci_95']}")
    ds = results['data_summary']
    print(f"\nBaselines: {ds.get('n_baselines', 1)}")
    print(f"Total perturbations: {ds.get('total_perturbations', ds.get('number_of_perturbations', 0))}")


if __name__ == '__main__':
    main()
