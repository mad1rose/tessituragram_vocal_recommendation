"""
Research Question 3: Score spread and internal validity.

(a) Do recommendation scores (final_score) spread out across songs, or do
many songs get almost the same score?
(b) Do the parts of the score (cosine similarity and avoid penalty) relate
to final_score in the way the formula says (higher similarity → higher
score; higher avoid time → lower score)?

Metrics: mean variance and range of final_score; Pearson r for three pairs:
  final_score vs cosine_similarity (expect positive),
  final_score vs avoid_penalty (expect negative),
  cosine_similarity vs favorite_overlap (expect positive).
95% bootstrap CI over M runs (Herlocker et al., 2004; Urbano et al., 2013;
Castells et al., 2018).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

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
)

ALPHA = 0.5
TOP_N_FAV = 4
BOTTOM_N_AVOID = 2
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 42
MIN_CANDIDATES = 10  # Each profile must have ≥ 10 songs (evaluation plan §3.4)
N_PROFILES = 25  # M = 20–30 per evaluation plan; use 25 for stable estimates


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


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. Returns 0 if std is 0 (undefined)."""
    if len(x) < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_rq3_experiment(library_path: Path) -> dict:
    """Run the RQ3 score spread and internal validity experiment."""
    random.seed(RANDOM_SEED)
    all_songs = load_tessituragrams(library_path)

    profiles: list[tuple[dict, int, int, list[int], list[int]]] = []
    for song in all_songs:
        if len(profiles) >= N_PROFILES:
            break
        try:
            user_min, user_max, fav_midis, avoid_midis = _derive_synthetic_profile(song)
        except ValueError:
            continue
        filtered = filter_by_range(all_songs, user_min, user_max)
        if len(filtered) < MIN_CANDIDATES:
            continue
        profiles.append((song, user_min, user_max, fav_midis, avoid_midis))

    if not profiles:
        return {
            'experiment': 'RQ3_score_spread_internal_validity',
            'error': f'No song yielded ≥ {MIN_CANDIDATES} candidates. Library too small.',
            'data_summary': {'total_songs': len(all_songs)},
        }

    per_run: list[dict] = []
    for song, user_min, user_max, fav_midis, avoid_midis in profiles:
        filtered = filter_by_range(all_songs, user_min, user_max)
        ideal_vec = build_ideal_vector(user_min, user_max, fav_midis, avoid_midis)
        results = score_songs(
            filtered, ideal_vec, user_min, user_max,
            avoid_midis, fav_midis, alpha=ALPHA,
        )

        final_scores = np.array([r['final_score'] for r in results])
        cos_sims = np.array([r['cosine_similarity'] for r in results])
        avoid_pens = np.array([r['avoid_penalty'] for r in results])
        fav_overlaps = np.array([r['favorite_overlap'] for r in results])

        var_final = float(np.var(final_scores, ddof=1)) if len(final_scores) > 1 else 0.0
        range_final = float(np.max(final_scores) - np.min(final_scores))

        r_final_cos = _pearson_r(final_scores, cos_sims)
        r_final_avoid = _pearson_r(final_scores, avoid_pens)
        r_cos_fav = _pearson_r(cos_sims, fav_overlaps)

        per_run.append({
            'source_song': song.get('filename', ''),
            'composer': song.get('composer', ''),
            'n_songs': len(results),
            'variance_final_score': round(var_final, 6),
            'range_final_score': round(range_final, 4),
            'r_final_score_cosine': round(r_final_cos, 4),
            'r_final_score_avoid': round(r_final_avoid, 4),
            'r_cosine_favorite_overlap': round(r_cos_fav, 4),
        })

    M = len(per_run)

    vars_final = [r['variance_final_score'] for r in per_run]
    ranges_final = [r['range_final_score'] for r in per_run]
    r_fc = [r['r_final_score_cosine'] for r in per_run]
    r_fa = [r['r_final_score_avoid'] for r in per_run]
    r_cf = [r['r_cosine_favorite_overlap'] for r in per_run]

    mean_var = float(np.mean(vars_final))
    std_var = float(np.std(vars_final, ddof=1)) if M > 1 else 0.0
    mean_range = float(np.mean(ranges_final))
    std_range = float(np.std(ranges_final, ddof=1)) if M > 1 else 0.0
    mean_r_fc = float(np.mean(r_fc))
    mean_r_fa = float(np.mean(r_fa))
    mean_r_cf = float(np.mean(r_cf))

    def bootstrap_mean(values: list[float]) -> tuple[float, float]:
        boot = np.array([np.mean(random.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES)])
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    ci_var = bootstrap_mean(vars_final)
    ci_range = bootstrap_mean(ranges_final)
    ci_r_fc = bootstrap_mean(r_fc)
    ci_r_fa = bootstrap_mean(r_fa)
    ci_r_cf = bootstrap_mean(r_cf)

    return {
        'experiment': 'RQ3_score_spread_internal_validity',
        'description': '(a) Spread of final_score (variance, range). (b) Internal validity: correlations between score parts and final_score.',
        'parameters': {
            'alpha': ALPHA,
            'top_n_favorite': TOP_N_FAV,
            'bottom_n_avoid': BOTTOM_N_AVOID,
            'min_candidates': MIN_CANDIDATES,
            'n_profiles': N_PROFILES,
            'bootstrap_samples': BOOTSTRAP_SAMPLES,
            'random_seed': RANDOM_SEED,
            'library_path': str(Path('data/tessituragrams.json')),
        },
        'data_summary': {
            'total_songs_in_library': len(all_songs),
            'n_profiles': M,
            'profiles_used': M,
        },
        'metrics': {
            'spread': {
                'mean_variance_final_score': round(mean_var, 6),
                'std_variance_final_score': round(std_var, 6),
                'ci_95_variance': [round(ci_var[0], 6), round(ci_var[1], 6)],
                'mean_range_final_score': round(mean_range, 4),
                'std_range_final_score': round(std_range, 4),
                'ci_95_range': [round(ci_range[0], 4), round(ci_range[1], 4)],
            },
            'correlations': {
                'r_final_score_cosine_similarity': {
                    'mean': round(mean_r_fc, 4),
                    'ci_95': [round(ci_r_fc[0], 4), round(ci_r_fc[1], 4)],
                    'expected_sign': 'positive',
                },
                'r_final_score_avoid_penalty': {
                    'mean': round(mean_r_fa, 4),
                    'ci_95': [round(ci_r_fa[0], 4), round(ci_r_fa[1], 4)],
                    'expected_sign': 'negative',
                },
                'r_cosine_similarity_favorite_overlap': {
                    'mean': round(mean_r_cf, 4),
                    'ci_95': [round(ci_r_cf[0], 4), round(ci_r_cf[1], 4)],
                    'expected_sign': 'positive',
                },
            },
        },
        'expected_signs': {
            'final_score–cosine': 'positive (higher similarity → higher score)',
            'final_score–avoid_penalty': 'negative (higher avoid time → lower score)',
            'cosine–favorite_overlap': 'positive (both reflect similarity to ideal)',
        },
        'per_run': per_run,
    }


def main() -> None:
    library_path = ROOT / 'data' / 'tessituragrams.json'
    out_dir = Path(__file__).resolve().parent

    print("Running RQ3 Score Spread and Internal Validity Experiment...")
    print(f"Library: {library_path}")

    results = run_rq3_experiment(library_path)

    out_json = out_dir / 'RQ3_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_json}")

    if 'error' in results:
        print(f"\nError: {results['error']}")
        return

    m = results['metrics']
    spread = m['spread']
    corr = m['correlations']
    print("\n--- Spread ---")
    print(f"Mean variance (final_score): {spread['mean_variance_final_score']}  [95% CI: {spread['ci_95_variance']}]")
    print(f"Mean range (final_score):    {spread['mean_range_final_score']}  [95% CI: {spread['ci_95_range']}]")
    print("\n--- Correlations (internal validity) ---")
    print(f"r(final_score, cosine_sim):  {corr['r_final_score_cosine_similarity']['mean']}  [95% CI: {corr['r_final_score_cosine_similarity']['ci_95']}]  (expected: +)")
    print(f"r(final_score, avoid_pen):   {corr['r_final_score_avoid_penalty']['mean']}  [95% CI: {corr['r_final_score_avoid_penalty']['ci_95']}]  (expected: -)")
    print(f"r(cosine_sim, fav_overlap):  {corr['r_cosine_similarity_favorite_overlap']['mean']}  [95% CI: {corr['r_cosine_similarity_favorite_overlap']['ci_95']}]  (expected: +)")
    print(f"\nProfiles: {results['data_summary']['n_profiles']}")


if __name__ == '__main__':
    main()
