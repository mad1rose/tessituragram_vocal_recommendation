"""
Research Question 1: Self-retrieval accuracy experiment.

When we build a synthetic user profile from one song (using that song's
most- and least-used pitches as favourites and avoids), does the system
rank that same song at position 1, or within the top 3 or top 5?

Metrics: HR@1, MRR, HR@3, HR@5 with 95% bootstrap confidence intervals.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

# Allow running from project root or from experiment folder
ROOT = Path(__file__).resolve().parent.parent
sys_path = list(__import__('sys').path)
if str(ROOT) not in sys_path:
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
RANDOM_SEED = 42  # For reproducible bootstrap; Urbano et al. (2013) stress reproducibility


def _derive_synthetic_profile(song: dict) -> tuple[int, int, list[int], list[int]]:
    """
    Derive synthetic user profile from a song.

    - User range = song's pitch range (min_midi, max_midi).
    - Favorites = top 4 MIDI by duration (L1-normalised); use all if < 4 pitches.
    - Avoids = bottom 2 MIDI by duration; use none if < 2 pitches.

    Returns (user_min, user_max, favorite_midis, avoid_midis).
    """
    pr = song.get('statistics', {}).get('pitch_range', {})
    user_min = pr.get('min_midi')
    user_max = pr.get('max_midi')
    if user_min is None or user_max is None:
        raise ValueError(f"Song {song.get('filename')} has no pitch range")

    tess = song.get('tessituragram', {})
    if not tess:
        raise ValueError(f"Song {song.get('filename')} has empty tessituragram")

    # L1-normalise: proportion of singing time per pitch
    total = sum(tess.values())
    if total <= 0:
        raise ValueError(f"Song {song.get('filename')} has zero total duration")
    proportions = [(int(midi), dur / total) for midi, dur in tess.items()]

    # Top by duration = favorites; bottom = avoids. Secondary sort by MIDI for
    # deterministic tie-breaking when durations are equal (reproducibility).
    sorted_by_duration = sorted(proportions, key=lambda x: (-x[1], x[0]))
    n_pitches = len(sorted_by_duration)

    favorite_midis = [m for m, _ in sorted_by_duration[: min(TOP_N_FAV, n_pitches)]]
    avoid_candidates = (
        [m for m, _ in sorted_by_duration[-BOTTOM_N_AVOID:]]
        if n_pitches >= BOTTOM_N_AVOID
        else []
    )
    # Ensure favorites and avoids are disjoint (avoid penalising a pitch we boost)
    avoid_midis = [m for m in avoid_candidates if m not in favorite_midis]

    return user_min, user_max, favorite_midis, avoid_midis


def run_rq1_experiment(library_path: Path) -> dict:
    """Run the RQ1 self-retrieval experiment. Returns full results dict."""
    random.seed(RANDOM_SEED)
    all_songs = load_tessituragrams(library_path)

    records: list[dict] = []

    for song in all_songs:
        filename = song.get('filename', '')
        try:
            user_min, user_max, fav_midis, avoid_midis = _derive_synthetic_profile(song)
        except ValueError:
            continue

        filtered = filter_by_range(all_songs, user_min, user_max)
        if len(filtered) < 2:
            continue

        ideal_vec = build_ideal_vector(user_min, user_max, fav_midis, avoid_midis)
        results = score_songs(
            filtered, ideal_vec, user_min, user_max,
            avoid_midis, fav_midis, alpha=ALPHA,
        )

        rank = None
        for r in results:
            if r['filename'] == filename:
                rank = r['rank']
                break

        if rank is None:
            continue  # should not happen if query song is in filtered set

        hit1 = 1 if rank == 1 else 0
        hit3 = 1 if rank <= 3 else 0
        hit5 = 1 if rank <= 5 else 0
        mrr = 1.0 / rank

        records.append({
            'filename': filename,
            'composer': song.get('composer'),
            'title': song.get('title'),
            'rank': rank,
            'hit@1': hit1,
            'hit@3': hit3,
            'hit@5': hit5,
            '1/rank': mrr,
        })

    n = len(records)

    # Aggregate metrics
    hr1 = np.mean([r['hit@1'] for r in records]) if n else 0.0
    hr3 = np.mean([r['hit@3'] for r in records]) if n else 0.0
    hr5 = np.mean([r['hit@5'] for r in records]) if n else 0.0
    mrr = np.mean([r['1/rank'] for r in records]) if n else 0.0

    # Bootstrap 95% CI (percentile method; Urbano et al., 2013; Efron & Tibshirani)
    def bootstrap_mean(values: list[float], n_samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
        boot = np.array([np.mean(random.choices(values, k=len(values))) for _ in range(n_samples)])
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        return lo, hi

    hr1_ci = bootstrap_mean([r['hit@1'] for r in records]) if n else (0.0, 0.0)
    hr3_ci = bootstrap_mean([r['hit@3'] for r in records]) if n else (0.0, 0.0)
    hr5_ci = bootstrap_mean([r['hit@5'] for r in records]) if n else (0.0, 0.0)
    mrr_ci = bootstrap_mean([r['1/rank'] for r in records]) if n else (0.0, 0.0)

    return {
        'experiment': 'RQ1_self_retrieval_accuracy',
        'description': 'When a synthetic user profile is derived from one song, does the system rank that song at position 1 or in the top 3/5?',
        'parameters': {
            'alpha': ALPHA,
            'top_n_favorite': TOP_N_FAV,
            'bottom_n_avoid': BOTTOM_N_AVOID,
            'bootstrap_samples': BOOTSTRAP_SAMPLES,
            'random_seed': RANDOM_SEED,
            'library_path': str(Path('data/tessituragrams.json')),
        },
        'data_summary': {
            'total_songs_in_library': len(all_songs),
            'valid_queries': n,
            'songs_skipped': len(all_songs) - n,
        },
        'metrics': {
            'HR@1': {'value': round(hr1, 4), 'ci_95': [round(hr1_ci[0], 4), round(hr1_ci[1], 4)]},
            'HR@3': {'value': round(hr3, 4), 'ci_95': [round(hr3_ci[0], 4), round(hr3_ci[1], 4)]},
            'HR@5': {'value': round(hr5, 4), 'ci_95': [round(hr5_ci[0], 4), round(hr5_ci[1], 4)]},
            'MRR': {'value': round(mrr, 4), 'ci_95': [round(mrr_ci[0], 4), round(mrr_ci[1], 4)]},
        },
        'formulas': {
            'HR@1': 'fraction of queries where the query song was ranked 1',
            'HR@3': 'fraction of queries where the query song was in top 3',
            'HR@5': 'fraction of queries where the query song was in top 5',
            'MRR': 'mean of 1/rank across queries (1 = best, rewards higher ranks)',
        },
        'per_query': records,
    }


def main() -> None:
    library_path = ROOT / 'data' / 'tessituragrams.json'
    out_dir = Path(__file__).resolve().parent

    print("Running RQ1 Self-Retrieval Accuracy Experiment...")
    print(f"Library: {library_path}")

    results = run_rq1_experiment(library_path)

    out_json = out_dir / 'RQ1_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_json}")

    # Summary
    m = results['metrics']
    print("\n--- Metrics ---")
    print(f"HR@1: {m['HR@1']['value']}  [95% CI: {m['HR@1']['ci_95']}]")
    print(f"HR@3: {m['HR@3']['value']}  [95% CI: {m['HR@3']['ci_95']}]")
    print(f"HR@5: {m['HR@5']['value']}  [95% CI: {m['HR@5']['ci_95']}]")
    print(f"MRR:  {m['MRR']['value']}  [95% CI: {m['MRR']['ci_95']}]")
    print(f"\nValid queries: {results['data_summary']['valid_queries']} / {results['data_summary']['total_songs_in_library']}")


if __name__ == '__main__':
    main()
