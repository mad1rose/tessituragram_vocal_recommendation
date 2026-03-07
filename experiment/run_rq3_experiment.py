"""
Research Question 3: Score spread and formula checks.

This experiment answers: (a) Do final_score values spread out across songs?
(b) Does the scoring formula behave as specified? We separate two kinds of
checks:

  - **Formula / identity checks** (strong): Verify that final_score equals
    cos_sim - alpha*avoid_penalty numerically (mean/max absolute residual),
    and optionally per-run regression final ~ cos + avoid (coefficients ≈ 1,
    -alpha; R² ≈ 1). These are the primary evidence that the formula behaves
    as intended.

  - **Sanity-check correlations** (descriptive): Pearson r for final_score vs
    cosine_similarity and final_score vs avoid_penalty (linear by construction),
    and Spearman rho for cosine_similarity vs favorite_overlap (bounded,
    possibly non-linear). Reported as pipeline sanity checks, not as
    "internal validity" in the strong sense.

Implementation details:
- Variance, range, and correlations are computed on **unrounded** score values
  (same formula as the app; no rounding before analysis).
- **Spearman** for cos vs favorite_overlap; **Pearson** for final vs cos/avoid.
- **Undefined correlations** (zero std): return NaN, exclude from aggregation,
  report n_excluded per correlation.
- **Fisher z** aggregation: atanh(r) with r clamped to (-1, 1) to avoid
  atanh(±1) undefined; average z (optionally weighted by n_songs − 3);
  back-transform with tanh.
- **Random** profile selection (seeded); each profile must have ≥ MIN_CANDIDATES.
- **Bootstrap**: 95% CI by resampling runs (profiles). The unit of analysis
  is the run/profile. A hierarchical bootstrap (resample profiles then songs
  within profile) would additionally reflect song-level variability and could
  be considered for future work if reviewers request it.

Metrics: spread (variance, range); sanity-check correlations (with 95% CI);
identity residual; regression summary. Output: experiment_results/RQ3_results.json.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Allow running from project root or experiment folder
ROOT = Path(__file__).resolve().parent.parent
_sys_path = list(__import__("sys").path)
if str(ROOT) not in _sys_path:
    __import__("sys").path.insert(0, str(ROOT))

from src.storage import load_tessituragrams
from src.recommend import (
    filter_by_range,
    build_ideal_vector,
    build_dense_vector,
    normalize_l1,
    cosine_similarity,
)

ALPHA = 0.5
TOP_N_FAV = 4
BOTTOM_N_AVOID = 2
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 42
MIN_CANDIDATES = 10
N_PROFILES = 25


def _derive_synthetic_profile(song: dict) -> tuple[int, int, list[int], list[int]]:
    """Same rule as RQ1: top-4 fav, bottom-2 avoid (disjoint)."""
    pr = song.get("statistics", {}).get("pitch_range", {})
    user_min = pr.get("min_midi")
    user_max = pr.get("max_midi")
    if user_min is None or user_max is None:
        raise ValueError(f"Song {song.get('filename')} has no pitch range")
    tess = song.get("tessituragram", {})
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
        if n_pitches >= BOTTOM_N_AVOID
        else []
    )
    avoid_midis = [m for m in avoid_candidates if m not in favorite_midis]
    return user_min, user_max, favorite_midis, avoid_midis


def _pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; return NaN if undefined (len < 2 or zero std)."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r) if not np.isnan(r) else float("nan")


def _spearman_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; return NaN if undefined."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r) if not np.isnan(r) else float("nan")


# Clamp r for Fisher z so atanh(r) is defined (atanh(±1) is undefined).
_FISHER_R_CLAMP = 0.999999


def _fisher_z_mean(r_values: list[float], n_songs_per_run: list[int] | None = None) -> float:
    """
    Aggregate correlations via Fisher z: atanh(r), average (optionally weighted
    by n-3), then tanh. r is clamped to (-_FISHER_R_CLAMP, _FISHER_R_CLAMP) so
    atanh(r) is always defined. Ignores NaN. Returns NaN if no valid r.
    """
    def _z(r: float) -> float:
        r_clamped = max(-_FISHER_R_CLAMP, min(_FISHER_R_CLAMP, r))
        return math.atanh(r_clamped)

    if n_songs_per_run is not None and len(n_songs_per_run) == len(r_values):
        pairs = [(r, n) for r, n in zip(r_values, n_songs_per_run) if not math.isnan(r) and n > 3]
        if not pairs:
            return float("nan")
        weights = [n - 3 for _, n in pairs]
        z_sum = sum(w * _z(r) for (r, _), w in zip(pairs, weights))
        z_mean = z_sum / sum(weights)
    else:
        valid = [r for r in r_values if not math.isnan(r)]
        if not valid:
            return float("nan")
        z_mean = sum(_z(r) for r in valid) / len(valid)
    return math.tanh(z_mean)


def _compute_run_stats(
    filtered_songs: list[dict],
    ideal_vec: np.ndarray,
    min_midi: int,
    max_midi: int,
    avoid_midis: list[int],
    favorite_midis: list[int],
) -> dict:
    """
    Compute per-run statistics using **unrounded** scores (same formula as
    score_songs but no rounding). Returns variance, range, correlations,
    identity residual, and optional regression summary.
    """
    avoid_indices = [m - min_midi for m in avoid_midis if min_midi <= m <= max_midi]
    fav_indices = [m - min_midi for m in favorite_midis if min_midi <= m <= max_midi]

    final_scores = []
    cos_sims = []
    avoid_pens = []
    fav_overlaps = []

    for song in filtered_songs:
        tess = song.get("tessituragram", {})
        dense = build_dense_vector(tess, min_midi, max_midi)
        normed = normalize_l1(dense)
        cos_sim = cosine_similarity(normed, ideal_vec)
        avoid_penalty = float(sum(normed[i] for i in avoid_indices)) if avoid_indices else 0.0
        final_score = cos_sim - ALPHA * avoid_penalty
        fav_overlap = float(sum(normed[i] for i in fav_indices)) if fav_indices else 0.0
        final_scores.append(final_score)
        cos_sims.append(cos_sim)
        avoid_pens.append(avoid_penalty)
        fav_overlaps.append(fav_overlap)

    final_scores = np.array(final_scores)
    cos_sims = np.array(cos_sims)
    avoid_pens = np.array(avoid_pens)
    fav_overlaps = np.array(fav_overlaps)
    n = len(final_scores)

    var_final = float(np.var(final_scores, ddof=1)) if n > 1 else 0.0
    range_final = float(np.max(final_scores) - np.min(final_scores))

    r_final_cos = _pearson_or_nan(final_scores, cos_sims)
    r_final_avoid = _pearson_or_nan(final_scores, avoid_pens)
    r_cos_fav = _spearman_or_nan(cos_sims, fav_overlaps)

    # Identity check: final should equal cos - alpha*avoid (numerically)
    reconstructed = cos_sims - ALPHA * avoid_pens
    residuals = np.abs(final_scores - reconstructed)
    mean_abs_residual = float(np.mean(residuals))
    max_abs_residual = float(np.max(residuals))

    # Per-run regression: final ~ cos + avoid (expect coef_cos≈1, coef_avoid≈-alpha, R²≈1)
    if n >= 3 and np.std(cos_sims) > 0 and np.std(avoid_pens) > 0:
        X = np.column_stack([cos_sims, avoid_pens])
        X = np.column_stack([np.ones(n), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, final_scores, rcond=None)
            pred = X @ beta
            ss_res = np.sum((final_scores - pred) ** 2)
            ss_tot = np.sum((final_scores - np.mean(final_scores)) ** 2)
            r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
            regress = {
                "intercept": float(beta[0]),
                "coef_cos": float(beta[1]),
                "coef_avoid": float(beta[2]),
                "r_squared": float(r_sq),
            }
        except Exception:
            regress = None
    else:
        regress = None

    return {
        "n_songs": n,
        "variance_final_score": var_final,
        "range_final_score": range_final,
        "r_final_score_cosine": r_final_cos,
        "r_final_score_avoid": r_final_avoid,
        "r_cosine_favorite_overlap": r_cos_fav,
        "identity_mean_abs_residual": mean_abs_residual,
        "identity_max_abs_residual": max_abs_residual,
        "regression": regress,
    }


def _select_profiles(
    all_songs: list[dict],
) -> list[tuple[dict, int, int, list[int], list[int]]]:
    """Collect all valid profiles, then sample N_PROFILES at random (seeded)."""
    candidates = []
    for song in all_songs:
        try:
            user_min, user_max, fav_midis, avoid_midis = _derive_synthetic_profile(song)
        except ValueError:
            continue
        filtered = filter_by_range(all_songs, user_min, user_max)
        if len(filtered) >= MIN_CANDIDATES:
            candidates.append((song, user_min, user_max, fav_midis, avoid_midis))
    if not candidates:
        return []
    n = min(N_PROFILES, len(candidates))
    return random.sample(candidates, n)


def run_rq3_experiment(library_path: Path) -> dict:
    """Run RQ3: unrounded stats, Fisher z, Spearman for cos–overlap, identity check, random profiles."""
    random.seed(RANDOM_SEED)
    all_songs = load_tessituragrams(library_path)

    profiles = _select_profiles(all_songs)
    if not profiles:
        return {
            "experiment": "RQ3_score_spread_formula_checks",
            "error": f"No song yielded ≥ {MIN_CANDIDATES} candidates. Library too small.",
            "data_summary": {"total_songs_in_library": len(all_songs)},
        }

    per_run: list[dict] = []
    for song, user_min, user_max, fav_midis, avoid_midis in profiles:
        filtered = filter_by_range(all_songs, user_min, user_max)
        ideal_vec = build_ideal_vector(user_min, user_max, fav_midis, avoid_midis)
        stats = _compute_run_stats(
            filtered, ideal_vec, user_min, user_max, avoid_midis, fav_midis
        )
        per_run.append({
            "source_song": song.get("filename", ""),
            "composer": song.get("composer", ""),
            **stats,
        })

    M = len(per_run)
    n_songs_list = [r["n_songs"] for r in per_run]
    vars_final = [r["variance_final_score"] for r in per_run]
    ranges_final = [r["range_final_score"] for r in per_run]
    r_fc = [r["r_final_score_cosine"] for r in per_run]
    r_fa = [r["r_final_score_avoid"] for r in per_run]
    r_cf = [r["r_cosine_favorite_overlap"] for r in per_run]
    id_resid_mean = [r["identity_mean_abs_residual"] for r in per_run]
    id_resid_max = [r["identity_max_abs_residual"] for r in per_run]

    n_excluded_fc = sum(1 for r in r_fc if math.isnan(r))
    n_excluded_fa = sum(1 for r in r_fa if math.isnan(r))
    n_excluded_cf = sum(1 for r in r_cf if math.isnan(r))

    mean_var = float(np.mean(vars_final))
    std_var = float(np.std(vars_final, ddof=1)) if M > 1 else 0.0
    mean_range = float(np.mean(ranges_final))
    std_range = float(np.std(ranges_final, ddof=1)) if M > 1 else 0.0

    mean_r_fc = _fisher_z_mean(r_fc, n_songs_list)
    mean_r_fa = _fisher_z_mean(r_fa, n_songs_list)
    mean_r_cf = _fisher_z_mean(r_cf, n_songs_list)

    mean_id_resid = float(np.mean(id_resid_mean))
    max_id_resid_over_runs = float(np.max(id_resid_max))

    regressions = [r["regression"] for r in per_run if r.get("regression") is not None]
    if regressions:
        mean_coef_cos = float(np.mean([rg["coef_cos"] for rg in regressions]))
        mean_coef_avoid = float(np.mean([rg["coef_avoid"] for rg in regressions]))
        mean_r_sq = float(np.mean([rg["r_squared"] for rg in regressions]))
        n_regressions = len(regressions)
    else:
        mean_coef_cos = mean_coef_avoid = mean_r_sq = float("nan")
        n_regressions = 0

    def bootstrap_mean(values: list[float]) -> tuple[float, float]:
        boot = np.array([np.mean(random.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES)])
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    def bootstrap_fisher(r_list: list[float], n_list: list[int]) -> tuple[float, float]:
        boot_means = []
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = random.choices(range(len(r_list)), k=len(r_list))
            r_b = [r_list[i] for i in idx]
            n_b = [n_list[i] for i in idx]
            m = _fisher_z_mean(r_b, n_b)
            if not math.isnan(m):
                boot_means.append(m)
        if not boot_means:
            return float("nan"), float("nan")
        boot_means = np.array(boot_means)
        return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    ci_var = bootstrap_mean(vars_final)
    ci_range = bootstrap_mean(ranges_final)
    ci_r_fc = bootstrap_fisher(r_fc, n_songs_list)
    ci_r_fa = bootstrap_fisher(r_fa, n_songs_list)
    ci_r_cf = bootstrap_fisher(r_cf, n_songs_list)

    def safe_round(x: float, ndigits: int) -> float:
        if math.isnan(x):
            return x
        return round(x, ndigits)

    return {
        "experiment": "RQ3_score_spread_formula_checks",
        "description": (
            "(a) Spread of final_score (variance, range) on unrounded values. "
            "(b) Sanity-check correlations (Pearson for final–cos/avoid, Spearman for cos–overlap); "
            "identity check final ≈ cos − α×avoid; optional regression summary. "
            "Random profiles; Fisher z aggregation; undefined correlations excluded."
        ),
        "parameters": {
            "alpha": ALPHA,
            "top_n_favorite": TOP_N_FAV,
            "bottom_n_avoid": BOTTOM_N_AVOID,
            "min_candidates": MIN_CANDIDATES,
            "n_profiles": N_PROFILES,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "random_seed": RANDOM_SEED,
            "library_path": str(Path("data/tessituragrams.json")),
        },
        "data_summary": {
            "total_songs_in_library": len(all_songs),
            "n_profiles": M,
            "n_excluded_r_final_cos": n_excluded_fc,
            "n_excluded_r_final_avoid": n_excluded_fa,
            "n_excluded_r_cos_fav": n_excluded_cf,
            "n_runs_with_regression": n_regressions,
        },
        "metrics": {
            "spread": {
                "mean_variance_final_score": safe_round(mean_var, 6),
                "std_variance_final_score": safe_round(std_var, 6),
                "ci_95_variance": [safe_round(ci_var[0], 6), safe_round(ci_var[1], 6)],
                "mean_range_final_score": safe_round(mean_range, 4),
                "std_range_final_score": safe_round(std_range, 4),
                "ci_95_range": [safe_round(ci_range[0], 4), safe_round(ci_range[1], 4)],
            },
            "correlations_sanity_check": {
                "r_final_score_cosine_similarity": {
                    "mean_fisher": safe_round(mean_r_fc, 4),
                    "ci_95": [safe_round(ci_r_fc[0], 4), safe_round(ci_r_fc[1], 4)],
                    "expected_sign": "positive",
                    "n_excluded_undefined": n_excluded_fc,
                },
                "r_final_score_avoid_penalty": {
                    "mean_fisher": safe_round(mean_r_fa, 4),
                    "ci_95": [safe_round(ci_r_fa[0], 4), safe_round(ci_r_fa[1], 4)],
                    "expected_sign": "negative",
                    "n_excluded_undefined": n_excluded_fa,
                },
                "r_cosine_similarity_favorite_overlap_spearman": {
                    "mean_fisher": safe_round(mean_r_cf, 4),
                    "ci_95": [safe_round(ci_r_cf[0], 4), safe_round(ci_r_cf[1], 4)],
                    "expected_sign": "positive",
                    "n_excluded_undefined": n_excluded_cf,
                },
            },
            "identity_check": {
                "mean_abs_residual_final_vs_cos_minus_alpha_avoid": safe_round(mean_id_resid, 8),
                "max_abs_residual_over_runs": safe_round(max_id_resid_over_runs, 8),
            },
            "regression_summary": {
                "mean_coef_cosine": safe_round(mean_coef_cos, 4),
                "mean_coef_avoid": safe_round(mean_coef_avoid, 4),
                "mean_R_squared": safe_round(mean_r_sq, 4),
                "n_runs": n_regressions,
            },
        },
        "expected_signs": {
            "final_score–cosine": "positive (sanity check: formula)",
            "final_score–avoid_penalty": "negative (sanity check: formula)",
            "cosine–favorite_overlap": "positive (Spearman; monotonic association)",
        },
        "per_run": [
            {
                "source_song": r["source_song"],
                "composer": r["composer"],
                "n_songs": r["n_songs"],
                "variance_final_score": round(r["variance_final_score"], 6),
                "range_final_score": round(r["range_final_score"], 4),
                "r_final_score_cosine": round(r["r_final_score_cosine"], 4) if not math.isnan(r["r_final_score_cosine"]) else None,
                "r_final_score_avoid": round(r["r_final_score_avoid"], 4) if not math.isnan(r["r_final_score_avoid"]) else None,
                "r_cosine_favorite_overlap": round(r["r_cosine_favorite_overlap"], 4) if not math.isnan(r["r_cosine_favorite_overlap"]) else None,
                "identity_mean_abs_residual": r["identity_mean_abs_residual"],
                "identity_max_abs_residual": r["identity_max_abs_residual"],
                "regression": r.get("regression"),
            }
            for r in per_run
        ],
    }


def _nan_to_none(obj):  # JSON does not support NaN
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main() -> None:
    library_path = ROOT / "data" / "tessituragrams.json"
    out_dir = ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)

    print("Running RQ3 Score Spread and Formula Checks...")
    print(f"Library: {library_path}")

    results = run_rq3_experiment(library_path)

    out_json = out_dir / "RQ3_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(results), f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_json}")

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    m = results["metrics"]
    spread = m["spread"]
    corr = m["correlations_sanity_check"]
    ident = m["identity_check"]
    reg = m["regression_summary"]
    ds = results["data_summary"]

    print("\n--- Spread (unrounded values) ---")
    print(f"Mean variance (final_score): {spread['mean_variance_final_score']}  [95% CI: {spread['ci_95_variance']}]")
    print(f"Mean range (final_score):    {spread['mean_range_final_score']}  [95% CI: {spread['ci_95_range']}]")
    print("\n--- Correlations (Fisher z; sanity checks) ---")
    print(f"r(final_score, cosine_sim) [Pearson]:  {corr['r_final_score_cosine_similarity']['mean_fisher']}  [95% CI: {corr['r_final_score_cosine_similarity']['ci_95']}]  excluded: {corr['r_final_score_cosine_similarity']['n_excluded_undefined']}")
    print(f"r(final_score, avoid_pen) [Pearson]:   {corr['r_final_score_avoid_penalty']['mean_fisher']}  [95% CI: {corr['r_final_score_avoid_penalty']['ci_95']}]  excluded: {corr['r_final_score_avoid_penalty']['n_excluded_undefined']}")
    print(f"r(cosine_sim, fav_overlap) [Spearman]: {corr['r_cosine_similarity_favorite_overlap_spearman']['mean_fisher']}  [95% CI: {corr['r_cosine_similarity_favorite_overlap_spearman']['ci_95']}]  excluded: {corr['r_cosine_similarity_favorite_overlap_spearman']['n_excluded_undefined']}")
    print("\n--- Identity check: final vs (cos - alpha*avoid) ---")
    print(f"Mean |residual|: {ident['mean_abs_residual_final_vs_cos_minus_alpha_avoid']}")
    print(f"Max |residual| over runs: {ident['max_abs_residual_over_runs']}")
    print("\n--- Regression summary (final ~ cos + avoid) ---")
    print(f"Mean coef_cos: {reg['mean_coef_cosine']}  (expect ~1); mean coef_avoid: {reg['mean_coef_avoid']}  (expect ~-{ALPHA})")
    print(f"Mean R²: {reg['mean_R_squared']}  (n_runs: {reg['n_runs']})")
    print(f"\nProfiles: {ds['n_profiles']}")


if __name__ == "__main__":
    main()
