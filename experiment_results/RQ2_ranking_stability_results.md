# RQ2: Ranking Stability Under One-Note Preference Changes

## Overview

This experiment tests whether the recommendation system remains **stable** when a user changes their preferences by exactly one note—adding or removing a favourite or an avoid. If the system is stable, small input changes should not cause large reorderings of the recommendation list.

## Methodology

- **Metric**: Kendall's τ (tau) between the original ranking R₀ and the ranking after each one-note perturbation. τ = 1 means identical order; τ = 0 means no relationship; τ = −1 means completely reversed.
- **Baselines**: 5 synthetic user profiles, each derived from a different song (top-4 pitches by duration as favourites, bottom-2 as avoids, disjoint). Baselines were **sampled at random** among all songs with ≥10 candidates (seed 42).
- **Perturbations**: For each baseline, we try every one-note change that keeps favourites and avoids disjoint. Perturbations are restricted to **notes that actually occur** in at least one song in the candidate set (musically relevant notes only).
- **Unit of analysis**: The **baseline** is the unit for confidence intervals. We compute mean τ per baseline, then bootstrap over baseline means (10,000 samples) to obtain a 95% CI.

## Results

| Metric | Value |
|--------|-------|
| Mean τ (all perturbations) | 0.849 |
| Std τ (all perturbations) | 0.114 |
| Mean τ per baseline | 0.848 |
| Std τ across baselines | 0.044 |
| **95% CI (baseline-level mean)** | **[0.808, 0.875]** |
| Baselines used | 5 |
| Total perturbations | 130 |

### Baseline Profiles

| Source Song | Composer | Perturbations | Mean τ |
|-------------|----------|---------------|--------|
| debussy-claude-3-chansons-de-france-no2-la-grotte... | Claude Debussy | 28 | 0.857 |
| schumann-clara-6-lieder-op13-no5-ich-hab-in-deinem-auge | Clara Schumann | 24 | 0.886 |
| schubert-franz-die-schone-mullerin-d795-no7-ungeduld | Franz Schubert | 30 | 0.861 |
| schubert-franz-die-schone-mullerin-d795-no3-halt | Franz Schubert | 24 | 0.864 |
| schubert-franz-die-schone-mullerin-d795-no18-trockne-blumen | Franz Schubert | 24 | 0.773 |

## Interpretation

- **τ > 0.7** → strong agreement (rankings very similar)
- **0.3 ≤ τ ≤ 0.7** → moderate agreement
- **τ < 0.3** → weak agreement

The mean τ of **0.848** (95% CI [0.808, 0.875]) falls in the **strong agreement** range. When a user adds or removes a single favourite or avoid note, the recommendation list stays largely the same—the system is **stable** under small preference changes.

The lowest baseline mean (0.773 for *Trockne Blumen*) still indicates strong stability; some perturbations (e.g. removing a heavily-used favourite) cause more reordering than others, but overall the rankings remain highly correlated.

## Parameters

- α = 0.5
- Top 4 pitches by duration = favourites; bottom 2 = avoids (disjoint)
- Min candidates per baseline: 10
- Bootstrap samples: 10,000
- Random seed: 42
