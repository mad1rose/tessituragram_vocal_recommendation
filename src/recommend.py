"""Core recommendation engine: filtering, vectorisation, scoring, ranking."""

from __future__ import annotations

import numpy as np
from music21 import pitch as m21pitch

# ── Helpers: note-name / MIDI conversion ────────────────────────────────────

# Standard note names for each pitch class (0-11), used for MIDI→name display.
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def midi_to_note_name(midi: int) -> str:
    """Convert a MIDI number to a readable note name, e.g. 60 → 'C4'."""
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def note_name_to_midi(name: str) -> int:
    """
    Convert a human-readable note name to a MIDI number.

    Accepts any format music21 understands:
        C4, F#4, Bb3, Eb5, G#5, D-4 (flat via '-'), …

    Raises ValueError on unparseable input.
    """
    name = name.strip()
    try:
        p = m21pitch.Pitch(name)
        return p.midi
    except Exception as exc:
        raise ValueError(
            f"'{name}' is not a valid note name.  "
            f"Examples: C4, F#4, Bb3, Eb5"
        ) from exc


# ── Filtering ────────────────────────────────────────────────────────────────

def filter_by_range(
    songs: list[dict],
    user_min_midi: int,
    user_max_midi: int,
) -> list[dict]:
    """
    Hard-filter: keep only songs whose *entire* range fits inside the user's
    specified range.  Returns a **new** list — never mutates the input.

    A song passes if:
        song.statistics.pitch_range.min_midi >= user_min_midi
        AND
        song.statistics.pitch_range.max_midi <= user_max_midi
    """
    filtered: list[dict] = []
    for song in songs:
        pr = song.get('statistics', {}).get('pitch_range', {})
        song_min = pr.get('min_midi')
        song_max = pr.get('max_midi')
        if song_min is None or song_max is None:
            continue  # skip songs without range data
        if song_min >= user_min_midi and song_max <= user_max_midi:
            filtered.append(song)
    return filtered


# ── Vectorisation ────────────────────────────────────────────────────────────

def build_dense_vector(
    tessituragram: dict[str, float],
    min_midi: int,
    max_midi: int,
) -> np.ndarray:
    """
    Convert a sparse tessituragram dict (keyed by MIDI string) into a dense
    numpy array over the global pitch space [min_midi … max_midi].

    Length of result = max_midi - min_midi + 1.
    Positions for MIDI values not present in the tessituragram are filled with 0.
    """
    length = max_midi - min_midi + 1
    vec = np.zeros(length, dtype=np.float64)
    for midi_str, duration in tessituragram.items():
        idx = int(midi_str) - min_midi
        if 0 <= idx < length:
            vec[idx] = duration
    return vec


def normalize_l1(vec: np.ndarray) -> np.ndarray:
    """
    L1-normalise so the vector sums to 1 (proportion of singing time).
    Returns a zero vector unchanged.
    """
    total = vec.sum()
    if total == 0:
        return vec.copy()
    return vec / total


def normalize_l2(vec: np.ndarray) -> np.ndarray:
    """
    L2-normalise to a unit-length direction vector (for cosine similarity).
    Returns a zero vector unchanged.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.copy()
    return vec / norm


# ── Ideal vector ─────────────────────────────────────────────────────────────

def build_ideal_vector(
    min_midi: int,
    max_midi: int,
    favorite_midis: list[int],
    avoid_midis: list[int],
    base: float = 0.2,
    fav_boost: float = 1.0,
    avoid_pen: float = -1.0,
) -> np.ndarray:
    """
    Construct and L2-normalise an ideal tessituragram vector.

    1. Initialise every position in [min_midi … max_midi] to 0.
    2. Set *base* weight for all in-range positions.
    3. Add *fav_boost* to favourite-note positions.
    4. Add *avoid_pen* (negative) to avoid-note positions.
    5. Clamp any value below 0 to 0  — keeps the vector non-negative so that
       cosine similarity stays in [0, 1] and is easy to explain.
    6. L2-normalise.

    The resulting direction vector peaks at favourite notes, is low at avoid
    notes, and has a modest baseline everywhere else.
    """
    length = max_midi - min_midi + 1
    vec = np.full(length, base, dtype=np.float64)

    for midi in favorite_midis:
        idx = midi - min_midi
        if 0 <= idx < length:
            vec[idx] += fav_boost

    for midi in avoid_midis:
        idx = midi - min_midi
        if 0 <= idx < length:
            vec[idx] += avoid_pen

    # Clamp negatives to 0
    np.clip(vec, 0, None, out=vec)

    return normalize_l2(vec)


# ── Similarity / scoring ────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.  Returns 0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_songs(
    filtered_songs: list[dict],
    ideal_vec: np.ndarray,
    min_midi: int,
    max_midi: int,
    avoid_midis: list[int],
    favorite_midis: list[int],
    alpha: float = 0.5,
) -> list[dict]:
    """
    Score every song against the ideal vector and return results sorted
    best-to-worst.

    For each song:
        1. Build dense vector in the global pitch space.
        2. L1-normalise (proportion of singing time).
        3. Compute cosine similarity with the ideal vector.
        4. Compute explicit avoid penalty =
               sum(song_vec[i] for i in avoid_note_indices).
        5. final_score = cosine_sim − α × avoid_penalty

    Returns a list of result dicts (sorted descending by final_score).
    """
    avoid_indices = [m - min_midi for m in avoid_midis if min_midi <= m <= max_midi]
    fav_indices = [m - min_midi for m in favorite_midis if min_midi <= m <= max_midi]

    results: list[dict] = []
    for song in filtered_songs:
        tess = song.get('tessituragram', {})
        dense = build_dense_vector(tess, min_midi, max_midi)
        normed = normalize_l1(dense)

        cos_sim = cosine_similarity(normed, ideal_vec)

        avoid_penalty = float(sum(normed[i] for i in avoid_indices)) if avoid_indices else 0.0
        final_score = cos_sim - alpha * avoid_penalty

        # Favourite-note overlap (proportion of singing time on favourites)
        fav_overlap = float(sum(normed[i] for i in fav_indices)) if fav_indices else 0.0

        results.append({
            'filename': song.get('filename', ''),
            'composer': song.get('composer', 'Unknown'),
            'title': song.get('title', ''),
            'final_score': round(final_score, 4),
            'cosine_similarity': round(cos_sim, 4),
            'avoid_penalty': round(avoid_penalty, 4),
            'favorite_overlap': round(fav_overlap, 4),
            'tessituragram': tess,
            'normalized_vector': {str(min_midi + i): round(float(v), 6) for i, v in enumerate(normed)},
            'statistics': song.get('statistics', {}),
        })

    # Sort best → worst; tie-break by filename (A–Z)
    results.sort(key=lambda r: (-r['final_score'], r['filename']))

    # Assign ranks and generate explanations
    for rank, result in enumerate(results, 1):
        result['rank'] = rank
        result['explanation'] = generate_explanation(
            result, min_midi, favorite_midis, avoid_midis,
        )

    return results


# ── Explanation generator ────────────────────────────────────────────────────

def generate_explanation(
    result: dict,
    min_midi: int,
    favorite_midis: list[int],
    avoid_midis: list[int],
) -> str:
    """
    Produce a brief, human-readable explanation for why a song ranked
    where it did.
    """
    parts: list[str] = []
    score = result['final_score']
    cos = result['cosine_similarity']
    avoid_pen = result['avoid_penalty']
    fav_over = result.get('favorite_overlap', 0.0)

    parts.append(f"Final score: {score:.2f} (cosine similarity {cos:.2f})")

    # Favourite notes commentary
    if favorite_midis:
        fav_names = ', '.join(midi_to_note_name(m) for m in favorite_midis)
        pct = fav_over * 100
        if pct >= 30:
            parts.append(
                f"Strong overlap with your favorite notes ({fav_names}): "
                f"{pct:.0f}% of singing time."
            )
        elif pct >= 10:
            parts.append(
                f"Moderate overlap with favorite notes ({fav_names}): "
                f"{pct:.0f}% of singing time."
            )
        else:
            parts.append(
                f"Low overlap with favorite notes ({fav_names}): "
                f"only {pct:.0f}% of singing time."
            )

    # Avoid notes commentary
    if avoid_midis:
        avoid_names = ', '.join(midi_to_note_name(m) for m in avoid_midis)
        pct = avoid_pen * 100
        if pct <= 2:
            parts.append(
                f"Minimal presence of avoid notes ({avoid_names}): "
                f"{pct:.1f}% of singing time."
            )
        elif pct <= 10:
            parts.append(
                f"Some presence of avoid notes ({avoid_names}): "
                f"{pct:.1f}% of singing time."
            )
        else:
            parts.append(
                f"Notable presence of avoid notes ({avoid_names}): "
                f"{pct:.1f}% of singing time — this lowered the score."
            )

    return '  '.join(parts)
