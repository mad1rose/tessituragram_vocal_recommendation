"""JSON storage and retrieval for tessituragrams and recommendations."""

import json
from pathlib import Path
from typing import Optional


def merge_songs(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Merge new songs into existing list, deduplicating by filename.

    Args:
        existing: Already-stored song dicts
        new: Newly processed song dicts

    Returns:
        Combined list with no duplicate filenames
    """
    seen = {s['filename'] for s in existing}
    merged = list(existing)
    for song in new:
        if song['filename'] not in seen:
            merged.append(song)
            seen.add(song['filename'])
    return merged


def save_tessituragrams(
    songs: list[dict],
    output_path: Path
) -> None:
    """
    Save tessituragrams to JSON file.
    
    Args:
        songs: List of song dictionaries with tessituragram data
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'songs': songs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_tessituragrams(input_path: Path) -> list[dict]:
    """
    Load tessituragrams from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        List of song dictionaries
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('songs', [])


# ── Recommendations I/O ──────────────────────────────────────────────────────

def save_recommendations(
    user_preferences: dict,
    ideal_vector: dict,
    recommendations: list[dict],
    output_path: Path,
) -> None:
    """
    Save ranked recommendations to a JSON file.

    Args:
        user_preferences: Dict with range, favorite_notes, avoid_notes, alpha
        ideal_vector: Dict mapping MIDI string → weight (the normalised ideal)
        recommendations: Ranked list of result dicts from score_songs()
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'user_preferences': user_preferences,
        'ideal_vector': ideal_vector,
        'recommendations': recommendations,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_recommendations(input_path: Path) -> dict:
    """
    Load recommendations from a JSON file.

    Args:
        input_path: Path to recommendations JSON file

    Returns:
        Full dict with keys: user_preferences, ideal_vector, recommendations
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def query_tessituragrams(
    songs: list[dict],
    composer: Optional[str] = None,
    title: Optional[str] = None,
    min_midi: Optional[int] = None,
    max_midi: Optional[int] = None
) -> list[dict]:
    """
    Filter tessituragrams based on criteria.
    
    Args:
        songs: List of song dictionaries
        composer: Filter by composer (partial match, case-insensitive)
        title: Filter by title (partial match, case-insensitive)
        min_midi: Only include songs whose range reaches at least this MIDI note
        max_midi: Only include songs whose range does not exceed this MIDI note
        
    Returns:
        Filtered list of song dictionaries
    """
    filtered = songs
    
    if composer:
        filtered = [
            song for song in filtered
            if composer.lower() in song.get('composer', '').lower()
        ]
    
    if title:
        filtered = [
            song for song in filtered
            if title.lower() in song.get('title', '').lower()
        ]
    
    if min_midi is not None or max_midi is not None:
        result = []
        for song in filtered:
            pitch_range = song.get('statistics', {}).get('pitch_range', {})
            song_min = pitch_range.get('min_midi')
            song_max = pitch_range.get('max_midi')
            if song_min is None or song_max is None:
                continue
            if min_midi is not None and song_max < min_midi:
                continue
            if max_midi is not None and song_min > max_midi:
                continue
            result.append(song)
        filtered = result
    
    return filtered
