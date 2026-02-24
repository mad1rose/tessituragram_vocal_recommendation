"""Interactive CLI for generating ranked song recommendations."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src.storage import load_tessituragrams, save_recommendations
from src.recommend import (
    note_name_to_midi,
    midi_to_note_name,
    filter_by_range,
    build_ideal_vector,
    score_songs,
)


# ── Input helpers ────────────────────────────────────────────────────────────

# Hyphen after a digit = range separator (avoids splitting B-4 meaning Bb4)
_RANGE_SEP_RE = re.compile(r'(?<=\d)-')

def _parse_note_or_range(token: str) -> list[int]:
    """
    Parse a token as either a single note or a range D4-E4 (D4 and E4 and all
    pitched notes in between).  Returns a list of MIDI numbers.
    Raises ValueError if the token cannot be parsed.
    """
    token = token.strip()
    if not token:
        return []
    # Check for range format: NOTE1-NOTE2 (e.g. D4-E4, Bb3-C4, B-4-C5)
    parts = _RANGE_SEP_RE.split(token, maxsplit=1)
    if len(parts) == 2:
        low_str, high_str = parts[0].strip(), parts[1].strip()
        if low_str and high_str:
            try:
                low_midi = note_name_to_midi(low_str)
                high_midi = note_name_to_midi(high_str)
                if low_midi > high_midi:
                    low_midi, high_midi = high_midi, low_midi
                return list(range(low_midi, high_midi + 1))
            except ValueError:
                pass  # Fall through to single-note parse
    # Single note
    return [note_name_to_midi(token)]


def _prompt_note(prompt_text: str) -> int:
    """Repeatedly prompt until the user enters a valid note name.  Returns MIDI."""
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            continue
        try:
            midi = note_name_to_midi(raw)
            print(f"  → {midi_to_note_name(midi)} (MIDI {midi})")
            return midi
        except ValueError as e:
            print(f"  Invalid note: {e}")
            print("  Please try again.  Examples: C4, F#4, Bb3, Eb5\n")


def _prompt_note_list(prompt_text: str) -> list[int]:
    """
    Prompt for a comma-separated list of note names and/or ranges.
    Each item can be a single note (e.g. A4) or a range (e.g. D4-E4 = D4, E4,
    and all pitches in between).  Returns a list of MIDI numbers, or [] if skip.
    """
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            return []
        midis: list[int] = []
        bad = False
        for token in raw.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                parsed = _parse_note_or_range(token)
                midis.extend(parsed)
            except ValueError as e:
                print(f"  Invalid note or range '{token}': {e}")
                bad = True
        if bad:
            print("  Please re-enter.  Examples: A4, D5  or  D4-E4  or  A4, C4-E4\n")
            continue
        if midis:
            # Dedupe while preserving order
            midis = list(dict.fromkeys(midis))
            midis.sort()
            names = ', '.join(f"{midi_to_note_name(m)} (MIDI {m})" for m in midis)
            print(f"  → {names}")
        return midis


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    library_path = Path('data/tessituragrams.json')
    output_path = Path('data/recommendations.json')

    # ── Banner ───────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("   Tessituragram Repertoire Recommender")
    print("=" * 64)
    print()
    print("This tool recommends songs from your library that best match")
    print("your vocal range and note preferences.")
    print()
    print("You will be asked to enter:")
    print("  1. Your vocal range  (lowest and highest notes)")
    print("  2. Your favorite notes  (notes you love singing — optional)")
    print("  3. Notes to avoid  (notes you'd rather not sing — optional)")
    print()
    print("Enter notes as a note name + octave number.")
    print("  Single notes:  C4, F#4, Bb3, Eb5")
    print("  Ranges (optional): D4-E4 = D4, E4, and all pitched notes in between")
    print("  Mix both:  A4, D4-E4, F5  (comma-separated)")
    print()
    print("-" * 64)
    print()

    # ── 1. Collect range ─────────────────────────────────────────────────
    print("STEP 1: Enter your vocal range\n")
    user_min = _prompt_note("  Lowest note in your range: ")
    user_max = _prompt_note("  Highest note in your range: ")

    if user_min > user_max:
        print("\n  (Swapping — your lowest note was higher than your highest.)")
        user_min, user_max = user_max, user_min

    print(f"\n  Your range: {midi_to_note_name(user_min)} – "
          f"{midi_to_note_name(user_max)}  (MIDI {user_min}–{user_max})")
    print()

    # ── 2. Favorite notes ────────────────────────────────────────────────
    print("-" * 64)
    print("\nSTEP 2: Enter your favorite notes (optional)\n")
    print("  These are notes you especially enjoy singing.")
    print("  Songs that emphasise these notes will rank higher.")
    print("  You may enter single notes (A4, D5) or ranges (D4-E4 = D4, E4,")
    print("  and all pitched notes in between).  Mix both:  A4, D4-E4, F5\n")
    favorite_midis = _prompt_note_list(
        "  Favorite notes (comma-separated; use D4-E4 for ranges; Enter to skip): "
    )

    # Warn if favourites fall outside range
    out_of_range = [m for m in favorite_midis if m < user_min or m > user_max]
    if out_of_range:
        names = ', '.join(midi_to_note_name(m) for m in out_of_range)
        print(f"\n  Note: {names} fall outside your range and will be ignored.")
        favorite_midis = [m for m in favorite_midis if user_min <= m <= user_max]
    print()

    # ── 3. Avoid notes ───────────────────────────────────────────────────
    print("-" * 64)
    print("\nSTEP 3: Enter notes to avoid (optional)\n")
    print("  Songs that spend a lot of time on these notes will be")
    print("  penalised in the ranking.  Single notes (C4) or ranges (D4-E4).\n")
    avoid_midis = _prompt_note_list(
        "  Notes to avoid (comma-separated; use D4-E4 for ranges; Enter to skip): "
    )

    out_of_range = [m for m in avoid_midis if m < user_min or m > user_max]
    if out_of_range:
        names = ', '.join(midi_to_note_name(m) for m in out_of_range)
        print(f"\n  Note: {names} fall outside your range and will be ignored.")
        avoid_midis = [m for m in avoid_midis if user_min <= m <= user_max]
    print()

    # ── 4. Load library ─────────────────────────────────────────────────
    print("-" * 64)
    print("\nLoading song library...")
    if not library_path.exists():
        print(f"\nError: Library not found at {library_path}")
        print("Run  python -m src.main  first to generate tessituragrams.")
        sys.exit(1)

    all_songs = load_tessituragrams(library_path)
    print(f"  {len(all_songs)} song(s) in library.")

    # ── 5. Filter ────────────────────────────────────────────────────────
    print("\nFiltering by range...")
    filtered = filter_by_range(all_songs, user_min, user_max)
    excluded = len(all_songs) - len(filtered)
    print(f"  {len(filtered)} song(s) fit within your range.")
    print(f"  {excluded} song(s) excluded (range extends outside yours).")

    if not filtered:
        print("\nNo songs match your range. Try widening it.")
        sys.exit(0)

    # ── 6. Build ideal vector & score ────────────────────────────────────
    print("\nBuilding ideal vector and scoring songs...")
    alpha = 0.5
    ideal_vec = build_ideal_vector(
        user_min, user_max, favorite_midis, avoid_midis,
    )

    results = score_songs(
        filtered, ideal_vec, user_min, user_max,
        avoid_midis, favorite_midis, alpha,
    )

    # ── 7. Print results ─────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("   RANKED RECOMMENDATIONS (best fit → worst fit)")
    print("=" * 64)
    print()

    for r in results:
        print(f"  #{r['rank']}  {r['title']}")
        print(f"      Composer: {r['composer']}")
        print(f"      File:     {r['filename']}")
        print(f"      {r['explanation']}")
        print()

    # ── 8. Save ──────────────────────────────────────────────────────────
    # Build serialisable ideal-vector dict
    ideal_dict = {
        str(user_min + i): round(float(v), 6)
        for i, v in enumerate(ideal_vec)
    }

    user_prefs = {
        'range': {
            'low': midi_to_note_name(user_min),
            'low_midi': user_min,
            'high': midi_to_note_name(user_max),
            'high_midi': user_max,
        },
        'favorite_notes': [midi_to_note_name(m) for m in favorite_midis],
        'favorite_midis': favorite_midis,
        'avoid_notes': [midi_to_note_name(m) for m in avoid_midis],
        'avoid_midis': avoid_midis,
        'alpha': alpha,
    }

    save_recommendations(user_prefs, ideal_dict, results, output_path)
    print(f"Results saved to {output_path}")
    print()
    print("To visualise the recommended songs' tessituragrams, run:")
    print("    python -m src.visualize_recommendations")
    print()


if __name__ == '__main__':
    main()
