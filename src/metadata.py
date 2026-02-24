"""Extract metadata (composer, title) from MusicXML files or filename parsing."""

import re
from pathlib import Path
from music21 import converter


def _parse_song_number_and_name(filepath: Path) -> tuple[str | None, str | None]:
    """
    Extract the individual song number and name from the filename.

    Returns (no_num, song_name) — either may be None if the pattern
    does not match.

    Example:
        "hensel-fanny-mendelssohn-6-lieder-op7-no1-nachtwanderer.mxl"
        → ("1", "Nachtwanderer")
    """
    match = re.search(r'no(\d+)-(.+)$', filepath.stem)
    if match:
        return match.group(1), match.group(2).replace('-', ' ').title()
    return None, None


def extract_metadata(filepath: Path) -> dict[str, str]:
    """
    Extract composer and title from MusicXML file.
    Falls back to filename parsing if metadata is missing.

    When MusicXML metadata supplies only a collection-level title
    (e.g. "6 Lieder, Op.7"), the individual song number and name are
    extracted from the filename and appended so each song is uniquely
    identified (e.g. "6 Lieder, Op.7 — No. 1 Nachtwanderer").

    Args:
        filepath: Path to .mxl file

    Returns:
        Dictionary with 'composer' and 'title' keys
    """
    # Try MusicXML metadata first
    try:
        score = converter.parse(str(filepath))
        if score.metadata:
            composer = score.metadata.composer
            title = score.metadata.title

            if composer and title:
                # Enrich with individual song info from the filename
                no_num, song_name = _parse_song_number_and_name(filepath)
                if no_num and song_name:
                    title = f"{title} — No. {no_num} {song_name}"
                elif no_num:
                    title = f"{title} — No. {no_num}"
                return {
                    'composer': composer,
                    'title': title,
                }
    except Exception:
        pass  # Fall back to filename parsing

    # Fallback: parse filename
    return parse_filename_metadata(filepath)


def parse_filename_metadata(filepath: Path) -> dict[str, str]:
    """
    Parse composer and title from filename pattern.
    
    Pattern: {lastname}-{firstname}-{...}-op{op}-no{no}-{title}.mxl
    Example: hensel-fanny-mendelssohn-6-lieder-op7-no1-nachtwanderer.mxl
    -> Composer: "Fanny Mendelssohn Hensel", Title: "Op. 7 No. 1 Nachtwanderer"
    
    Args:
        filepath: Path to .mxl file
        
    Returns:
        Dictionary with 'composer' and 'title' keys
    """
    filename = filepath.stem  # Remove .mxl extension
    
    # Pattern: {lastname}-{firstname}-{...}-op{op}-no{no}-{title}
    # Match op{number}-no{number}-{title}
    match = re.search(r'op(\d+)-no(\d+)-(.+)$', filename)
    
    if match:
        op_num = match.group(1)
        no_num = match.group(2)
        title_part = match.group(3).replace('-', ' ').title()
        title = f"Op. {op_num} No. {no_num} {title_part}"
    else:
        # Fallback: use filename as title
        title = filename.replace('-', ' ').title()
    
    # Extract composer name (everything before op{number} or last part)
    composer_match = re.search(r'^(.+?)(?:-op\d+|$)', filename)
    if composer_match:
        composer_parts = composer_match.group(1).split('-')
        # Typically: {lastname}-{firstname}-{middle/lastname}
        # Examples: hensel-fanny-mendelssohn, schumann-clara, schumann-robert
        
        if len(composer_parts) >= 2:
            # Handle cases like "hensel-fanny-mendelssohn" -> "Fanny Mendelssohn Hensel"
            # or "schumann-clara" -> "Clara Schumann"
            first_name = composer_parts[1].title()
            last_name = composer_parts[0].title()
            
            # If there's a third part, it might be another last name
            if len(composer_parts) >= 3:
                # Check if it's a known pattern (e.g., "mendelssohn" after "hensel")
                middle_part = composer_parts[2].title()
                composer = f"{first_name} {middle_part} {last_name}"
            else:
                composer = f"{first_name} {last_name}"
        else:
            composer = filename.split('-')[0].title()
    else:
        composer = "Unknown"
    
    return {
        'composer': composer,
        'title': title
    }
