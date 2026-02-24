"""Parse MusicXML files and extract vocal line notes."""

from pathlib import Path
from music21 import converter, stream, note, chord


def extract_vocal_line(filepath: Path) -> list[note.Note | note.Rest]:
    """
    Extract all notes from the vocal line(s) of a MusicXML file.
    
    Identifies vocal parts by:
    1. Parts with lyrics
    2. Highest voice part (if no lyrics found)
    3. First part (fallback)
    
    Args:
        filepath: Path to .mxl file
        
    Returns:
        List of Note and Rest objects from the vocal line
    """
    score = converter.parse(str(filepath))
    
    # Find vocal part(s)
    vocal_parts = _identify_vocal_parts(score)
    
    if not vocal_parts:
        # Fallback: use first part
        if len(score.parts) > 0:
            vocal_parts = [score.parts[0]]
        else:
            return []
    
    # Extract all notes from vocal parts
    notes = []
    for part in vocal_parts:
        # Flatten the part to get all elements
        flat_part = part.flat
        for element in flat_part:
            if isinstance(element, note.Note):
                notes.append(element)
            elif isinstance(element, note.Rest):
                notes.append(element)
            elif isinstance(element, chord.Chord):
                # For chords, take the highest note (typically the melody)
                highest_note = element.sortAscending().pitches[-1]
                note_obj = note.Note(highest_note)
                note_obj.duration = element.duration
                notes.append(note_obj)
    
    return notes


def _identify_vocal_parts(score: stream.Score) -> list[stream.Part]:
    """
    Identify which parts in the score are vocal parts.
    
    Priority:
    1. Parts with lyrics
    2. Highest voice part (highest average pitch)
    3. First part (fallback)
    
    Args:
        score: music21 Score object
        
    Returns:
        List of Part objects identified as vocal parts
    """
    parts_with_lyrics = []
    parts_without_lyrics = []
    
    for part in score.parts:
        # Check if part has lyrics
        flat_part = part.flat
        has_lyrics = any(
            hasattr(element, 'lyrics') and element.lyrics
            for element in flat_part
            if isinstance(element, note.Note)
        )
        
        if has_lyrics:
            parts_with_lyrics.append(part)
        else:
            parts_without_lyrics.append(part)
    
    # If we found parts with lyrics, return those
    if parts_with_lyrics:
        return parts_with_lyrics
    
    # Otherwise, find the highest voice part
    if parts_without_lyrics:
        highest_part = max(
            parts_without_lyrics,
            key=lambda p: _get_average_pitch(p)
        )
        return [highest_part]
    
    # Fallback: return first part
    if len(score.parts) > 0:
        return [score.parts[0]]
    
    return []


def _get_average_pitch(part: stream.Part) -> float:
    """
    Calculate average MIDI pitch number for a part.
    Used to identify the highest voice part.
    
    Args:
        part: music21 Part object
        
    Returns:
        Average MIDI pitch (0 if no pitches found)
    """
    pitches = []
    flat_part = part.flat
    
    for element in flat_part:
        if isinstance(element, note.Note):
            pitches.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            # Use highest note of chord
            pitches.append(max(p.midi for p in element.pitches))
    
    return sum(pitches) / len(pitches) if pitches else 0.0
