"""Generate duration-weighted tessituragrams keyed by MIDI note number."""

from collections import defaultdict
from music21 import note, pitch


def generate_tessituragram(notes: list[note.Note | note.Rest]) -> dict[str, float]:
    """
    Generate a duration-weighted tessituragram from vocal line notes.
    
    Uses MIDI note numbers as keys so that enharmonic equivalents (e.g.,
    F#4 and Gb4) are collapsed into a single entry, making cross-song
    comparison consistent regardless of key signature.
    
    Args:
        notes: List of Note and Rest objects from vocal line
        
    Returns:
        Dictionary mapping MIDI note numbers (as strings) to weighted
        duration (quarterLength)
    """
    tessituragram = defaultdict(float)
    
    for note_obj in notes:
        # Skip rests
        if isinstance(note_obj, note.Rest):
            continue
        
        # Use MIDI number as key — collapses enharmonic equivalents
        midi_num = note_obj.pitch.midi
        duration = note_obj.duration.quarterLength
        
        tessituragram[str(midi_num)] += duration
    
    return dict(tessituragram)


def calculate_statistics(
    notes: list[note.Note | note.Rest],
    tessituragram: dict[str, float]
) -> dict[str, object]:
    """
    Calculate statistics about the tessituragram.
    
    Args:
        notes: List of Note and Rest objects
        tessituragram: Tessituragram dictionary
        
    Returns:
        Dictionary with statistics:
        - total_duration: Total duration in quarterLength
        - pitch_range: Dict with 'min' and 'max' pitch names
        - unique_pitches: Number of unique pitch variants
    """
    # Calculate total duration
    total_duration = sum(
        note_obj.duration.quarterLength
        for note_obj in notes
    )
    
    # Find pitch range
    pitches = []
    for note_obj in notes:
        if isinstance(note_obj, note.Note):
            pitches.append(note_obj.pitch)
    
    if pitches:
        min_pitch = min(pitches, key=lambda p: p.midi)
        max_pitch = max(pitches, key=lambda p: p.midi)
        pitch_range = {
            'min': min_pitch.nameWithOctave,
            'min_midi': min_pitch.midi,
            'max': max_pitch.nameWithOctave,
            'max_midi': max_pitch.midi,
        }
    else:
        pitch_range = {'min': None, 'min_midi': None,
                       'max': None, 'max_midi': None}
    
    # Count unique pitches
    unique_pitches = len(tessituragram)
    
    return {
        'total_duration': total_duration,
        'pitch_range': pitch_range,
        'unique_pitches': unique_pitches
    }
