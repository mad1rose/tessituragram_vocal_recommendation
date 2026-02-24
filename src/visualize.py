"""Generate a Jupyter notebook containing tessituragram histograms."""

import json
import re
import sys
from pathlib import Path

import nbformat

# Standard note names for each pitch class (0–11).
# Uses the most common enharmonic spelling for each chromatic degree.
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _midi_to_note_name(midi_num: int) -> str:
    """Convert a MIDI number to a standard note name, e.g. 60 -> 'C4'."""
    octave = (midi_num // 12) - 1
    note_index = midi_num % 12
    return f"{NOTE_NAMES[note_index]}{octave}"


def _pitch_sort_key(pitch_key: str) -> int:
    """Sort key for MIDI-number-based pitch keys."""
    try:
        return int(pitch_key)
    except ValueError:
        return 0


def _pretty_pitch(pitch_key: str) -> str:
    """Convert a MIDI number string to a label like 'C4 (60)'."""
    try:
        midi_num = int(pitch_key)
        return f"{_midi_to_note_name(midi_num)} ({midi_num})"
    except ValueError:
        return pitch_key


def _song_label(song: dict) -> str:
    """Build a readable label from filename, title, and composer."""
    filename = song.get('filename', '')
    m = re.search(r'no(\d+)-(.+)\.mxl$', filename)
    if m:
        no = m.group(1)
        name = m.group(2).replace('-', ' ').title()
        subtitle = f"No. {no} \u2013 {name}"
    else:
        subtitle = filename

    title = song.get('title', '')
    composer = song.get('composer', 'Unknown')
    return f"{title}, {subtitle}\n{composer}"


def _make_plot_code(song: dict, index: int, total: int) -> str:
    """Return a Python code string that plots one tessituragram histogram."""
    tess = song.get('tessituragram', {})
    if not tess:
        return f"# Song {index}/{total} has no tessituragram data."

    sorted_keys = sorted(tess.keys(), key=_pitch_sort_key)
    labels = [_pretty_pitch(k) for k in sorted_keys]
    values = [tess[k] for k in sorted_keys]
    label = _song_label(song)
    num_pitches = len(labels)
    fig_width = max(7, num_pitches * 0.75)

    return (
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        f"labels = {labels!r}\n"
        f"values = {values!r}\n"
        f"colors = plt.cm.viridis(np.linspace(0.25, 0.85, {num_pitches}))\n"
        "\n"
        f"fig, ax = plt.subplots(figsize=({fig_width:.1f}, 6))\n"
        "ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)\n"
        "ax.set_ylabel('Duration (quarter-note beats)')\n"
        "ax.set_xlabel('Pitch — Note Name (MIDI Number)')\n"
        f"ax.set_title({label!r}, fontsize=11, fontweight='bold')\n"
        "ax.tick_params(axis='x', labelsize=7)\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "fig.tight_layout()\n"
        "plt.show()\n"
    )


def generate_notebook(
    json_path: str = 'data/tessituragrams.json',
    output_path: str = 'tessituragrams.ipynb',
) -> None:
    """
    Read tessituragram JSON and write a Jupyter notebook with one
    histogram per cell.
    """
    src = Path(json_path)
    if not src.exists():
        print(f"Error: {json_path} not found. Run main.py first to generate tessituragrams.")
        sys.exit(1)

    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    songs = data.get('songs', [])

    if not songs:
        print("No tessituragrams found in data file.")
        sys.exit(1)

    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }

    # Title cell
    nb.cells.append(nbformat.v4.new_markdown_cell(
        "# Tessituragram Visualizations\n\n"
        "Each cell below renders a histogram for one song.  \n"
        "Run **Cell \u2192 Run All** (or **Ctrl+Shift+Enter** / **\u21e7\u2318\u23ce**) "
        "to display every chart, then scroll through the notebook.\n\n"
        "**Reading the x-axis:** Each pitch is labeled with its standard note "
        "name followed by its MIDI number in parentheses, e.g. **C4 (60)**.  \n"
        "MIDI numbers are absolute \u2014 enharmonic equivalents (like F\u266f and G\u266d) "
        "share the same number, so pitches are directly comparable across songs."
    ))

    # One code cell per song
    for i, song in enumerate(songs, 1):
        code = _make_plot_code(song, i, len(songs))
        nb.cells.append(nbformat.v4.new_code_cell(code))

    dest = Path(output_path)
    with open(dest, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook written to {dest.resolve()}")
    print(f"Contains {len(songs)} tessituragram(s).")
    print()
    print("To view, run one of:")
    print(f"    jupyter notebook {output_path}")
    print(f"    jupyter lab {output_path}")
    print("Or open the .ipynb file directly in VS Code / Cursor.")


if __name__ == '__main__':
    generate_notebook()
