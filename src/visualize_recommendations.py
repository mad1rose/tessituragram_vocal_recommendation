"""Generate a Jupyter notebook visualising ranked song recommendations."""

import json
import re
import sys
from pathlib import Path

import nbformat

# Standard note names for each pitch class (0–11).
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _midi_to_note_name(midi_num: int) -> str:
    octave = (midi_num // 12) - 1
    return f"{NOTE_NAMES[midi_num % 12]}{octave}"


def _pretty_pitch(midi_num: int) -> str:
    return f"{_midi_to_note_name(midi_num)} ({midi_num})"


def _song_label(rec: dict) -> str:
    filename = rec.get('filename', '')
    m = re.search(r'no(\d+)-(.+)\.mxl$', filename)
    if m:
        no = m.group(1)
        name = m.group(2).replace('-', ' ').title()
        subtitle = f"No. {no} \u2013 {name}"
    else:
        subtitle = filename

    title = rec.get('title', '')
    composer = rec.get('composer', 'Unknown')
    return f"{title}, {subtitle}\\n{composer}"


def _make_plot_code(
    rec: dict,
    ideal_vector: dict,
    min_midi: int,
    max_midi: int,
) -> str:
    """
    Return Python code that plots:
      - The song's normalised tessituragram as bars.
      - The ideal vector as a translucent line overlay.
    """
    normed = rec.get('normalized_vector', {})
    if not normed:
        return f"# Song '{rec.get('filename', '?')}' has no vector data."

    # Build parallel lists over the full pitch space
    all_midis = list(range(min_midi, max_midi + 1))
    labels = [_pretty_pitch(m) for m in all_midis]
    song_values = [normed.get(str(m), 0.0) for m in all_midis]
    ideal_values = [ideal_vector.get(str(m), 0.0) for m in all_midis]
    num_pitches = len(labels)
    fig_width = max(8, num_pitches * 0.75)

    rank = rec.get('rank', '?')
    score = rec.get('final_score', 0)
    label = _song_label(rec)

    return (
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        f"labels = {labels!r}\n"
        f"song_vals = {song_values!r}\n"
        f"ideal_vals = {ideal_values!r}\n"
        f"num = {num_pitches}\n"
        "\n"
        "x = np.arange(num)\n"
        f"colors = plt.cm.viridis(np.linspace(0.25, 0.85, num))\n"
        "\n"
        f"fig, ax = plt.subplots(figsize=({fig_width:.1f}, 6))\n"
        "ax.bar(x, song_vals, color=colors, edgecolor='white', linewidth=0.5,\n"
        "       label='Song (normalised)', zorder=2)\n"
        "ax.plot(x, ideal_vals, color='red', linewidth=2, marker='o',\n"
        "        markersize=4, alpha=0.7, label='Ideal vector', zorder=3)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)\n"
        "ax.set_ylabel('Proportion of singing time')\n"
        "ax.set_xlabel('Pitch — Note Name (MIDI Number)')\n"
        f"ax.set_title('#{rank}  (score {score:.2f})  ' + {label!r},\n"
        "             fontsize=10, fontweight='bold')\n"
        "ax.legend(fontsize=8)\n"
        "fig.tight_layout()\n"
        "plt.show()\n"
    )


def generate_notebook(
    json_path: str = 'data/recommendations.json',
    output_path: str = 'recommendations.ipynb',
) -> None:
    """
    Read recommendation JSON and write a Jupyter notebook with one
    histogram per recommended song, overlaid with the ideal vector.
    """
    src = Path(json_path)
    if not src.exists():
        print(f"Error: {json_path} not found.")
        print("Run  python -m src.run_recommendations  first to generate recommendations.")
        sys.exit(1)

    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prefs = data.get('user_preferences', {})
    ideal_vector = data.get('ideal_vector', {})
    recs = data.get('recommendations', [])

    if not recs:
        print("No recommendations found in data file.")
        sys.exit(1)

    rng = prefs.get('range', {})
    min_midi = rng.get('low_midi', 0)
    max_midi = rng.get('high_midi', 127)

    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }

    # ── Title / summary cell ─────────────────────────────────────────────
    fav_str = ', '.join(prefs.get('favorite_notes', [])) or '(none)'
    avoid_str = ', '.join(prefs.get('avoid_notes', [])) or '(none)'

    summary = (
        "# Recommended Songs — Tessituragram Comparison\n\n"
        f"**Your range:** {rng.get('low', '?')} – {rng.get('high', '?')}  \n"
        f"**Favorite notes:** {fav_str}  \n"
        f"**Avoid notes:** {avoid_str}  \n"
        f"**Songs evaluated:** {len(recs)}  \n\n"
        "Each chart below shows the song's normalised tessituragram (bars) "
        "overlaid with the **ideal vector** (red line).  \n"
        "Songs are ordered from best match (#1) to worst match.\n\n"
        "Run **Cell \u2192 Run All** (or **Ctrl+Shift+Enter**) to render all charts."
    )
    nb.cells.append(nbformat.v4.new_markdown_cell(summary))

    # ── One markdown + code cell per recommendation ──────────────────────
    for rec in recs:
        rank = rec.get('rank', '?')
        title = rec.get('title', '')
        composer = rec.get('composer', '')
        explanation = rec.get('explanation', '')

        md = (
            f"### #{rank}  {title}\n"
            f"**{composer}**  \n"
            f"{explanation}"
        )
        nb.cells.append(nbformat.v4.new_markdown_cell(md))

        code = _make_plot_code(rec, ideal_vector, min_midi, max_midi)
        nb.cells.append(nbformat.v4.new_code_cell(code))

    dest = Path(output_path)
    with open(dest, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook written to {dest.resolve()}")
    print(f"Contains {len(recs)} recommended song(s).")
    print()
    print("To view, run one of:")
    print(f"    jupyter notebook {output_path}")
    print(f"    jupyter lab {output_path}")
    print("Or open the .ipynb file directly in VS Code / Cursor.")


if __name__ == '__main__':
    generate_notebook()
