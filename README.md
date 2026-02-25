# Tessituragram-Based Repertoire Recommender

A Python system that analyses vocal-line MusicXML files, generates duration-weighted tessituragrams (pitch distribution frequencies keyed by MIDI number), and recommends songs from a personal library that best match a singer's vocal range and note preferences.

## Overview

The system has two main stages:

1. **Tessituragram Generation** — Parse `.mxl` files, extract the vocal line, and store each song's pitch-duration profile in a JSON library (`data/tessituragrams.json`).
2. **Recommendation Engine** — Accept a singer's vocal range, favorite notes, and notes to avoid; filter the library; build an ideal vector; score every qualifying song with cosine similarity + an explicit avoid-note penalty; and output ranked recommendations. Favorite and avoid notes can be single notes (A4) or ranges (D4-E4 = D4, E4, and all pitched notes in between).

## How It Works

### Tessituragrams as Sparse Vectors

Each song's tessituragram is a sparse vector keyed by MIDI number. The value at each key is the total duration (in quarter-note beats) the singer spends on that pitch. For example:

```json
{ "60": 1.5, "62": 6.5, "64": 8.0, "65": 14.5 }
```

MIDI numbers collapse enharmonic equivalents (F#4 and Gb4 are both MIDI 66), making cross-song comparison consistent regardless of key signature.

### The Recommendation Pipeline

```
User Input (range, favorites, avoids)
        │
        ▼
   Note-name → MIDI conversion
        │
        ▼
   Load main library (data/tessituragrams.json)
        │
        ▼
   Hard-filter by range
   (song's entire min–max must fit inside user's range)
        │
        ▼
   Define global pitch space = [user_min_midi … user_max_midi]
        │
        ├──────────────────────────────┐
        ▼                              ▼
   Convert each song to a         Build ideal vector:
   dense, L1-normalised vector      base weight everywhere,
   (proportions summing to 1)       + boost at favourites,
                                    + penalty at avoids,
                                    L2-normalised
        │                              │
        └──────────┬───────────────────┘
                   ▼
   Score each song:
     cosine_similarity(song_vec, ideal_vec)
     − α × Σ song_vec[avoid_indices]
                   │
                   ▼
   Rank best → worst, generate explanations
                   │
                   ▼
   Save to data/recommendations.json
```

### Normalisation

- **Song vectors**: L1-normalised (divide by sum) so every vector sums to 1. Values become the *proportion of singing time* on each pitch. This makes songs of different lengths directly comparable.
- **Ideal vector**: L2-normalised (unit direction vector). This is the standard reference side for cosine similarity.

### Ideal Vector Construction

Given the user's range `[min_midi, max_midi]`, favourite MIDI notes, and avoid MIDI notes:

1. Initialise a zero vector of length `(max_midi − min_midi + 1)`.
2. Set a **base weight** (default 0.2) at every position.
3. Add a **favourite boost** (+1.0) at favourite-note positions.
4. Add an **avoid penalty** (−1.0) at avoid-note positions.
5. Clamp negative values to 0.
6. L2-normalise to a unit vector.

The result peaks at favourite notes, drops to near-zero at avoid notes, and has a modest baseline elsewhere.

### Scoring

For each song that passes the range filter:

```
final_score = cosine_similarity(song_vec, ideal_vec)
            − α × Σ song_vec[i]   for i in avoid_note_indices
```

- **Cosine similarity** is shape-based and magnitude-insensitive. It naturally rewards songs whose pitch distribution aligns with the ideal vector.
- **Explicit avoid penalty** (`α = 0.5` by default) further suppresses songs that spend significant time on undesired pitches. This makes the scoring transparent and easy to explain.

## Project Structure

```
tessituragram_vocal_recommendation/
├── README.md                           ← You are here
├── requirements.txt                    ← Python dependencies
├── how_tos/
│   ├── how_to_create_tessituragrams.txt  ← Beginner guide: generating tessituragrams
│   ├── how_to_view_tessituragrams.txt    ← Beginner guide: viewing histograms
│   └── how_to_get_recommendations.txt    ← Beginner guide: running the recommender
├── src/
│   ├── __init__.py
│   ├── parser.py                       ← MusicXML parsing, vocal-line extraction
│   ├── tessituragram.py                ← Tessituragram generation + statistics
│   ├── metadata.py                     ← Composer/title extraction
│   ├── storage.py                      ← JSON I/O (library + recommendations)
│   ├── main.py                         ← CLI: generate tessituragrams
│   ├── visualize.py                    ← Generate tessituragrams.ipynb
│   ├── recommend.py                    ← Core recommendation engine
│   ├── run_recommendations.py          ← Interactive CLI: get recommendations
│   └── visualize_recommendations.py    ← Generate recommendations.ipynb
├── data/
│   ├── tessituragrams.json             ← Main song library (generated)
│   └── recommendations.json            ← Ranked recommendations (generated)
├── songs/
│   └── mxl_songs/                      ← Input .mxl files
├── experiment/
│   ├── run_rq1_experiment.py           ← RQ1: Self-retrieval accuracy
│   ├── run_rq2_experiment.py           ← RQ2: Ranking stability
│   ├── run_rq3_experiment.py           ← RQ3: Score spread & internal validity
│   ├── visualize_rq1.py               ← RQ1 visualizations
│   ├── visualize_rq2.py               ← RQ2 visualizations
│   └── visualize_rq3.py               ← RQ3 visualizations
├── tessituragrams.ipynb                ← Visualisations of all songs (generated)
└── recommendations.ipynb               ← Visualisations of recommendations (generated)
```

## Data

The original MusicXML files were obtained from the [OpenScore Lieder Corpus](https://doi.org/10.17613/1my2-dm23) (Gotham and Jonas, 2022), available under CC0. The spreadsheet `songs/tessitura_recommendation_songs.xlsx` lists every song in the corpus with its composer, title, and the direct MuseScore URL from which each file was downloaded, so the full dataset can be independently reconstructed. Pre-extracted tessituragrams are provided in `data/tessituragrams.json`. To regenerate from source files, place `.mxl` files in `songs/mxl_songs/` and run `python -m src.main`.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `music21`, `numpy`, `matplotlib`, `nbformat`, `jupyter`, `scipy`.

## Quick Start

### 1. Generate tessituragrams from your MusicXML files

```bash
python -m src.main
```

This processes all `.mxl` files in `songs/mxl_songs/` and writes `data/tessituragrams.json`. Duplicate filenames are automatically skipped.

### 2. Get personalised song recommendations

```bash
python -m src.run_recommendations
```

The interactive prompt will ask for your range, favourite notes, and avoid notes. You can enter single notes (e.g. A4, D5) or ranges (e.g. D4-E4 = D4, E4, and all pitched notes in between). Results are printed to the terminal and saved to `data/recommendations.json`.

### 3. Visualise the recommendations (optional)

```bash
python -m src.visualize_recommendations
```

This generates `recommendations.ipynb` — a Jupyter notebook with one chart per recommended song, overlaid with the ideal vector for comparison. Open it in Jupyter or VS Code / Cursor.

### 4. Visualise all tessituragrams (optional)

```bash
python -m src.visualize
```

This generates `tessituragrams.ipynb` — histograms for every song in the library.

## JSON Schemas

### Main Library (`data/tessituragrams.json`)

```json
{
  "songs": [
    {
      "composer": "Clara Schumann",
      "title": "6 Lieder, Op.13",
      "filename": "schumann-clara-6-lieder-op13-no1-ich-stand-in-dunklen-traumen.mxl",
      "tessituragram": { "62": 1.0, "63": 3.5, "65": 2.5, ... },
      "statistics": {
        "total_duration": 111.0,
        "pitch_range": { "min": "D4", "min_midi": 62, "max": "F5", "max_midi": 77 },
        "unique_pitches": 14
      }
    }
  ]
}
```

### Recommendations (`data/recommendations.json`)

```json
{
  "user_preferences": {
    "range": { "low": "C4", "low_midi": 60, "high": "G5", "high_midi": 79 },
    "favorite_notes": ["A4", "D5"],
    "avoid_notes": ["C4"],
    "alpha": 0.5
  },
  "ideal_vector": { "60": 0.0, "61": 0.12, "62": 0.12, ... },
  "recommendations": [
    {
      "rank": 1,
      "filename": "...",
      "composer": "...",
      "title": "...",
      "final_score": 0.87,
      "cosine_similarity": 0.91,
      "avoid_penalty": 0.08,
      "favorite_overlap": 0.34,
      "explanation": "Final score: 0.87 (cosine similarity 0.91) ...",
      "tessituragram": { ... },
      "normalized_vector": { ... },
      "statistics": { ... }
    }
  ]
}
```

## Dependencies

| Package | Purpose |
| --- | --- |
| `music21>=9.0` | MusicXML parsing, pitch/MIDI conversion |
| `numpy>=1.24` | Vector operations, cosine similarity |
| `matplotlib>=3.8` | Histogram visualisation |
| `nbformat>=5.9` | Jupyter notebook generation |
| `jupyter` | Notebook viewer |
| `scipy>=1.10` | Kendall's tau for RQ2 experiment |