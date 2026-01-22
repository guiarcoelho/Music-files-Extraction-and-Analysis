# Music Library Analyzer (BPM + Camelot Key + Energy) 🎧

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![librosa](https://img.shields.io/pypi/v/librosa?label=librosa)](https://pypi.org/project/librosa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small Python script that scans a folder of audio files, estimates each track’s **BPM** and **Camelot key** (DJ-friendly), computes a simple **energy score**, and writes a report you can sort for DJ prep / library triage. 🧰

It’s heuristic-based (not “perfect music theory”), but fast and practical. ⚡️

## What it does ✅

- Reads audio files from a folder (`.mp3`, `.wav`, `.flac`, `.m4a`)
- Estimates tempo (BPM) from percussive onset strength, and optionally normalizes common half/double-time mistakes into a target BPM range
- Estimates key using harmonic chroma + Krumhansl–Schmuckler key-profile correlation, then maps to Camelot notation
- Computes an energy score (1–10) from RMS loudness + onset strength + spectral centroid (normalized within the analyzed folder)
- Outputs `music_comprehensive_report.txt` (sorted by name by default)

## Output format 📄

The report is written to `music_comprehensive_report.txt` and looks like:

```text
FILE NAME                                          | BPM      | KEY      | ENERGY (1-10)
------------------------------------------------------------------------------------------
Folder/track_name.mp3                              | 124.0    | 8A       | 7
```

## Requirements 🧩

- Python 3.9+ recommended
- `librosa` (brings in `numpy` and other scientific deps)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install librosa
```

Notes:
- MP3 decoding may require system support (commonly `ffmpeg`). On macOS: `brew install ffmpeg`.

## Quick start 🚀

Run it with CLI flags:

```bash
python3 extrac_analysis.py --music-dir "/path/to/your/music/folder" --output-dir out
```

Grab your report:

- `out/music_comprehensive_report.txt`

Common useful options:

```bash
# Sort by Camelot key then BPM, and scan subfolders
python3 extrac_analysis.py --music-dir "/path/to/music" --output-dir out --recursive --sort key
```

## How it works (high level) 🧠

- **BPM**: percussive component → onset strength → `librosa.beat.beat_track` (optionally normalized to your BPM range).
- **Key/Camelot**: harmonic component → `chroma_cqt` → key-profile correlation (major/minor) → Camelot mapping.
- **Energy (1–10)**: RMS loudness + onset strength + spectral centroid, normalized within the analyzed folder.

## Customization 🎛️

Run `python3 extrac_analysis.py --help` for all options. The most useful ones:

- `--offset` / `--duration`: change what part of the track is analyzed (`--duration 0` analyzes the full file).
- `--bpm-min` / `--bpm-max`: tune the BPM normalization range (helps fix half/double-time mistakes).
- `--no-bpm-normalize`: disable BPM normalization if it makes a specific genre worse.
- `--start-bpm`: initial tempo guess for the tracker (often helps stability).
- `--sort`: `name` (default), `key`, `bpm`, `energy`.

## Limitations / gotchas ⚠️

- BPM and key detection are approximate, especially for:
  - live drummers / fluctuating tempo
  - tracks with long beatless intros/outros
  - tracks with key modulations
  - very percussive / atonal / heavily layered sections
- Some files may return `Unknown` key or `0.0` BPM if decoding or analysis fails (use `--verbose` to see per-file errors).

If BPM is consistently off for your music:

- Increase `--duration` (more context helps tempo estimation).
- Try a narrower `--bpm-min/--bpm-max` closer to your genre.
- If you’re getting 70 instead of 140 (or vice versa), keep normalization enabled (default).

## Contributing 🤝

PRs and issues are welcome. Nice next steps if you want to extend it:

- Write CSV/JSON output options
- Cache results so re-runs are faster
- Add a confidence score for BPM/key
- Improve key detection further (more robust profiles / tuning estimation)

## License 📜

Published under the MIT License. Feel free to use and adapt!
