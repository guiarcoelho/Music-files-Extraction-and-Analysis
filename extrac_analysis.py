import argparse
import concurrent.futures
import os
import re
import sys
import time

import librosa
import numpy as np

# Mapping: Traditional Key to Camelot Notation
CAMELOT_MAP = {
    'C Major': '8B', 'C Minor': '5A', 'C# Major': '3B', 'C# Minor': '12A',
    'D Major': '10B', 'D Minor': '7A', 'D# Major': '5B', 'D# Minor': '2A',
    'E Major': '12B', 'E Minor': '9A', 'F Major': '7B', 'F Minor': '4A',
    'F# Major': '2B', 'F# Minor': '11A', 'G Major': '9B', 'G Minor': '6A',
    'G# Major': '4B', 'G# Minor': '1A', 'A Major': '11B', 'A Minor': '8A',
    'A# Major': '6B', 'A# Minor': '3A', 'B Major': '1B', 'B Minor': '10A'
}

KEY_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles (C major / C minor), used via correlation.
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)


def _as_scalar(x, default=0.0):
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def normalize_tempo(tempo, bpm_min, bpm_max):
    tempo = float(tempo)
    if not np.isfinite(tempo) or tempo <= 0:
        return 0.0
    if bpm_min <= 0 or bpm_max <= 0 or bpm_min >= bpm_max:
        return tempo
    while tempo < bpm_min:
        tempo *= 2.0
    while tempo > bpm_max:
        tempo /= 2.0
    return tempo


def estimate_bpm_from_onset(onset_env, sr, start_bpm, bpm_min, bpm_max, normalize=True):
    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=start_bpm,
        tightness=100,
    )
    tempo = _as_scalar(tempo, default=0.0)
    if normalize:
        tempo = normalize_tempo(tempo, bpm_min=bpm_min, bpm_max=bpm_max)
    return round(float(tempo), 1)


def estimate_camelot_key(y_harm, sr):
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    if not np.any(np.isfinite(chroma_avg)) or np.allclose(chroma_avg, 0):
        return "Unknown"

    chroma_avg = np.maximum(chroma_avg, 0)
    chroma_norm = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-12)

    major_scores = np.array(
        [np.dot(chroma_norm, np.roll(KRUMHANSL_MAJOR, i)) for i in range(12)]
    )
    minor_scores = np.array(
        [np.dot(chroma_norm, np.roll(KRUMHANSL_MINOR, i)) for i in range(12)]
    )

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))
    if float(major_scores[best_major]) >= float(minor_scores[best_minor]):
        key_name = f"{KEY_LABELS[best_major]} Major"
    else:
        key_name = f"{KEY_LABELS[best_minor]} Minor"

    return CAMELOT_MAP.get(key_name, "Unknown")


def get_audio_features_with_error(
    file_path,
    *,
    sr=22050,
    offset=10.0,
    duration=120.0,
    start_bpm=120.0,
    bpm_min=70.0,
    bpm_max=200.0,
    normalize_bpm=True,
):
    try:
        y, sr = librosa.load(
            file_path, sr=sr, offset=offset, duration=duration)

        # Optimization (1): compute harmonic/percussive separation once.
        y_harm, y_perc = librosa.effects.hpss(y)

        # Optimization (2): compute onset envelope once and reuse it for BPM + energy.
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)

        bpm = estimate_bpm_from_onset(
            onset_env,
            sr,
            start_bpm=start_bpm,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            normalize=normalize_bpm,
        )
        camelot_key = estimate_camelot_key(y_harm, sr)

        # Energy features (simple heuristics)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flux = float(np.mean(onset_env))

        raw_energy = (rms * 40) + (flux * 1.5) + (centroid / 2500)
        return bpm, camelot_key, float(raw_energy), None
    except Exception as exc:
        return 0.0, "Unknown", 0.0, str(exc)


def _analyze_one_task(task):
    (
        display_name,
        path,
        sr,
        offset,
        duration,
        start_bpm,
        bpm_min,
        bpm_max,
        normalize_bpm,
    ) = task
    bpm, camelot, raw_e, error = get_audio_features_with_error(
        path,
        sr=sr,
        offset=offset,
        duration=duration,
        start_bpm=start_bpm,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        normalize_bpm=normalize_bpm,
    )
    return display_name, bpm, camelot, raw_e, error


def iter_audio_files(music_dir, valid_exts, recursive):
    if recursive:
        for root, _, files in os.walk(music_dir):
            for name in sorted(files):
                if not name.lower().endswith(valid_exts):
                    continue
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, music_dir)
                yield rel_path, abs_path
    else:
        for name in sorted(os.listdir(music_dir)):
            if not name.lower().endswith(valid_exts):
                continue
            abs_path = os.path.join(music_dir, name)
            yield name, abs_path


def camelot_sort_key(camelot_key):
    match = re.fullmatch(r"(\d{1,2})([AB])", str(camelot_key).strip())
    if not match:
        return (1, 99, "Z")
    return (0, int(match.group(1)), match.group(2))


def generate_report(
    music_dir,
    output_dir,
    *,
    recursive=False,
    sr=22050,
    offset=10.0,
    duration=120.0,
    start_bpm=120.0,
    bpm_min=70.0,
    bpm_max=200.0,
    normalize_bpm=True,
    jobs=1,
    output_filename="music_comprehensive_report.txt",
    sort_by="name",
    verbose=False,
):
    valid_exts = ('.mp3', '.wav', '.flac', '.m4a')
    if not os.path.isdir(music_dir):
        print(f"Music folder not found: {music_dir}", file=sys.stderr)
        return
    if not output_dir.strip():
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    files = list(iter_audio_files(
        music_dir, valid_exts=valid_exts, recursive=recursive))
    total_files = len(files)
    if total_files == 0:
        print("No music files found.")
        return

    raw_results = []
    start_time = time.time()
    print(f"--- Starting Analysis of {total_files} files ---")
    failures = 0

    jobs = int(jobs)
    if jobs == 0:
        jobs = os.cpu_count() or 1

    tasks = [
        (
            display_name,
            path,
            sr,
            offset,
            duration,
            start_bpm,
            bpm_min,
            bpm_max,
            normalize_bpm,
        )
        for (display_name, path) in files
    ]

    if jobs <= 1 or total_files <= 1:
        for index, task in enumerate(tasks, start=1):
            display_name = task[0]
            percent = (index / total_files) * 100
            print(
                f"[{percent:.1f}%] ({index}/{total_files}) Analysing: {display_name[:40]}...          ",
                end="\r",
            )
            display_name, bpm, camelot, raw_e, error = _analyze_one_task(task)
            if error:
                failures += 1
                if verbose:
                    print(
                        f"\nFailed to analyse {display_name}: {error}", file=sys.stderr)
            raw_results.append(
                {'name': display_name, 'bpm': bpm, 'key': camelot, 'raw_energy': raw_e})
    else:
        # Optimization (6): parallelize across tracks using multiple processes.
        completed = 0
        max_workers = min(jobs, total_files)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(
                _analyze_one_task, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_name):
                display_name = future_to_name[future]
                completed += 1
                percent = (completed / total_files) * 100
                print(
                    f"[{percent:.1f}%] ({completed}/{total_files}) Analysing: {display_name[:40]}...          ",
                    end="\r",
                )
                try:
                    display_name, bpm, camelot, raw_e, error = future.result()
                except Exception as exc:  # pragma: no cover
                    failures += 1
                    if verbose:
                        print(
                            f"\nFailed to analyse {display_name}: {exc}", file=sys.stderr)
                    raw_results.append(
                        {'name': display_name, 'bpm': 0.0, 'key': "Unknown", 'raw_energy': 0.0})
                    continue

                if error:
                    failures += 1
                    if verbose:
                        print(
                            f"\nFailed to analyse {display_name}: {error}", file=sys.stderr)
                raw_results.append(
                    {'name': display_name, 'bpm': bpm, 'key': camelot, 'raw_energy': raw_e})

    # --- ENERGY NORMALIZATION (Reliability Step) ---
    # Convert raw scores into a 1-10 scale based on the range of the current folder
    energies = [r['raw_energy'] for r in raw_results if r['raw_energy'] > 0]
    if energies:
        min_e, max_e = min(energies), max(energies)
        for r in raw_results:
            if r['raw_energy'] == 0:
                r['energy'] = 0
            else:
                # Map raw score to 1-10 range
                norm = (r['raw_energy'] - min_e) / (max_e - min_e + 1e-6)
                r['energy'] = round(1 + (norm * 9))
    else:
        for r in raw_results:
            r['energy'] = 0

    if sort_by == "name":
        raw_results.sort(key=lambda x: x['name'].lower())
    elif sort_by == "key":
        raw_results.sort(key=lambda x: (camelot_sort_key(
            x["key"]), x["bpm"], x["name"].lower()))
    elif sort_by == "bpm":
        raw_results.sort(key=lambda x: (
            x["bpm"], camelot_sort_key(x["key"]), x["name"].lower()))
    elif sort_by == "energy":
        raw_results.sort(key=lambda x: (x["energy"], x["name"].lower()))

    # Write Report
    report_path = os.path.join(output_dir, output_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"{'FILE NAME':<50} | {'BPM':<8} | {'KEY':<8} | {'ENERGY (1-10)'}\n")
        f.write("-" * 90 + "\n")
        for r in raw_results:
            f.write(
                f"{r['name'][:49]:<50} | {r['bpm']:<8} | {r['key']:<8} | {r['energy']}\n")

    print(f"\n\nSuccess! Completed in {(time.time() - start_time)/60:.1f}m.")
    if failures:
        print(f"Failed to analyse: {failures}/{total_files}")
    print(f"Report location: {report_path}")


# --- PATHS ---
MUSIC_LOCATION = ""
REPORT_LOCATION = "out"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Analyze a music folder for BPM, Camelot key, and an energy score.",
    )
    parser.add_argument("--music-dir", default=MUSIC_LOCATION,
                        help="Folder containing audio files.")
    parser.add_argument("--output-dir", default=REPORT_LOCATION,
                        help="Folder to write the report into.")
    parser.add_argument("--recursive", action="store_true",
                        help="Scan music-dir recursively.")
    parser.add_argument("--sr", type=int, default=22050,
                        help="Sample rate used for analysis.")
    parser.add_argument("--offset", type=float, default=10.0,
                        help="Start time (seconds) for analysis.")
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Amount of audio (seconds) to analyze (use 0 for full file).",
    )
    parser.add_argument(
        "--start-bpm",
        type=float,
        default=120.0,
        help="Initial BPM guess for the tracker (helps stabilize results).",
    )
    parser.add_argument("--bpm-min", type=float, default=70.0,
                        help="Minimum BPM for normalization.")
    parser.add_argument("--bpm-max", type=float, default=200.0,
                        help="Maximum BPM for normalization.")
    parser.add_argument(
        "--no-bpm-normalize",
        action="store_true",
        help="Disable half/double-tempo normalization into the bpm-min/bpm-max range.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes (use 0 for all CPU cores).",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "key", "bpm", "energy"],
        default="name",
        help="How to sort rows in the output report.",
    )
    parser.add_argument(
        "--output-filename",
        default="music_comprehensive_report.txt",
        help="Output report filename (written inside output-dir).",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-file errors to stderr.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.music_dir:
        print("Missing --music-dir (or set MUSIC_LOCATION at the bottom of the script).", file=sys.stderr)
        return 2

    duration = None if args.duration == 0 else args.duration
    generate_report(
        args.music_dir,
        args.output_dir,
        recursive=args.recursive,
        sr=args.sr,
        offset=args.offset,
        duration=duration,
        start_bpm=args.start_bpm,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        normalize_bpm=(not args.no_bpm_normalize),
        jobs=args.jobs,
        output_filename=args.output_filename,
        sort_by=args.sort,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
