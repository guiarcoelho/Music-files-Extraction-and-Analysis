import os
import librosa
import numpy as np
import time

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


def get_audio_info(file_path):
    try:
        # Loading audio (using 120s as per your preference)
        y, sr = librosa.load(file_path, sr=22050, offset=10)

        # 1. Detect BPM (Fixed the Deprecation Warning here)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # tempo is often returned as an array, e.g., [120.0]
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]

        bpm = round(float(tempo), 1)

        # 2. Detect Key
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_avg)

        minor_third_idx = (key_idx + 3) % 12
        major_third_idx = (key_idx + 4) % 12
        scale = "Minor" if chroma_avg[minor_third_idx] > chroma_avg[major_third_idx] else "Major"

        camelot_key = CAMELOT_MAP.get(
            f"{KEY_LABELS[key_idx]} {scale}", "Unknown")
        return bpm, camelot_key
    except Exception:
        return 0.0, "Unknown"


def generate_report(music_dir, output_dir):
    valid_exts = ('.mp3', '.wav', '.flac', '.m4a')

    if not output_dir.strip():
        output_dir = os.getcwd()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = [f for f in os.listdir(
        music_dir) if f.lower().endswith(valid_exts)]
    total_files = len(all_files)

    if total_files == 0:
        print("No music files found.")
        return

    results = []
    start_time = time.time()
    print(f"--- Starting Analysis of {total_files} files ---")

    for index, file in enumerate(all_files, start=1):
        percent = (index / total_files) * 100
        # The space at the end of the print clears old characters when filenames change length
        print(
            f"[{percent:.1f}%] ({index}/{total_files}) Analysing: {file[:40]}...          ", end="\r")

        path = os.path.join(music_dir, file)
        bpm, camelot = get_audio_info(path)
        results.append({'name': file, 'bpm': bpm, 'key': camelot})

    # Sort results
    results.sort(key=lambda x: (x['key'], x['bpm']))

    # Write to file
    report_path = os.path.join(output_dir, "music_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'CAMELOT':<10} | {'BPM':<8} | {'FILE NAME'}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['key']:<10} | {r['bpm']:<8} | {r['name']}\n")

    end_time = time.time()
    total_duration = end_time - start_time

    print(
        f"\n\nSuccess! Analysis complete in {total_duration/60:.1f} minutes.")
    print(f"Report location: {report_path}")


# --- PATHS ---
MUSIC_LOCATION = "/Users/guiarcoelho/Desktop/Eletrónica/"
REPORT_LOCATION = "/Users/guiarcoelho/Desktop/Music Name Extraction and Analysis/out/"

generate_report(MUSIC_LOCATION, REPORT_LOCATION)
