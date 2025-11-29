"""
Aggregate frame-level embeddings and metrics to chord-level.
Creates chord-to-chord transition and similarity analysis.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def normalize_chord_label(chord_label: str) -> str:
    """
    Normalize chord label by removing redundant quality suffixes.

    Examples:
        "A:maj" -> "A"
        "E:min" -> "E:min"  (keep :min to distinguish from E major)
        "D/F#" -> "D/F#"  (keep slash notation)
        "G7" -> "G7"
    """
    # Remove ":maj" suffix (redundant - major is default)
    if chord_label.endswith(":maj"):
        chord_label = chord_label[:-4]

    return chord_label


def parse_chord_quality(chord_label: str) -> Dict[str, str]:
    """
    Parse chord label into root, quality, bass.

    Examples:
        "C" -> {root: "C", quality: "maj", bass: None}
        "Am" -> {root: "A", quality: "min", bass: None}
        "G7" -> {root: "G", quality: "7", bass: None}
        "D/F#" -> {root: "D", quality: "maj", bass: "F#"}
        "N" -> {root: "N", quality: "none", bass: None}
    """
    if not chord_label or chord_label in ["N", "X", "none"]:
        return {"root": "N", "quality": "none", "bass": None, "full": chord_label}

    # Split on / for inversions
    parts = chord_label.split("/")
    main = parts[0]
    bass = parts[1] if len(parts) > 1 else None
    has_slash = len(parts) > 1  # Flag for slash chord

    # Parse root and quality
    root = main[0]
    if len(main) > 1 and main[1] in ["#", "b"]:
        root = main[:2]
        quality_part = main[2:]
    else:
        quality_part = main[1:] if len(main) > 1 else ""

    # Normalize quality_part: remove leading colon if present
    # E.g., ":maj" -> "maj", ":min" -> "min"
    if quality_part.startswith(":"):
        quality_part = quality_part[1:]

    # Determine base quality
    if not quality_part or quality_part == "" or quality_part == "maj":
        base_quality = "maj"
    elif quality_part == "min" or quality_part.lower().startswith("m"):
        base_quality = "min"
    elif "7" in quality_part:
        base_quality = "7" if not quality_part.lower().startswith("m") else "min7"
    elif "dim" in quality_part.lower():
        base_quality = "dim"
    elif "aug" in quality_part.lower():
        base_quality = "aug"
    else:
        base_quality = quality_part

    # Add "_slash" suffix for inversions
    # E.g., "D/F#" becomes quality="maj_slash" instead of "maj"
    if has_slash:
        quality = f"{base_quality}_slash"
    else:
        quality = base_quality

    return {"root": root, "quality": quality, "bass": bass, "full": chord_label}


def aggregate_frames_to_chords(
    embeddings: np.ndarray, frame_labels: np.ndarray, times: np.ndarray
) -> pd.DataFrame:
    """
    Aggregate frame-level embeddings to chord segments.

    Args:
        embeddings: (T, D) frame embeddings
        frame_labels: (T,) chord labels per frame
        times: (T,) timestamps

    Returns:
        DataFrame with columns:
            - chord_idx: segment index
            - chord_label: chord name
            - start_time: segment start
            - end_time: segment end
            - duration: segment length
            - embedding_mean: mean embedding vector
            - embedding_std: std of embeddings (stability metric)
    """
    segments = []
    current_chord = None
    start_idx = 0

    for i, label in enumerate(frame_labels):
        if label != current_chord:
            # Save previous segment
            if current_chord is not None:
                seg_embeddings = embeddings[start_idx:i]
                parsed = parse_chord_quality(current_chord)

                segments.append(
                    {
                        "chord_idx": len(segments),
                        "chord_label": normalize_chord_label(
                            current_chord
                        ),  # Normalized label
                        "chord_root": parsed["root"],
                        "chord_quality": parsed["quality"],
                        "chord_bass": parsed["bass"],
                        "start_time": times[start_idx],
                        "end_time": times[i - 1],
                        "duration": times[i - 1] - times[start_idx],
                        "n_frames": i - start_idx,
                        "embedding_mean": seg_embeddings.mean(axis=0),
                        "embedding_std": seg_embeddings.std(
                            axis=0
                        ).mean(),  # avg std across dims
                    }
                )

            # Start new segment
            current_chord = label
            start_idx = i

    # Handle last segment
    if current_chord is not None:
        seg_embeddings = embeddings[start_idx:]
        parsed = parse_chord_quality(current_chord)

        segments.append(
            {
                "chord_idx": len(segments),
                "chord_label": normalize_chord_label(current_chord),  # Normalized label
                "chord_root": parsed["root"],
                "chord_quality": parsed["quality"],
                "chord_bass": parsed["bass"],
                "start_time": times[start_idx],
                "end_time": times[-1],
                "duration": times[-1] - times[start_idx],
                "n_frames": len(embeddings) - start_idx,
                "embedding_mean": seg_embeddings.mean(axis=0),
                "embedding_std": seg_embeddings.std(axis=0).mean(),
            }
        )

    return pd.DataFrame(segments)


def compute_chord_to_chord_similarity(chord_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute chord-to-chord similarity matrix and transitions.

    Returns:
        DataFrame with columns:
            - song_id
            - chord_i_idx, chord_i_label, chord_i_root, chord_i_quality
            - chord_j_idx, chord_j_label, chord_j_root, chord_j_quality
            - cosine_similarity
            - is_transition (1 if chords are adjacent in time)
            - same_root, same_quality (binary indicators)
    """
    # Extract mean embeddings
    embeddings = np.vstack(chord_segments["embedding_mean"].values)

    # Compute all-to-all similarity
    sim_matrix = cosine_similarity(embeddings)

    # Create pairwise records
    records = []
    n_chords = len(chord_segments)

    for i in range(n_chords):
        for j in range(i, n_chords):  # Upper triangular (symmetric)
            chord_i = chord_segments.iloc[i]
            chord_j = chord_segments.iloc[j]

            is_transition = j == i + 1  # Adjacent chords
            same_root = chord_i["chord_root"] == chord_j["chord_root"]
            same_quality = chord_i["chord_quality"] == chord_j["chord_quality"]
            same_chord = chord_i["chord_label"] == chord_j["chord_label"]

            records.append(
                {
                    "chord_i_idx": int(chord_i["chord_idx"]),
                    "chord_i_label": chord_i["chord_label"],
                    "chord_i_root": chord_i["chord_root"],
                    "chord_i_quality": chord_i["chord_quality"],
                    "chord_i_duration": chord_i["duration"],
                    "chord_j_idx": int(chord_j["chord_idx"]),
                    "chord_j_label": chord_j["chord_label"],
                    "chord_j_root": chord_j["chord_root"],
                    "chord_j_quality": chord_j["chord_quality"],
                    "chord_j_duration": chord_j["duration"],
                    "cosine_similarity": float(sim_matrix[i, j]),
                    "is_transition": int(is_transition),
                    "is_diagonal": int(i == j),
                    "same_root": int(same_root),
                    "same_quality": int(same_quality),
                    "same_chord": int(same_chord),
                    "time_distance": abs(chord_j["start_time"] - chord_i["start_time"]),
                }
            )

    return pd.DataFrame(records)


def compute_chord_statistics(chord_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics per unique chord.

    Returns:
        DataFrame with chord-level stats:
            - chord_label, root, quality
            - occurrences: how many times it appears
            - total_duration: total time
            - avg_duration: average segment length
            - avg_embedding_std: average stability
    """
    stats = (
        chord_segments.groupby(["chord_label", "chord_root", "chord_quality"])
        .agg(
            {
                "chord_idx": "count",  # occurrences
                "duration": ["sum", "mean", "std"],
                "embedding_std": "mean",
                "n_frames": "sum",
            }
        )
        .reset_index()
    )

    stats.columns = [
        "chord_label",
        "chord_root",
        "chord_quality",
        "occurrences",
        "total_duration",
        "avg_duration",
        "std_duration",
        "avg_embedding_stability",
        "total_frames",
    ]

    return stats
