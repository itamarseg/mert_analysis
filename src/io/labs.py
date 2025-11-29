from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable

def read_lab(path: str) -> pd.DataFrame:
    """Read 3-column lab: start end label (space/tab separated)."""
    df = pd.read_csv(path, sep=r"\s+", names=["start", "end", "label"], engine="python")
    df["start"] = df["start"].astype(float)
    df["end"]   = df["end"].astype(float)
    df["label"] = df["label"].astype(str).str.strip()
    return df

def align_labels_to_frames(frame_times: np.ndarray, lab_df: pd.DataFrame) -> List[str]:
    labels = np.full(frame_times.shape, "N", dtype=object)
    j = 0
    for i, t in enumerate(frame_times):
        while j < len(lab_df) and t >= lab_df.iloc[j]["end"]:
            j += 1
        if j < len(lab_df) and lab_df.iloc[j]["start"] <= t < lab_df.iloc[j]["end"]:
            labels[i] = lab_df.iloc[j]["label"]
    return labels.tolist()

def collapse_frames_to_segments(times: np.ndarray, labels: List[str]) -> List[Tuple[float,float,str]]:
    segs = []
    start = times[0]
    last = labels[0]
    for i in range(1, len(times)):
        if labels[i] != last:
            segs.append((float(start), float(times[i]), str(last)))
            start = times[i]
            last = labels[i]
    segs.append((float(start), float(times[-1]), str(last)))
    return segs

def align_labels_to_frames_indices(
    frame_times: np.ndarray,
    lab_df: pd.DataFrame,
    to_indices: Callable[[str], Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-frame (root_idx, quality_idx).
    - Uses `to_indices(label)` to map labels like 'C:maj' or 'N' to ints.
    - Fills frames outside any interval with 'N'.
    """
    n = len(frame_times)
    # ask converter once to find the 'N' indices
    n_root, n_qual = to_indices("N")

    root = np.full(n, n_root, dtype=int)
    qual = np.full(n, n_qual, dtype=int)

    j = 0
    for i, t in enumerate(frame_times):
        while j < len(lab_df) and t >= float(lab_df.iloc[j]["end"]):
            j += 1
        if j < len(lab_df) and float(lab_df.iloc[j]["start"]) <= t < float(lab_df.iloc[j]["end"]):
            r, q = to_indices(str(lab_df.iloc[j]["label"]))
            root[i], qual[i] = r, q
    return root, qual
