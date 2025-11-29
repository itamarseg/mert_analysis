import yaml, numpy as np, pandas as pd
from pathlib import Path
from src.io.labs import read_lab, align_labels_to_frames, collapse_frames_to_segments

cfg = yaml.safe_load(open("configs/dataset.yaml"))
man = pd.read_csv("data/processed/manifest.csv")

for _, row in man.iterrows():
    song_id = row["song_id"]
    npz = Path(cfg["cache_dir"]) / f"{song_id}.npz"
    data = np.load(npz, allow_pickle=True)
    times = data["times"]
    lab_df = read_lab(row["lab_path"])
    frame_labels = align_labels_to_frames(times, lab_df)

    # save augmented cache
    np.savez_compressed(npz, **{k:data[k] for k in data.files}, frame_labels=np.array(frame_labels, dtype=object))

    # friendly CSV to eyeball
    csv_path = Path(cfg["cache_dir"]) / f"{song_id}_frames.csv"
    pd.DataFrame({"time": times, "label": frame_labels}).to_csv(csv_path, index=False)

    # collapsed lab (pred==gt here; just aligned to frame grid)
    segs = collapse_frames_to_segments(times, frame_labels)
    out_lab = Path(cfg["cache_dir"]) / f"{song_id}_aligned.lab"
    with open(out_lab, "w") as f:
        for s, e, lb in segs:
            f.write(f"{s:.6f}\t{e:.6f}\t{lb}\n")

    print("aligned:", song_id, "→", csv_path.name, "and", out_lab.name)
