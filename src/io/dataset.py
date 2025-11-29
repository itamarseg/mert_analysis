from __future__ import annotations
from pathlib import Path
import pandas as pd

def make_manifest(audio_glob: str) -> pd.DataFrame:
    rows = []
    for mp3 in sorted(Path().glob(audio_glob)):
        lab = next(mp3.parent.glob("*.lab"), None)
        if not lab:  # skip if missing GT
            continue
        rows.append({
            "song_id": mp3.parent.name,      # "01", "02", ...
            "audio_path": mp3.as_posix(),
            "lab_path": lab.as_posix(),
        })
    return pd.DataFrame(rows)
