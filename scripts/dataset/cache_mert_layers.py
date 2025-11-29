#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, librosa
from pathlib import Path
from src.mert.featurize import MertExtractor, MertConfig

def main():
    ap = argparse.ArgumentParser(description="Cache MERT embeddings per layer into layer-named folders.")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--layers", nargs="+", default=["-1","12","18"], help="Layers to export, e.g. -1 12 18 or 'all'")
    ap.add_argument("--chunk", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--out_root", default="data/processed/mert_layers")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)

    # Decide layer setting for featurizer
    layer_arg = "all" if (len(args.layers)==1 and args.layers[0]=="all") else [int(x) for x in args.layers]
    mert = MertExtractor(MertConfig(layer=layer_arg, chunk_secs=args.chunk, overlap_secs=args.overlap))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in man.iterrows():
        sid = str(row["song_id"])
        y, sr = librosa.load(row["audio_path"], sr=None, mono=True)
        pack = mert.encode(y, sr)  # returns {"times":..., "emb_L<idx>": ...}

        # Save times once per song (common across layers we exported)
        # Also keep a small pointer file at root for convenience.
        # Each layer gets its own file in L<idx>/<sid>.npz
        for k in pack:
            if not k.startswith("emb_L"): 
                continue
            layer_name = k.replace("emb_", "")  # e.g., L-1, L12
            layer_dir = out_root / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                layer_dir / f"{sid}.npz",
                emb=pack[k],
                times=pack["times"],
                sr=pack["sr"],
                meta={"audio": row["audio_path"], "layer": layer_name}
            )
            print(f"cached {sid} → {layer_name}/{sid}.npz  {pack[k].shape}")

if __name__ == "__main__":
    main()
