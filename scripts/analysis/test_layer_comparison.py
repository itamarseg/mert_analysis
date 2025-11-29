#!/usr/bin/env python3
"""
Test that all layers are being processed correctly.
Shows how chord aggregation works across different MERT layers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from src.analysis.chord_aggregation import (
    aggregate_frames_to_chords,
    compute_chord_to_chord_similarity
)


def test_multiple_layers(song_id: str, 
                         mert_layers_dir: str = "data/processed/mert_layers",
                         data_dir: str = "data/processed"):
    """Test chord aggregation across all MERT layers."""
    mert_layers_dir = Path(mert_layers_dir)
    data_dir = Path(data_dir)
    
    print(f"🎵 Testing Layer Comparison: Song {song_id}")
    print("=" * 80)
    
    # Find all layer directories
    layer_dirs = sorted([d for d in mert_layers_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("L")])
    
    if len(layer_dirs) == 0:
        print(f"❌ No layer directories found in {mert_layers_dir}")
        print("Expected directories like: L-1, L12, L18")
        return
    
    print(f"\n📁 Found {len(layer_dirs)} layers: {[d.name for d in layer_dirs]}")
    
    # Load chord labels (shared across layers)
    base_path = data_dir / f"{song_id}.npz"
    if not base_path.exists():
        print(f"❌ No chord labels found: {base_path}")
        return
    
    base_data = np.load(base_path, allow_pickle=True)
    chord_labels = base_data["frame_labels"]
    base_times = base_data["times"]
    
    # Process each layer
    results = []
    
    for layer_dir in layer_dirs:
        layer_file = layer_dir / f"{song_id}.npz"
        if not layer_file.exists():
            print(f"⚠️  Skipping {layer_dir.name}: file not found")
            continue
        
        # Load embeddings
        layer_data = np.load(layer_file, allow_pickle=True)
        embeddings = layer_data["emb"]
        emb_times = layer_data["times"]
        
        # Align labels
        aligned_labels = []
        for t in emb_times:
            idx = np.argmin(np.abs(base_times - t))
            aligned_labels.append(chord_labels[idx])
        aligned_labels = np.array(aligned_labels)
        
        # Aggregate to chords
        chord_segments = aggregate_frames_to_chords(embeddings, aligned_labels, emb_times)
        
        # Compute similarity
        chord_similarity = compute_chord_to_chord_similarity(chord_segments)
        
        # Calculate metrics
        self_similarity = chord_similarity[
            (chord_similarity["same_chord"] == 1) & 
            (chord_similarity["is_diagonal"] == 0)
        ]["cosine_similarity"].mean()
        
        maj_to_min_sim = chord_similarity[
            (chord_similarity["chord_i_quality"] == "maj") &
            (chord_similarity["chord_j_quality"].str.contains("min"))
        ]["cosine_similarity"].mean()
        
        transition_sim = chord_similarity[
            chord_similarity["is_transition"] == 1
        ]["cosine_similarity"].mean()
        
        results.append({
            "layer": layer_dir.name,
            "n_chords": len(chord_segments),
            "n_frames": len(embeddings),
            "self_similarity": self_similarity,
            "maj_to_min_similarity": maj_to_min_sim,
            "transition_similarity": transition_sim,
            "avg_chord_duration": chord_segments["duration"].mean(),
        })
    
    # Display results
    df = pd.DataFrame(results)
    
    print(f"\n📊 Layer Comparison Results:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print(f"\n💡 Interpretation:")
    print(f"  - Self-similarity: Higher = chords more consistent within layer")
    print(f"  - Maj→Min similarity: Lower = better separation of major/minor")
    print(f"  - Transition similarity: Shows how smooth chord changes are")
    
    # Best layer analysis
    print(f"\n🏆 Best Layers:")
    best_self_sim = df.loc[df["self_similarity"].idxmax()]
    best_separation = df.loc[df["maj_to_min_similarity"].idxmin()]
    
    print(f"  - Highest self-similarity: {best_self_sim['layer']} ({best_self_sim['self_similarity']:.3f})")
    print(f"  - Best maj/min separation: {best_separation['layer']} ({best_separation['maj_to_min_similarity']:.3f})")
    
    print(f"\n✅ All {len(results)} layers processed successfully!")
    print(f"\n📤 When exported to BigQuery:")
    print(f"  - Each layer gets separate records with 'layer_name' field")
    print(f"  - Total records per song: {len(chord_segments)} × {len(results)} = {len(chord_segments) * len(results)}")
    print(f"  - Filter in Looker Studio: WHERE layer_name = 'L-1'")
    print(f"\n📚 See LAYER_ANALYSIS_GUIDE.md for Looker Studio setup")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test layer comparison")
    parser.add_argument("song_id", help="Song ID to test")
    parser.add_argument("--mert_layers_dir", default="data/processed/mert_layers",
                       help="Directory with layer-wise embeddings")
    parser.add_argument("--data_dir", default="data/processed",
                       help="Directory with chord labels")
    args = parser.parse_args()
    
    test_multiple_layers(args.song_id, args.mert_layers_dir, args.data_dir)

