#!/usr/bin/env python3
"""
Test chord-level aggregation on a single song.
Shows how many chords vs frames, and preview of chord-to-chord similarity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from src.analysis.chord_aggregation import (
    aggregate_frames_to_chords,
    compute_chord_to_chord_similarity,
    compute_chord_statistics
)


def test_chord_aggregation(song_id: str, data_dir: str = "data/processed"):
    """Test chord aggregation on a single song."""
    data_dir = Path(data_dir)
    
    print(f"🎵 Testing Chord Aggregation: Song {song_id}")
    print("=" * 60)
    
    # Load data
    base_path = data_dir / f"{song_id}.npz"
    mert_path = data_dir / f"{song_id}_mert.npz"
    
    if not base_path.exists() or not mert_path.exists():
        print(f"❌ Error: Data not found for song {song_id}")
        return
    
    base_data = np.load(base_path, allow_pickle=True)
    mert_data = np.load(mert_path, allow_pickle=True)
    
    # Extract data
    chord_labels = base_data["frame_labels"]
    base_times = base_data["times"]
    embeddings = mert_data["emb"]
    emb_times = mert_data["times"]
    
    print(f"\n📊 Original Frame-Level Data:")
    print(f"  - Number of frames: {len(emb_times):,}")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Duration: {emb_times[-1]:.2f} seconds")
    print(f"  - Frame SSM size: {len(emb_times):,} × {len(emb_times):,}")
    print(f"  - Frame SSM total cells: {len(emb_times) ** 2:,}")
    
    # Align labels
    aligned_labels = []
    for t in emb_times:
        idx = np.argmin(np.abs(base_times - t))
        aligned_labels.append(chord_labels[idx])
    aligned_labels = np.array(aligned_labels)
    
    # Aggregate to chords
    print(f"\n⚙️  Aggregating to chord segments...")
    chord_segments = aggregate_frames_to_chords(embeddings, aligned_labels, emb_times)
    
    print(f"\n📦 Chord-Level Data:")
    print(f"  - Number of chord segments: {len(chord_segments)}")
    print(f"  - Unique chords: {chord_segments['chord_label'].nunique()}")
    print(f"  - Chord SSM size: {len(chord_segments)} × {len(chord_segments)}")
    print(f"  - Chord SSM total cells: {len(chord_segments) ** 2:,}")
    print(f"  - Size reduction: {len(emb_times) ** 2 / len(chord_segments) ** 2:.1f}x smaller!")
    
    # Show chord segments
    print(f"\n🎼 Chord Segments (first 10):")
    display_cols = ["chord_idx", "chord_label", "chord_root", "chord_quality", 
                    "start_time", "end_time", "duration", "n_frames"]
    print(chord_segments[display_cols].head(10).to_string(index=False))
    
    # Compute chord-to-chord similarity
    print(f"\n⚙️  Computing chord-to-chord similarity...")
    chord_similarity = compute_chord_to_chord_similarity(chord_segments)
    
    print(f"\n🔗 Chord-to-Chord Similarity:")
    print(f"  - Total pairs: {len(chord_similarity):,}")
    print(f"  - Transitions (adjacent chords): {chord_similarity['is_transition'].sum()}")
    print(f"  - Same chord pairs: {chord_similarity['same_chord'].sum()}")
    print(f"  - Same root pairs: {chord_similarity['same_root'].sum()}")
    print(f"  - Same quality pairs: {chord_similarity['same_quality'].sum()}")
    
    # Show some examples
    print(f"\n📊 Example Chord Transitions:")
    transitions = chord_similarity[chord_similarity["is_transition"] == 1].head(5)
    for _, row in transitions.iterrows():
        print(f"  {row['chord_i_label']:8s} → {row['chord_j_label']:8s}  "
              f"similarity: {row['cosine_similarity']:.3f}")
    
    # Compute statistics
    print(f"\n⚙️  Computing chord statistics...")
    chord_stats = compute_chord_statistics(chord_segments)
    
    print(f"\n📈 Chord Statistics:")
    print(chord_stats.sort_values("total_duration", ascending=False).to_string(index=False))
    
    print(f"\n💡 For Looker Studio:")
    print(f"  ✓ Use chord_similarity table for SSM heatmap")
    print(f"  ✓ Filter by chord_quality: {sorted(chord_segments['chord_quality'].unique())}")
    print(f"  ✓ Filter by chord_root: {sorted(chord_segments['chord_root'].unique())}")
    print(f"  ✓ Pivot table: chord_i_label (rows) × chord_j_label (cols) → avg(cosine_similarity)")
    print(f"  ✓ Much more manageable than {len(emb_times):,}×{len(emb_times):,} frame SSM!")
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test chord-level aggregation")
    parser.add_argument("song_id", help="Song ID to test")
    parser.add_argument("--data_dir", default="data/processed", help="Data directory")
    args = parser.parse_args()
    
    test_chord_aggregation(args.song_id, args.data_dir)

