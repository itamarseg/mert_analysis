#!/usr/bin/env python3
"""
Test chord centroid analysis on a single song.
Shows nearest neighbors in chord embedding space.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.analysis.chord_aggregation import aggregate_frames_to_chords, normalize_chord_label
from src.analysis.chord_centroids import (
    compute_chord_centroids,
    find_nearest_neighbors,
    check_music_theory_alignment,
    analyze_neighbor_patterns
)


def test_centroids(song_id: str, data_dir: str = "data/processed"):
    """Test chord centroid nearest neighbor analysis."""
    data_dir = Path(data_dir)
    
    print(f"🎵 Chord Centroid Analysis: Song {song_id}")
    print("=" * 80)
    
    # Load data
    base_path = data_dir / f"{song_id}.npz"
    mert_path = data_dir / f"{song_id}_mert.npz"
    
    if not base_path.exists() or not mert_path.exists():
        print(f"❌ Data not found for song {song_id}")
        return
    
    base_data = np.load(base_path, allow_pickle=True)
    mert_data = np.load(mert_path, allow_pickle=True)
    
    chord_labels = base_data["frame_labels"]
    base_times = base_data["times"]
    embeddings = mert_data["emb"]
    emb_times = mert_data["times"]
    
    print(f"\n📊 Data Overview:")
    print(f"  - Total frames: {len(emb_times):,}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Duration: {emb_times[-1]:.2f} seconds")
    
    # Align labels
    aligned_labels = []
    for t in emb_times:
        idx = np.argmin(np.abs(base_times - t))
        aligned_labels.append(chord_labels[idx])
    aligned_labels = np.array(aligned_labels)
    
    # Aggregate to chord segments
    chord_segments = aggregate_frames_to_chords(embeddings, aligned_labels, emb_times)
    
    print(f"\n📦 Chord Segments:")
    print(f"  - Number of segments: {len(chord_segments)}")
    print(f"  - Unique chords: {chord_segments['chord_label'].nunique()}")
    
    # Compute centroids
    print(f"\n⚙️  Computing chord centroids...")
    centroids = compute_chord_centroids(chord_segments)
    
    print(f"\n🎼 Chord Centroids ({len(centroids)} unique chords):")
    for chord_label, centroid in sorted(centroids.items()):
        print(f"  {chord_label:15s} - {centroid.n_occurrences:2d} occurrences, "
              f"{centroid.total_frames:5d} frames, "
              f"stability: {centroid.std_dev:.3f}")
    
    # Find nearest neighbors
    print(f"\n⚙️  Finding nearest neighbors (top 5)...")
    neighbors_df = find_nearest_neighbors(centroids, k=5, metric='cosine')
    
    # Check music theory alignment
    neighbors_df = check_music_theory_alignment(neighbors_df)
    
    # Display results for each chord
    print(f"\n🔍 Nearest Neighbors:")
    print("=" * 80)
    
    for query_chord in sorted(centroids.keys()):
        query_data = neighbors_df[neighbors_df['query_chord'] == query_chord].head(5)
        
        if len(query_data) == 0:
            continue
        
        print(f"\n{query_chord} ({centroids[query_chord].chord_quality}):")
        
        for _, row in query_data.iterrows():
            # Build relationship indicators
            indicators = []
            if row['is_same_root']:
                indicators.append("same root")
            if row['is_same_quality']:
                indicators.append("same quality")
            if row['is_circle_of_fifths']:
                indicators.append("⭐ circle of 5ths")
            if row['is_relative_major_minor']:
                indicators.append("⭐ relative maj/min")
            
            indicator_str = f" [{', '.join(indicators)}]" if indicators else ""
            
            print(f"  {row['neighbor_rank']}. {row['neighbor_chord']:15s} "
                  f"similarity: {row['similarity']:.3f}{indicator_str}")
    
    # Aggregate metrics
    print(f"\n📊 Music Theory Alignment Metrics:")
    print("=" * 80)
    metrics = analyze_neighbor_patterns(neighbors_df)
    
    print(f"  Top-3 Neighbors Analysis:")
    print(f"    - Same root: {metrics['pct_same_root_in_top3']:.1%}")
    print(f"    - Same quality: {metrics['pct_same_quality_in_top3']:.1%}")
    print(f"    - Circle of fifths: {metrics['pct_circle_of_fifths_in_top3']:.1%}")
    print(f"    - Relative major/minor: {metrics['pct_relative_major_minor_in_top3']:.1%}")
    print(f"\n  Average Similarities:")
    print(f"    - Nearest neighbor: {metrics['avg_nearest_neighbor_similarity']:.3f}")
    print(f"    - 5th neighbor: {metrics['avg_rank5_similarity']:.3f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if metrics['pct_circle_of_fifths_in_top3'] > 0.2:
        print(f"  ✅ Model captures circle-of-fifths relationships!")
    else:
        print(f"  ❌ Circle-of-fifths neighbors are rare ({metrics['pct_circle_of_fifths_in_top3']:.1%})")
    
    if metrics['pct_relative_major_minor_in_top3'] > 0.15:
        print(f"  ✅ Model captures relative major-minor relationships!")
    else:
        print(f"  ❌ Relative major-minor neighbors are rare ({metrics['pct_relative_major_minor_in_top3']:.1%})")
    
    if metrics['pct_same_quality_in_top3'] > 0.5:
        print(f"  ✅ Model groups chords by quality (major/minor)")
    else:
        print(f"  ~ Model doesn't strongly cluster by quality")
    
    if metrics['avg_nearest_neighbor_similarity'] > 0.8:
        print(f"  ⚠️  All chords very similar - weak discrimination")
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💾 Next step: Run full analysis across all songs and layers")
    print(f"   python scripts/analysis/compute_chord_centroids.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test chord centroid analysis")
    parser.add_argument("song_id", help="Song ID to test")
    parser.add_argument("--data_dir", default="data/processed", help="Data directory")
    args = parser.parse_args()
    
    test_centroids(args.song_id, args.data_dir)

