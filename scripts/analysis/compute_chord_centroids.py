#!/usr/bin/env python3
"""
Compute chord centroids across all songs and layers.
Analyzes nearest neighbor patterns and music theory alignment.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.chord_aggregation import aggregate_frames_to_chords
from src.analysis.chord_centroids import (
    compute_chord_centroids,
    find_nearest_neighbors,
    check_music_theory_alignment,
    analyze_neighbor_patterns,
    create_chord_space_summary,
    compute_all_pairs_centroid_similarity
)


def process_song_layer(song_id: str, layer_dir: Path, data_dir: Path) -> Dict:
    """Process one song/layer to compute chord centroids."""
    # Load layer embeddings
    layer_file = layer_dir / f"{song_id}.npz"
    if not layer_file.exists():
        return None
    
    layer_data = np.load(layer_file, allow_pickle=True)
    embeddings = layer_data["emb"]
    emb_times = layer_data["times"]
    layer_name = layer_dir.name
    
    # Load chord labels
    base_path = data_dir / f"{song_id}.npz"
    if not base_path.exists():
        return None
    
    base_data = np.load(base_path, allow_pickle=True)
    chord_labels = base_data["frame_labels"]
    base_times = base_data["times"]
    
    # Align labels
    aligned_labels = []
    for t in emb_times:
        idx = np.argmin(np.abs(base_times - t))
        aligned_labels.append(chord_labels[idx])
    aligned_labels = np.array(aligned_labels)
    
    # Aggregate to chords
    chord_segments = aggregate_frames_to_chords(embeddings, aligned_labels, emb_times)
    
    # Compute centroids
    centroids = compute_chord_centroids(chord_segments)
    
    if len(centroids) < 2:
        return None  # Need at least 2 chords for NN
    
    # Find nearest neighbors (top-5)
    neighbors_df = find_nearest_neighbors(centroids, k=5, metric='cosine')
    neighbors_df = check_music_theory_alignment(neighbors_df)
    neighbors_df['song_id'] = song_id
    neighbors_df['layer_name'] = layer_name
    
    # Compute ALL pairs for heatmap visualization
    all_pairs_df = compute_all_pairs_centroid_similarity(centroids)
    all_pairs_df['song_id'] = song_id
    all_pairs_df['layer_name'] = layer_name
    
    # Compute metrics
    metrics = analyze_neighbor_patterns(neighbors_df)
    metrics['song_id'] = song_id
    metrics['layer_name'] = layer_name
    metrics['n_unique_chords'] = len(centroids)
    
    # Create summary
    summary = create_chord_space_summary(centroids, neighbors_df)
    summary['song_id'] = song_id
    summary['layer_name'] = layer_name
    
    return {
        "neighbors": neighbors_df,
        "all_pairs": all_pairs_df,
        "metrics": metrics,
        "summary": summary
    }


def main():
    parser = argparse.ArgumentParser(description="Compute chord centroids across all songs/layers")
    parser.add_argument("--mert_layers_dir", default="data/processed/mert_layers",
                       help="Directory with layer-wise MERT embeddings")
    parser.add_argument("--data_dir", default="data/processed",
                       help="Directory with chord labels")
    parser.add_argument("--output_dir", default="data/analysis/chord_centroids",
                       help="Output directory")
    parser.add_argument("--song_ids", nargs="*", default=None,
                       help="Specific songs to process")
    args = parser.parse_args()
    
    mert_layers_dir = Path(args.mert_layers_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find layers and songs
    layer_dirs = sorted([d for d in mert_layers_dir.iterdir() if d.is_dir() and d.name.startswith("L")])
    
    if args.song_ids:
        song_ids = args.song_ids
    else:
        song_files = sorted(layer_dirs[0].glob("*.npz"))
        song_ids = [f.stem for f in song_files]
    
    print(f"Processing {len(song_ids)} songs × {len(layer_dirs)} layers...")
    
    # Collect results
    all_neighbors = []
    all_pairs = []
    all_metrics = []
    all_summaries = []
    
    for song_id in tqdm(song_ids, desc="Songs"):
        for layer_dir in layer_dirs:
            try:
                result = process_song_layer(song_id, layer_dir, data_dir)
                if result is not None:
                    all_neighbors.append(result["neighbors"])
                    all_pairs.append(result["all_pairs"])
                    all_metrics.append(result["metrics"])
                    all_summaries.append(result["summary"])
            except Exception as e:
                print(f"Error: {song_id}/{layer_dir.name}: {e}")
                continue
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # 1. Nearest neighbors table (top-5 only)
    df_neighbors = pd.concat(all_neighbors, ignore_index=True)
    neighbors_path = output_dir / "chord_neighbors.csv"
    df_neighbors.to_csv(neighbors_path, index=False)
    print(f"✓ Saved {len(df_neighbors):,} neighbor records to {neighbors_path}")
    
    # 2. All-pairs centroid similarity (for heatmap!)
    df_all_pairs = pd.concat(all_pairs, ignore_index=True)
    all_pairs_path = output_dir / "chord_centroid_similarity.csv"
    df_all_pairs.to_csv(all_pairs_path, index=False)
    print(f"✓ Saved {len(df_all_pairs):,} all-pairs similarities to {all_pairs_path}")
    
    # 3. Per-song metrics
    df_metrics = pd.DataFrame(all_metrics)
    metrics_path = output_dir / "centroid_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"✓ Saved {len(df_metrics):,} song/layer metrics to {metrics_path}")
    
    # 4. Chord summaries
    df_summaries = pd.concat(all_summaries, ignore_index=True)
    summary_path = output_dir / "chord_summaries.csv"
    df_summaries.to_csv(summary_path, index=False)
    print(f"✓ Saved {len(df_summaries):,} chord summaries to {summary_path}")
    
    # Aggregate statistics
    print(f"\n📊 Overall Statistics:")
    print("=" * 80)
    
    # Layer-wise averages
    layer_stats = df_metrics.groupby('layer_name').agg({
        'pct_circle_of_fifths_in_top3': 'mean',
        'pct_relative_major_minor_in_top3': 'mean',
        'pct_same_quality_in_top3': 'mean',
        'avg_nearest_neighbor_similarity': 'mean',
    }).round(3)
    
    print("\nBy Layer (averaged across all songs):")
    print(layer_stats.to_string())
    
    # Save layer stats
    layer_stats_path = output_dir / "layer_stats.csv"
    layer_stats.to_csv(layer_stats_path)
    print(f"\n✓ Layer statistics saved to {layer_stats_path}")
    
    print(f"\n✅ Analysis complete!")
    print(f"\nOutput files:")
    print(f"  1. {neighbors_path} - Top-5 nearest neighbors")
    print(f"  2. {all_pairs_path} - All-pairs centroid similarity (for heatmap!)")
    print(f"  3. {metrics_path} - Per-song/layer metrics")
    print(f"  4. {summary_path} - Per-chord summaries")
    print(f"  5. {layer_stats_path} - Layer-wise averages")
    
    print(f"\n📊 Key Insights:")
    best_layer = layer_stats['pct_relative_major_minor_in_top3'].idxmax()
    best_score = layer_stats.loc[best_layer, 'pct_relative_major_minor_in_top3']
    print(f"  - Best layer for relative major/minor: {best_layer} ({best_score:.1%})")
    
    worst_similarity = layer_stats['avg_nearest_neighbor_similarity'].min()
    print(f"  - Lowest avg NN similarity: {worst_similarity:.3f}")
    
    if worst_similarity > 0.85:
        print(f"  ⚠️  All layers show high similarity (>{worst_similarity:.2f}) - weak chord discrimination")


if __name__ == "__main__":
    main()

