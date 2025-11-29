#!/usr/bin/env python3
"""
Quick test script to analyze a single song.
Useful for debugging and understanding the analysis pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.analysis.ssm import (
    compute_embedding_ssm,
    compute_chord_ssm,
    compute_ssm_alignment_metrics,
    compute_structure_metrics
)


def test_song(song_id: str, data_dir: str = "data/processed"):
    """Test SSM analysis on a single song."""
    data_dir = Path(data_dir)
    
    print(f"🎵 Analyzing song: {song_id}")
    print("=" * 60)
    
    # Load data
    base_path = data_dir / f"{song_id}.npz"
    mert_path = data_dir / f"{song_id}_mert.npz"
    
    if not base_path.exists():
        print(f"❌ Error: {base_path} not found")
        return
    if not mert_path.exists():
        print(f"❌ Error: {mert_path} not found")
        return
    
    base_data = np.load(base_path, allow_pickle=True)
    mert_data = np.load(mert_path, allow_pickle=True)
    
    # Extract data
    chord_labels = base_data["frame_labels"]
    base_times = base_data["times"]
    embeddings = mert_data["emb"]
    emb_times = mert_data["times"]
    
    print(f"\n📊 Data Info:")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Embedding dim: {embeddings.shape[1]}")
    print(f"  - Number of frames: {len(emb_times)}")
    print(f"  - Duration: {emb_times[-1]:.2f} seconds")
    print(f"  - Unique chords: {len(np.unique(chord_labels))}")
    
    # Align chord labels to embedding times
    print(f"\n⚙️  Aligning chord labels to embedding times...")
    aligned_labels = []
    for t in emb_times:
        idx = np.argmin(np.abs(base_times - t))
        aligned_labels.append(chord_labels[idx])
    aligned_labels = np.array(aligned_labels)
    
    # Compute SSMs
    print(f"⚙️  Computing self-similarity matrices...")
    emb_ssm = compute_embedding_ssm(embeddings, metric="cosine")
    chord_ssm = compute_chord_ssm(aligned_labels)
    
    print(f"  - SSM shape: {emb_ssm.shape}")
    
    # Compute metrics
    print(f"\n⚙️  Computing metrics...")
    alignment_metrics = compute_ssm_alignment_metrics(emb_ssm, chord_ssm)
    emb_structure = compute_structure_metrics(emb_ssm)
    chord_structure = compute_structure_metrics(chord_ssm)
    
    # Display results
    print(f"\n📈 Alignment Metrics (Embedding vs Chord SSM):")
    print(f"  - Pearson Correlation: {alignment_metrics['pearson_r']:.3f} (p={alignment_metrics['pearson_p']:.2e})")
    print(f"  - Spearman Correlation: {alignment_metrics['spearman_r']:.3f}")
    print(f"  - F1 Score: {alignment_metrics['f1']:.3f}")
    print(f"  - Accuracy: {alignment_metrics['accuracy']:.3f}")
    print(f"  - Precision: {alignment_metrics['precision']:.3f}")
    print(f"  - Recall: {alignment_metrics['recall']:.3f}")
    
    print(f"\n🎼 Embedding Structure:")
    print(f"  - Local Homogeneity: {emb_structure['local_homogeneity']:.3f}")
    print(f"  - Repetition Score: {emb_structure['repetition_score']:.3f}")
    print(f"  - Contrast: {emb_structure['contrast']:.3f}")
    print(f"  - Boundary Clarity: {emb_structure['boundary_clarity']:.3f}")
    
    print(f"\n🎹 Chord Structure:")
    print(f"  - Local Homogeneity: {chord_structure['local_homogeneity']:.3f}")
    print(f"  - Repetition Score: {chord_structure['repetition_score']:.3f}")
    print(f"  - Contrast: {chord_structure['contrast']:.3f}")
    print(f"  - Boundary Clarity: {chord_structure['boundary_clarity']:.3f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if alignment_metrics['pearson_r'] > 0.5:
        print(f"  ✓ Strong alignment with chord structure! This layer captures harmonic patterns well.")
    elif alignment_metrics['pearson_r'] > 0.3:
        print(f"  ~ Moderate alignment. Some harmonic information is captured.")
    else:
        print(f"  ✗ Weak alignment. This layer may not encode chord structure strongly.")
    
    if alignment_metrics['f1'] > 0.7:
        print(f"  ✓ Good chord boundary detection! Could work for segmentation.")
    elif alignment_metrics['f1'] > 0.5:
        print(f"  ~ Moderate boundary detection. May need refinement.")
    else:
        print(f"  ✗ Poor boundary detection. Embeddings don't clearly separate chords.")
    
    if emb_structure['repetition_score'] > 0.5:
        print(f"  ✓ Captures structural repetition (verse/chorus patterns).")
    else:
        print(f"  ~ Low repetition score. May not capture song form strongly.")
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💾 To visualize this song's SSMs, run:")
    print(f"   python scripts/analysis/visualize_ssm.py --mode single --song_id {song_id}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test SSM analysis on a single song")
    parser.add_argument("song_id", help="Song ID to analyze")
    parser.add_argument("--data_dir", default="data/processed", help="Data directory")
    args = parser.parse_args()
    
    test_song(args.song_id, args.data_dir)

