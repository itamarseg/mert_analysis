#!/usr/bin/env python3
"""
Export chord-level analysis (not frame-level) to BigQuery.
Much more manageable for Looker Studio - enables chord quality filtering and SSM visualization.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from google.cloud import bigquery
from google.oauth2 import service_account
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.chord_aggregation import (
    aggregate_frames_to_chords,
    compute_chord_to_chord_similarity,
    compute_chord_statistics,
    parse_chord_quality
)


def create_chord_segments_schema():
    """Schema for chord segments table (one row per chord segment)."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_idx", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("chord_label", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_bass", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("start_time", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("end_time", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("duration", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("n_frames", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("embedding_stability", "FLOAT", mode="NULLABLE"),
    ]


def create_chord_similarity_schema():
    """Schema for chord-to-chord similarity table."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        
        # Chord i
        bigquery.SchemaField("chord_i_idx", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("chord_i_label", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_i_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_i_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_i_duration", "FLOAT", mode="REQUIRED"),
        
        # Chord j
        bigquery.SchemaField("chord_j_idx", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("chord_j_label", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_j_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_j_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_j_duration", "FLOAT", mode="REQUIRED"),
        
        # Similarity & relationships
        bigquery.SchemaField("cosine_similarity", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("is_transition", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("is_diagonal", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("same_root", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("same_quality", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("same_chord", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("time_distance", "FLOAT", mode="REQUIRED"),
    ]


def create_chord_statistics_schema():
    """Schema for per-chord statistics table."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_label", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("occurrences", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("total_duration", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("avg_duration", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("std_duration", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("avg_embedding_stability", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("total_frames", "INTEGER", mode="REQUIRED"),
    ]


def process_song_layer(song_id: str, layer_dir: Path, data_dir: Path) -> Dict:
    """Process one song/layer combination."""
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
    
    # Align labels to embedding times
    aligned_labels = []
    for t in emb_times:
        idx = np.argmin(np.abs(base_times - t))
        aligned_labels.append(chord_labels[idx])
    aligned_labels = np.array(aligned_labels)
    
    # Aggregate to chords
    chord_segments = aggregate_frames_to_chords(embeddings, aligned_labels, emb_times)
    chord_segments["song_id"] = song_id
    chord_segments["layer_name"] = layer_name
    
    # Compute chord-to-chord similarity (before renaming)
    chord_similarity = compute_chord_to_chord_similarity(chord_segments)
    chord_similarity["song_id"] = song_id
    chord_similarity["layer_name"] = layer_name
    
    # Compute chord statistics (before renaming)
    chord_stats = compute_chord_statistics(chord_segments)
    chord_stats["song_id"] = song_id
    chord_stats["layer_name"] = layer_name
    
    # Now rename for export to BigQuery
    chord_segments.rename(columns={"embedding_std": "embedding_stability"}, inplace=True)
    
    return {
        "segments": chord_segments,
        "similarity": chord_similarity,
        "statistics": chord_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Export chord-level analysis to BigQuery")
    parser.add_argument("--mert_layers_dir", default="data/processed/mert_layers",
                       help="Directory with layer-wise MERT embeddings")
    parser.add_argument("--data_dir", default="data/processed",
                       help="Directory with chord labels")
    parser.add_argument("--project_id", required=True,
                       help="Google Cloud project ID")
    parser.add_argument("--dataset_id", default="mert_analysis",
                       help="BigQuery dataset ID")
    parser.add_argument("--credentials", default=None,
                       help="Path to service account JSON")
    parser.add_argument("--song_ids", nargs="*", default=None,
                       help="Specific songs to process (default: all)")
    args = parser.parse_args()
    
    mert_layers_dir = Path(args.mert_layers_dir)
    data_dir = Path(args.data_dir)
    
    # Initialize BigQuery client
    if args.credentials:
        credentials = service_account.Credentials.from_service_account_file(args.credentials)
        client = bigquery.Client(credentials=credentials, project=args.project_id)
    else:
        client = bigquery.Client(project=args.project_id)
    
    # Create dataset if needed
    dataset_ref = f"{args.project_id}.{args.dataset_id}"
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_ref}")
    
    # Find layers and songs
    layer_dirs = sorted([d for d in mert_layers_dir.iterdir() if d.is_dir() and d.name.startswith("L")])
    
    if args.song_ids:
        song_ids = args.song_ids
    else:
        song_files = sorted(layer_dirs[0].glob("*.npz"))
        song_ids = [f.stem for f in song_files]
    
    print(f"Processing {len(song_ids)} songs × {len(layer_dirs)} layers...")
    
    # Collect all data
    all_segments = []
    all_similarity = []
    all_statistics = []
    
    for song_id in tqdm(song_ids, desc="Songs"):
        for layer_dir in layer_dirs:
            try:
                result = process_song_layer(song_id, layer_dir, data_dir)
                if result is not None:
                    all_segments.append(result["segments"])
                    all_similarity.append(result["similarity"])
                    all_statistics.append(result["statistics"])
            except Exception as e:
                print(f"Error processing {song_id}/{layer_dir.name}: {e}")
                continue
    
    # Combine and export
    print("\n📤 Uploading to BigQuery...")
    
    # Table 1: Chord segments
    df_segments = pd.concat(all_segments, ignore_index=True)
    df_segments = df_segments[["song_id", "layer_name", "chord_idx", "chord_label", 
                                "chord_root", "chord_quality", "chord_bass",
                                "start_time", "end_time", "duration", "n_frames", 
                                "embedding_stability"]]
    
    job_config = bigquery.LoadJobConfig(
        schema=create_chord_segments_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(
        df_segments, 
        f"{dataset_ref}.chord_segments",
        job_config=job_config
    )
    job.result()
    print(f"✓ Uploaded {len(df_segments)} chord segments")
    
    # Table 2: Chord-to-chord similarity
    df_similarity = pd.concat(all_similarity, ignore_index=True)
    
    job_config = bigquery.LoadJobConfig(
        schema=create_chord_similarity_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(
        df_similarity,
        f"{dataset_ref}.chord_similarity",
        job_config=job_config
    )
    job.result()
    print(f"✓ Uploaded {len(df_similarity)} chord similarity pairs")
    
    # Table 3: Chord statistics
    df_statistics = pd.concat(all_statistics, ignore_index=True)
    
    job_config = bigquery.LoadJobConfig(
        schema=create_chord_statistics_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(
        df_statistics,
        f"{dataset_ref}.chord_statistics",
        job_config=job_config
    )
    job.result()
    print(f"✓ Uploaded {len(df_statistics)} chord statistics")
    
    print("\n✅ Export complete!")
    print(f"\nBigQuery tables created:")
    print(f"  1. {dataset_ref}.chord_segments - Individual chord occurrences")
    print(f"  2. {dataset_ref}.chord_similarity - Chord-to-chord SSM (manageable size!)")
    print(f"  3. {dataset_ref}.chord_statistics - Per-chord aggregated stats")
    print(f"\n📊 Ready for Looker Studio!")
    print(f"   - Filter by chord_quality (maj, min, 7, etc.)")
    print(f"   - Create chord SSM heatmap using chord_similarity table")
    print(f"   - Analyze transitions using is_transition=1")


if __name__ == "__main__":
    main()

