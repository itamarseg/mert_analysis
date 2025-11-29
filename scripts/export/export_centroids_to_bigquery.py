#!/usr/bin/env python3
"""
Export chord centroid nearest neighbor analysis to BigQuery.
"""
import argparse
import pandas as pd
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account


def create_chord_neighbors_schema():
    """Schema for chord nearest neighbors table."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("query_chord", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("query_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("query_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("neighbor_rank", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("neighbor_chord", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("neighbor_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("neighbor_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("similarity", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("distance", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("is_same_root", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("is_same_quality", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("is_circle_of_fifths", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("is_relative_major_minor", "BOOLEAN", mode="REQUIRED"),
    ]


def create_centroid_metrics_schema():
    """Schema for per-song/layer centroid metrics."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("n_unique_chords", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("pct_same_root_in_top3", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("pct_same_quality_in_top3", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("pct_circle_of_fifths_in_top3", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("pct_relative_major_minor_in_top3", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("avg_nearest_neighbor_similarity", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("avg_rank5_similarity", "FLOAT", mode="REQUIRED"),
    ]


def create_chord_summaries_schema():
    """Schema for chord summaries table."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_label", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("n_occurrences", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("total_frames", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("centroid_stability", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("nearest_neighbor", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("nn_similarity", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("nn_is_theory_related", "BOOLEAN", mode="NULLABLE"),
    ]


def create_centroid_similarity_schema():
    """Schema for all-pairs centroid similarity (for heatmap)."""
    return [
        bigquery.SchemaField("song_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("layer_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_i", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_i_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_i_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_j", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_j_root", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chord_j_quality", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("centroid_similarity", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("is_diagonal", "INTEGER", mode="REQUIRED"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Export centroid analysis to BigQuery")
    parser.add_argument("--analysis_dir", default="data/analysis/chord_centroids",
                       help="Directory with analysis results")
    parser.add_argument("--project_id", required=True,
                       help="Google Cloud project ID")
    parser.add_argument("--dataset_id", default="mert_analysis",
                       help="BigQuery dataset ID")
    parser.add_argument("--credentials", default=None,
                       help="Path to service account JSON")
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    
    # Check if analysis files exist
    neighbors_path = analysis_dir / "chord_neighbors.csv"
    all_pairs_path = analysis_dir / "chord_centroid_similarity.csv"
    metrics_path = analysis_dir / "centroid_metrics.csv"
    summaries_path = analysis_dir / "chord_summaries.csv"
    
    if not neighbors_path.exists():
        print(f"❌ Error: {neighbors_path} not found")
        print(f"Run: python scripts/analysis/compute_chord_centroids.py")
        return
    
    # Load data
    print(f"📂 Loading analysis results...")
    df_neighbors = pd.read_csv(neighbors_path)
    df_all_pairs = pd.read_csv(all_pairs_path) if all_pairs_path.exists() else None
    df_metrics = pd.read_csv(metrics_path)
    df_summaries = pd.read_csv(summaries_path)
    
    # Convert song_id to string (BigQuery expects STRING type)
    df_neighbors['song_id'] = df_neighbors['song_id'].astype(str)
    df_metrics['song_id'] = df_metrics['song_id'].astype(str)
    df_summaries['song_id'] = df_summaries['song_id'].astype(str)
    if df_all_pairs is not None:
        df_all_pairs['song_id'] = df_all_pairs['song_id'].astype(str)
    
    print(f"  - Neighbors: {len(df_neighbors):,} records")
    if df_all_pairs is not None:
        print(f"  - All-pairs: {len(df_all_pairs):,} records")
    print(f"  - Metrics: {len(df_metrics):,} records")
    print(f"  - Summaries: {len(df_summaries):,} records")
    
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
    
    print(f"\n📤 Uploading to BigQuery...")
    
    # Upload table 1: Neighbors
    job_config = bigquery.LoadJobConfig(
        schema=create_chord_neighbors_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(
        df_neighbors,
        f"{dataset_ref}.chord_neighbors",
        job_config=job_config
    )
    job.result()
    print(f"✓ Uploaded {len(df_neighbors):,} neighbor records")
    
    # Upload table 2: Metrics
    job_config = bigquery.LoadJobConfig(
        schema=create_centroid_metrics_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(
        df_metrics,
        f"{dataset_ref}.centroid_metrics",
        job_config=job_config
    )
    job.result()
    print(f"✓ Uploaded {len(df_metrics):,} metric records")
    
    # Upload table 3: Summaries
    job_config = bigquery.LoadJobConfig(
        schema=create_chord_summaries_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(
        df_summaries,
        f"{dataset_ref}.chord_summaries",
        job_config=job_config
    )
    job.result()
    print(f"✓ Uploaded {len(df_summaries):,} chord summaries")
    
    # Upload table 4: All-pairs centroid similarity (if available)
    if df_all_pairs is not None:
        job_config = bigquery.LoadJobConfig(
            schema=create_centroid_similarity_schema(),
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        job = client.load_table_from_dataframe(
            df_all_pairs,
            f"{dataset_ref}.chord_centroid_similarity",
            job_config=job_config
        )
        job.result()
        print(f"✓ Uploaded {len(df_all_pairs):,} all-pairs centroid similarities")
    
    print(f"\n✅ Export complete!")
    print(f"\nBigQuery tables created:")
    print(f"  1. {dataset_ref}.chord_neighbors (top-5 neighbors)")
    print(f"  2. {dataset_ref}.centroid_metrics (per-song/layer stats)")
    print(f"  3. {dataset_ref}.chord_summaries (per-chord info)")
    if df_all_pairs is not None:
        print(f"  4. {dataset_ref}.chord_centroid_similarity (for heatmap!) ⭐")
    
    print(f"\n📊 Looker Studio Visualizations:")
    print(f"  - Layer comparison: Which layer captures music theory best?")
    print(f"  - Chord space map: Which chords are nearest neighbors?")
    print(f"  - Theory alignment: Circle of 5ths vs relative major/minor")
    print(f"  - Quality clustering: Do majors group with majors?")


if __name__ == "__main__":
    main()

