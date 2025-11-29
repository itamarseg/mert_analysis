"""
Chord centroid analysis - compute average embeddings per chord class
and analyze nearest neighbor relationships.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from dataclasses import dataclass


@dataclass
class ChordCentroid:
    """Represents a chord class centroid."""
    chord_label: str
    chord_root: str
    chord_quality: str
    centroid: np.ndarray  # (D,) average embedding
    n_occurrences: int
    total_frames: int
    std_dev: float  # Average std across dimensions


def compute_chord_centroids(chord_segments_df: pd.DataFrame) -> Dict[str, ChordCentroid]:
    """
    Compute centroid embedding for each unique chord.
    
    Args:
        chord_segments_df: DataFrame from aggregate_frames_to_chords with 'embedding_mean' column
    
    Returns:
        Dictionary mapping chord_label to ChordCentroid object
    """
    centroids = {}
    
    # Group by chord_label to get all occurrences of each chord
    for chord_label, group in chord_segments_df.groupby('chord_label'):
        # Stack all mean embeddings for this chord
        embeddings = np.vstack(group['embedding_mean'].values)
        
        # Compute centroid (average across all occurrences)
        centroid = embeddings.mean(axis=0)
        
        # Get chord info from first occurrence
        first = group.iloc[0]
        
        centroids[chord_label] = ChordCentroid(
            chord_label=chord_label,
            chord_root=first['chord_root'],
            chord_quality=first['chord_quality'],
            centroid=centroid,
            n_occurrences=len(group),
            total_frames=group['n_frames'].sum(),
            std_dev=embeddings.std(axis=0).mean()  # How much do occurrences vary?
        )
    
    return centroids


def find_nearest_neighbors(centroids: Dict[str, ChordCentroid], 
                           k: int = 5,
                           metric: str = 'cosine') -> pd.DataFrame:
    """
    Find k nearest neighbors for each chord centroid.
    
    Args:
        centroids: Dictionary of chord centroids
        k: Number of neighbors to find
        metric: 'cosine' or 'euclidean'
    
    Returns:
        DataFrame with columns:
            - chord (query chord)
            - neighbor_rank (1 = closest)
            - neighbor_chord
            - distance/similarity
    """
    chord_labels = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[c].centroid for c in chord_labels])
    
    # Compute pairwise distances/similarities
    if metric == 'cosine':
        # Cosine similarity (higher = more similar)
        sim_matrix = cosine_similarity(centroid_matrix)
        # For each row, sort by similarity (descending)
        is_similarity = True
    else:
        # Euclidean distance (lower = more similar)
        sim_matrix = euclidean_distances(centroid_matrix)
        is_similarity = False
    
    # Find neighbors
    records = []
    for i, query_chord in enumerate(chord_labels):
        # Get similarities/distances for this chord
        scores = sim_matrix[i]
        
        # Sort by score (descending for similarity, ascending for distance)
        if is_similarity:
            sorted_indices = np.argsort(scores)[::-1]
        else:
            sorted_indices = np.argsort(scores)
        
        # Skip self (rank 0) and get top k neighbors
        for rank, idx in enumerate(sorted_indices[1:k+1], start=1):
            neighbor_chord = chord_labels[idx]
            score = scores[idx]
            
            records.append({
                "query_chord": query_chord,
                "query_root": centroids[query_chord].chord_root,
                "query_quality": centroids[query_chord].chord_quality,
                "neighbor_rank": rank,
                "neighbor_chord": neighbor_chord,
                "neighbor_root": centroids[neighbor_chord].chord_root,
                "neighbor_quality": centroids[neighbor_chord].chord_quality,
                "similarity": float(score) if is_similarity else float(1.0 / (1.0 + score)),
                "distance": float(score) if not is_similarity else float(1.0 - score),
            })
    
    return pd.DataFrame(records)


# Music theory relationships
CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
RELATIVE_MINORS = {
    'C': 'A', 'G': 'E', 'D': 'B', 'A': 'F#', 'E': 'C#', 'B': 'G#',
    'F#': 'D#', 'Db': 'Bb', 'Ab': 'F', 'Eb': 'C', 'Bb': 'G', 'F': 'D'
}


def check_music_theory_alignment(neighbors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if nearest neighbors align with music theory relationships.
    
    Args:
        neighbors_df: Output from find_nearest_neighbors
    
    Returns:
        DataFrame with music theory relationship flags added
    """
    def is_circle_of_fifths_neighbor(root1: str, root2: str) -> bool:
        """Check if two roots are adjacent in circle of fifths."""
        try:
            idx1 = CIRCLE_OF_FIFTHS.index(root1)
            idx2 = CIRCLE_OF_FIFTHS.index(root2)
            # Check if adjacent (wrapping around)
            return abs(idx1 - idx2) == 1 or abs(idx1 - idx2) == 11
        except ValueError:
            return False
    
    def is_relative_major_minor(query_label: str, neighbor_label: str,
                                 query_root: str, neighbor_root: str,
                                 query_quality: str, neighbor_quality: str) -> bool:
        """Check if chords are relative major/minor."""
        # One must be major, one must be minor
        is_maj_min = (('maj' in query_quality and 'min' in neighbor_quality) or
                      ('min' in query_quality and 'maj' in neighbor_quality))
        
        if not is_maj_min:
            return False
        
        # Check if roots match the relative relationship
        if 'maj' in query_quality:
            # Query is major, check if neighbor is its relative minor
            return RELATIVE_MINORS.get(query_root) == neighbor_root
        else:
            # Query is minor, check if neighbor is its relative major
            return RELATIVE_MINORS.get(neighbor_root) == query_root
    
    # Add relationship flags
    neighbors_df['is_same_root'] = (neighbors_df['query_root'] == neighbors_df['neighbor_root'])
    neighbors_df['is_same_quality'] = (neighbors_df['query_quality'] == neighbors_df['neighbor_quality'])
    
    neighbors_df['is_circle_of_fifths'] = neighbors_df.apply(
        lambda row: is_circle_of_fifths_neighbor(row['query_root'], row['neighbor_root']),
        axis=1
    )
    
    neighbors_df['is_relative_major_minor'] = neighbors_df.apply(
        lambda row: is_relative_major_minor(
            row['query_chord'], row['neighbor_chord'],
            row['query_root'], row['neighbor_root'],
            row['query_quality'], row['neighbor_quality']
        ),
        axis=1
    )
    
    return neighbors_df


def analyze_neighbor_patterns(neighbors_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate statistics about neighbor patterns.
    
    Returns:
        Dictionary of metrics about music theory alignment
    """
    # Only look at top-3 neighbors (most relevant)
    top3 = neighbors_df[neighbors_df['neighbor_rank'] <= 3]
    
    metrics = {
        # What % of top-3 neighbors have same root?
        "pct_same_root_in_top3": top3['is_same_root'].mean(),
        
        # What % of top-3 neighbors have same quality?
        "pct_same_quality_in_top3": top3['is_same_quality'].mean(),
        
        # What % of top-3 neighbors are circle-of-fifths related?
        "pct_circle_of_fifths_in_top3": top3['is_circle_of_fifths'].mean(),
        
        # What % of top-3 neighbors are relative major/minor?
        "pct_relative_major_minor_in_top3": top3['is_relative_major_minor'].mean(),
        
        # Average similarity to nearest neighbor
        "avg_nearest_neighbor_similarity": neighbors_df[neighbors_df['neighbor_rank'] == 1]['similarity'].mean(),
        
        # Average similarity to rank-5 neighbor
        "avg_rank5_similarity": neighbors_df[neighbors_df['neighbor_rank'] == 5]['similarity'].mean(),
    }
    
    return metrics


def compute_all_pairs_centroid_similarity(centroids: Dict[str, ChordCentroid]) -> pd.DataFrame:
    """
    Compute ALL pairwise similarities between chord centroids (for heatmap).
    Unlike find_nearest_neighbors which only returns top-k, this returns all pairs.
    
    Returns:
        DataFrame with all chord-to-chord centroid similarities
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    chord_labels = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[c].centroid for c in chord_labels])
    
    # Compute all-to-all similarity
    sim_matrix = cosine_similarity(centroid_matrix)
    
    # Create records for all pairs (upper triangular to avoid duplicates)
    records = []
    for i, chord_i in enumerate(chord_labels):
        for j, chord_j in enumerate(chord_labels):
            records.append({
                "chord_i": chord_i,
                "chord_i_root": centroids[chord_i].chord_root,
                "chord_i_quality": centroids[chord_i].chord_quality,
                "chord_j": chord_j,
                "chord_j_root": centroids[chord_j].chord_root,
                "chord_j_quality": centroids[chord_j].chord_quality,
                "centroid_similarity": float(sim_matrix[i, j]),
                "is_diagonal": int(i == j),
            })
    
    return pd.DataFrame(records)


def create_chord_space_summary(centroids: Dict[str, ChordCentroid],
                               neighbors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of chord centroids with their properties.
    
    Returns:
        DataFrame with one row per chord:
            - chord_label, root, quality
            - n_occurrences, total_frames
            - centroid_std (variability)
            - nearest_neighbor, nn_similarity
            - music_theory_alignment_score
    """
    records = []
    
    for chord_label, centroid in centroids.items():
        # Get top neighbor for this chord
        neighbors = neighbors_df[
            (neighbors_df['query_chord'] == chord_label) & 
            (neighbors_df['neighbor_rank'] == 1)
        ]
        
        if len(neighbors) > 0:
            nn = neighbors.iloc[0]
            nearest_neighbor = nn['neighbor_chord']
            nn_similarity = nn['similarity']
            is_theory_related = (nn['is_circle_of_fifths'] or nn['is_relative_major_minor'])
        else:
            nearest_neighbor = None
            nn_similarity = None
            is_theory_related = False
        
        records.append({
            "chord_label": chord_label,
            "chord_root": centroid.chord_root,
            "chord_quality": centroid.chord_quality,
            "n_occurrences": centroid.n_occurrences,
            "total_frames": centroid.total_frames,
            "centroid_stability": centroid.std_dev,  # Lower = more stable
            "nearest_neighbor": nearest_neighbor,
            "nn_similarity": nn_similarity,
            "nn_is_theory_related": is_theory_related,
        })
    
    return pd.DataFrame(records)

