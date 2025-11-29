"""
Self-Similarity Matrix (SSM) analysis for MERT embeddings and chord labels.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict


def compute_embedding_ssm(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute self-similarity matrix from embeddings.
    
    Args:
        embeddings: (T, D) array of embeddings
        metric: 'cosine' for cosine similarity, 'euclidean' for euclidean distance
    
    Returns:
        (T, T) similarity matrix, where higher values = more similar
    """
    if metric == "cosine":
        # Cosine similarity: range [-1, 1], 1 = identical
        return cosine_similarity(embeddings)
    elif metric == "euclidean":
        # Convert distance to similarity: lower distance = higher similarity
        dist = squareform(pdist(embeddings, metric='euclidean'))
        # Normalize to [0, 1] range
        return 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_chord_ssm(chord_labels: np.ndarray) -> np.ndarray:
    """
    Compute binary self-similarity matrix from chord labels.
    
    Args:
        chord_labels: (T,) array of chord labels (strings)
    
    Returns:
        (T, T) binary matrix where 1 = same chord, 0 = different chord
    """
    T = len(chord_labels)
    ssm = np.zeros((T, T), dtype=np.float32)
    
    for i in range(T):
        ssm[i, :] = (chord_labels == chord_labels[i]).astype(np.float32)
    
    return ssm


def compute_ssm_alignment_metrics(emb_ssm: np.ndarray, chord_ssm: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics comparing embedding SSM to chord SSM.
    
    Args:
        emb_ssm: (T, T) embedding similarity matrix
        chord_ssm: (T, T) binary chord similarity matrix
    
    Returns:
        Dictionary of metrics
    """
    # Flatten to vectors (upper triangular, excluding diagonal)
    mask = np.triu(np.ones_like(emb_ssm, dtype=bool), k=1)
    emb_flat = emb_ssm[mask]
    chord_flat = chord_ssm[mask]
    
    # Pearson correlation
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(emb_flat, chord_flat)
    
    # Spearman correlation (rank-based)
    spearman_r, spearman_p = spearmanr(emb_flat, chord_flat)
    
    # Binary classification metrics (treat chord SSM as ground truth)
    # Threshold embedding SSM at median to create binary predictions
    threshold = np.median(emb_flat)
    emb_binary = (emb_flat > threshold).astype(int)
    chord_binary = chord_flat.astype(int)
    
    # Accuracy
    accuracy = np.mean(emb_binary == chord_binary)
    
    # Precision, Recall, F1 for "same chord" class
    tp = np.sum((emb_binary == 1) & (chord_binary == 1))
    fp = np.sum((emb_binary == 1) & (chord_binary == 0))
    fn = np.sum((emb_binary == 0) & (chord_binary == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_structure_metrics(ssm: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics that capture structural properties of the SSM.
    
    Args:
        ssm: (T, T) similarity matrix
    
    Returns:
        Dictionary of structural metrics
    """
    # Homogeneity: how self-similar are nearby frames (local structure)
    # Average similarity within a small diagonal band
    band_size = min(50, ssm.shape[0] // 10)  # ~50 frames or 10% of song
    band_mask = np.abs(np.subtract.outer(np.arange(ssm.shape[0]), np.arange(ssm.shape[0]))) <= band_size
    local_homogeneity = ssm[band_mask].mean()
    
    # Repetition: how much non-local similarity exists (off-diagonal structure)
    # Average similarity outside the diagonal band
    off_diag_mask = ~band_mask
    np.fill_diagonal(off_diag_mask, False)  # exclude diagonal
    repetition_score = ssm[off_diag_mask].mean()
    
    # Contrast: difference between high and low similarities
    contrast = ssm.std()
    
    # Block structure: detect if there are clear block patterns
    # Use downsampled SSM to detect larger patterns
    downsample_factor = max(1, ssm.shape[0] // 100)
    if downsample_factor > 1:
        ssm_down = ssm[::downsample_factor, ::downsample_factor]
        # Compute variance of each row (high variance = clear boundaries)
        boundary_clarity = ssm_down.var(axis=1).mean()
    else:
        boundary_clarity = ssm.var(axis=1).mean()
    
    return {
        "local_homogeneity": float(local_homogeneity),
        "repetition_score": float(repetition_score),
        "contrast": float(contrast),
        "boundary_clarity": float(boundary_clarity),
    }


def downsample_ssm(ssm: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample SSM for visualization by averaging blocks.
    
    Args:
        ssm: (T, T) similarity matrix
        factor: downsampling factor
    
    Returns:
        (T//factor, T//factor) downsampled matrix
    """
    T = ssm.shape[0]
    T_new = T // factor
    
    ssm_down = np.zeros((T_new, T_new), dtype=np.float32)
    for i in range(T_new):
        for j in range(T_new):
            block = ssm[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            ssm_down[i, j] = block.mean()
    
    return ssm_down

