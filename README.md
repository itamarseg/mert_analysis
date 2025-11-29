# MERT Embedding Analysis for Automatic Chord Recognition

Analysis of MERT (Music Understanding Model) embeddings to evaluate their suitability for automatic chord recognition tasks. This project examines what harmonic and structural information MERT layers capture through Self-Similarity Matrix analysis and chord centroid nearest-neighbor analysis.

## 🎯 Research Questions

1. **Do MERT embeddings capture chord structure?**
   - Are chord boundaries detectable in embedding space?
   - Do same-chord occurrences have similar embeddings?

2. **Which MERT layer best encodes harmonic information?**
   - Layer-by-layer comparison across all 25 transformer layers
   - Trade-offs between consistency and discrimination

3. **Does MERT organize chords according to music theory?**
   - Circle of fifths relationships
   - Relative major-minor pairs
   - Chord quality clustering (major vs minor)

4. **Can MERT succeed at chord recognition without fine-tuning?**
   - Quantitative evaluation of embedding-based chord discrimination

## 📊 Key Findings

### Self-Similarity Matrix Analysis

**Weak chord discrimination across all layers:**
- Pearson correlation with chord-label SSM: **r = 0.069** (very weak)
- F1 score for chord boundary detection: **0.296** (poor)
- All embeddings show 0.75-0.90 similarity regardless of chord (minimal separation)

**Layer comparison:**
- Layer L24: Best separation but still weak (similarity = 0.81 vs 0.93 in other layers)
- No layer shows strong chord-specific clustering
- Embeddings appear "smoothed" - capture general musical features, not fine-grained harmony

### Chord Centroid Nearest Neighbor Analysis

**Modest music theory alignment:**
- **28% of top-3 neighbors** are circle-of-fifths related (Layer L1) ✅
- **9% of top-3 neighbors** are relative major-minor pairs (Layer L19)
- **19% same quality clustering** (majors with majors, minors with minors)

**Critical limitation:**
- Average nearest-neighbor similarity: **0.936-0.988** across all layers
- Weak discrimination: all chords appear highly similar
- Model knows relationships exist but cannot distinguish strongly

### Overall Conclusion

**MERT embeddings show limited suitability for chord recognition:**
- ✅ Captures coarse harmonic relationships (circle of fifths, relative keys)
- ❌ Cannot discriminate between chords effectively (all 0.9+ similar)
- ❌ No layer specialization for harmonic structure
- 💡 **Task-specific fine-tuning required** for practical chord recognition

## 🏗️ Project Structure

```
mert_analysis/
├── configs/
│   └── data.yaml                  # Data paths and audio parameters
│
├── src/
│   ├── io/
│   │   ├── dataset.py            # Dataset utilities
│   │   └── labs.py               # Chord label file parsing
│   │
│   ├── mert/
│   │   └── featurize.py          # MERT embedding extraction
│   │
│   └── analysis/
│       ├── ssm.py                # Self-similarity matrix computation
│       ├── chord_aggregation.py  # Frame-to-chord aggregation
│       └── chord_centroids.py    # Centroid & nearest neighbor analysis
│
├── scripts/
│   ├── dataset/
│   │   ├── cache_mert_layers.py  # Extract embeddings per layer
│   │   └── align_labels.py       # Align chord labels to frames
│   │
│   ├── analysis/
│   │   ├── test_single_song.py           # Quick frame-level test
│   │   ├── test_chord_aggregation.py     # Test chord-level aggregation
│   │   ├── test_chord_centroids.py       # Test centroid analysis
│   │   ├── test_layer_comparison.py      # Compare multiple layers
│   │   └── compute_chord_centroids.py    # Full centroid analysis
│   │
│   ├── export/
│   │   ├── export_chord_level_to_bigquery.py   # Export SSM analysis
│   │   └── export_centroids_to_bigquery.py     # Export centroid analysis
│   │
│   └── run_chord_analysis.sh     # Main pipeline script
│
├── data/  (symlink to external drive)
│   ├── raw/                      # Audio files (.mp3) and chord labels (.lab)
│   └── processed/                # Cached features and MERT embeddings
│
└── requirements.txt              # Python dependencies
```

### Data Structure

Expected data organization:
```
data/
├── raw/
│   ├── 01/
│   │   ├── 01 - Song Title.mp3
│   │   └── 01 Song Title.lab      # Chord annotations
│   ├── 02/
│   └── ...
└── processed/
    ├── manifest.csv                # Song metadata
    ├── {song_id}.npz              # Cached features + chord labels
    ├── {song_id}_mert.npz         # MERT embeddings (single layer)
    └── mert_layers/               # Layer-wise embeddings
        ├── L0/, L1/, ..., L24/
        └── Each contains {song_id}.npz
```

## 🔬 Methodology

### 1. Self-Similarity Matrix Analysis

For each song and layer:
1. Extract MERT embeddings at ~75 Hz (12,000 frames for a 3-minute song)
2. Aggregate frames to chord segments (reduce to ~100 segments)
3. Compute embedding SSM: cosine similarity between all segment pairs
4. Compute chord-label SSM: binary matrix (same chord = 1)
5. Compare SSMs using Pearson correlation and F1 score

### 2. Chord Centroid Analysis

For each song and layer:
1. Compute centroid (mean embedding) for each unique chord
2. Find k-nearest neighbors in centroid space
3. Check if neighbors align with music theory:
   - Circle of fifths: C↔G, G↔D, D↔A, etc.
   - Relative major-minor: C↔Am, G↔Em, etc.
   - Same quality: major chords grouped together
4. Compute alignment percentages

