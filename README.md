# MERT Layer Analysis for Chord Recognition

Analyzing MERT (Music Understanding Model) embeddings using Self-Similarity Matrices (SSMs) to understand which layers best capture harmonic structure and chord progressions.

## 🎯 Project Goal

Determine if MERT embeddings can succeed in automatic chord recognition by:
1. Computing Self-Similarity Matrices from embeddings at different layers
2. Comparing embedding SSMs to chord-label SSMs
3. Identifying which layers best encode harmonic structure
4. Visualizing results in a Looker Studio dashboard via BigQuery

## 📁 Project Structure

```
mert_analysis/
├── configs/
│   └── data.yaml              # Data paths and processing config
├── src/
│   ├── io/
│   │   ├── dataset.py         # Dataset utilities
│   │   └── labs.py            # Chord label parsing
│   ├── mert/
│   │   └── featurize.py       # MERT embedding extraction
│   └── analysis/
│       └── ssm.py             # SSM computation and metrics
├── scripts/
│   ├── dataset/
│   │   ├── cache_mert_layers.py    # Extract MERT embeddings by layer
│   │   └── align_labels.py         # Align chord labels to frames
│   ├── analysis/
│   │   ├── compute_ssm_analysis.py # Single-layer SSM analysis
│   │   ├── compare_mert_layers.py  # Multi-layer comparison
│   │   ├── visualize_ssm.py        # Local visualization
│   │   └── README.md               # Analysis pipeline docs
│   └── export/
│       └── export_to_bigquery.py   # BigQuery export for dashboard
├── data/                      # Symlink to external drive
│   ├── raw/                   # Audio files and chord labels
│   └── processed/             # Cached features and embeddings
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/mert_analysis.git
cd mert_analysis

# Install dependencies
source .venv/bin/activate  # or: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Setup

Your data is on an external drive. Create a symlink:

```bash
# External drive should be mounted at /Volumes/SP PHD U3/
ln -s "/Volumes/SP PHD U3/auto_chord_data/data" data
```

Data structure:
```
data/
├── raw/
│   ├── 01/
│   │   ├── 01 - Song.mp3
│   │   └── 01 Song.lab
│   ├── 02/
│   └── ...
└── processed/
    ├── manifest.csv          # Song metadata
    ├── 1.npz                 # Features + chord labels
    ├── 1_mert.npz            # MERT embeddings
    └── mert_layers/          # Layer-wise embeddings
        ├── L-1/              # Last layer
        ├── L12/              # Layer 12
        └── L18/              # Layer 18
```

### 3. Extract MERT Embeddings (if not cached)

```bash
# Extract embeddings for specific layers
python scripts/dataset/cache_mert_layers.py \
    --manifest data/processed/manifest.csv \
    --layers -1 12 18 \
    --out_root data/processed/mert_layers
```

### 4. Run Analysis

```bash
# Compare all MERT layers
python scripts/analysis/compare_mert_layers.py \
    --mert_layers_dir data/processed/mert_layers \
    --output_dir data/analysis/layer_comparison

# Visualize results locally
python scripts/analysis/visualize_ssm.py \
    --mode summary \
    --results_csv data/analysis/layer_comparison/layer_comparison_results.csv

# Visualize specific song SSMs
python scripts/analysis/visualize_ssm.py \
    --mode single \
    --song_id 100 \
    --data_dir data/processed
```

### 5. Export to BigQuery & Create Dashboard

```bash
# Export to BigQuery
python scripts/export/export_to_bigquery.py \
    --analysis_csv data/analysis/layer_comparison/layer_comparison_results.csv \
    --project_id YOUR_GCP_PROJECT_ID \
    --dataset_id mert_analysis \
    --credentials path/to/service-account-key.json

# Then create Looker Studio dashboard using the BigQuery table
```

## 📊 Key Metrics

### Alignment Metrics (How well embeddings match chord structure)

- **Pearson Correlation**: Linear correlation between embedding SSM and chord SSM
  - Range: [-1, 1]
  - Higher = better alignment with harmonic structure
  
- **F1 Score**: Binary classification of "same chord" vs "different chord"
  - Range: [0, 1]
  - Higher = better chord boundary detection

### Structural Metrics (What patterns embeddings capture)

- **Repetition Score**: Average off-diagonal similarity
  - Captures verse-chorus repetition patterns
  
- **Local Homogeneity**: Average similarity within nearby frames
  - Measures within-chord stability
  
- **Boundary Clarity**: Variance in similarity patterns
  - Detects clear structural boundaries

## 🎵 Analysis Questions

1. **Which MERT layer best captures chord structure?**
   - Look at Pearson correlation by layer
   - Higher correlation = better harmonic encoding

2. **Can MERT predict chord changes?**
   - Look at F1 score for chord boundary detection
   - High F1 + high Pearson = good for chord recognition

3. **Does MERT capture song structure (verse/chorus)?**
   - Look at repetition score
   - High score = detects repeated sections

4. **Are embeddings stable within chords?**
   - Look at local homogeneity
   - High homogeneity + low boundary clarity = may miss chord changes

## 📈 Expected Results

Based on music understanding research, we expect:
- **Middle layers (L12-L18)** to perform best for harmonic structure
- **Earlier layers** to focus on low-level audio features
- **Later layers** to capture high-level song structure
- **Trade-off** between homogeneity and boundary detection

## 🔧 Dependencies

Key packages:
- `torch`, `torchaudio`: Deep learning and audio processing
- `transformers`: MERT model loading
- `librosa`: Audio feature extraction
- `scipy`, `scikit-learn`: SSM computation and metrics
- `google-cloud-bigquery`: Data export
- `matplotlib`, `seaborn`: Visualization

## 📝 Citation

MERT model:
```bibtex
@inproceedings{li2023mert,
  title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author={Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Lin, Chenghua and Ragni, Anton and Benetos, Emmanouil and Gyenge, Norbert and others},
  booktitle={ICASSP},
  year={2024}
}
```

## 🤝 Contributing

This is a research project. Feel free to:
- Experiment with different MERT layers
- Try alternative similarity metrics
- Analyze different music datasets
- Extend to other MIR tasks

## 📧 Contact

For questions or collaboration: [Your contact info]
