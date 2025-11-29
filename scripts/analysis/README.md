# Chord-Level Analysis Scripts

These scripts analyze MERT embeddings at the **chord level** (not frame level) to create manageable datasets for Looker Studio dashboards.

## Why Chord-Level?

**Frame-level problem:**
- Song with 11,807 frames → SSM is 11,807 × 11,807 = 139 million cells ❌
- Too large to visualize or analyze
- Redundant (most frames within a chord are similar)

**Chord-level solution:**
- Same song → ~97 chord segments → SSM is 97 × 97 = 9,409 cells ✅
- 14,816x smaller!
- Meaningful patterns (chord-to-chord relationships)
- Filter by chord quality (major, minor, 7th, etc.)

## Available Scripts

### 1. `test_single_song.py`
Quick test to see frame-level embeddings and metrics for one song.

```bash
python scripts/analysis/test_single_song.py 100
```

**Shows:**
- Frame-level data dimensions
- SSM alignment metrics
- Interpretation (but frame-level)

**Note:** This uses the last layer (L-1) only. Good for debugging.

---

### 2. `test_chord_aggregation.py` ⭐
Test chord-level aggregation on one song.

```bash
python scripts/analysis/test_chord_aggregation.py 100
```

**Shows:**
- Size reduction (frames → chords)
- Chord segments with quality parsing
- Chord-to-chord similarity examples
- Statistics per unique chord

**Output example:**
```
Frame-Level: 11,807 × 11,807 = 139M cells
Chord-Level: 97 × 97 = 9,409 cells
Size reduction: 14,816x smaller!
```

---

### 3. `visualize_ssm.py`
Create local SSM visualizations.

```bash
# Visualize single song SSM (frame-level)
python scripts/analysis/visualize_ssm.py \
    --mode single \
    --song_id 100

# Note: Creates frame-level visualization (large)
# For chord-level, use Looker Studio after BigQuery export
```

**Output:** PNG file in `data/analysis/visualizations/`

---

## Export Pipeline

### Main Script: `export_chord_level_to_bigquery.py`

Located in `scripts/export/export_chord_level_to_bigquery.py`

```bash
python scripts/export/export_chord_level_to_bigquery.py \
    --mert_layers_dir data/processed/mert_layers \
    --project_id YOUR_GCP_PROJECT_ID \
    --credentials path/to/key.json
```

**Creates 3 BigQuery tables:**
1. `chord_segments` - Individual chord occurrences
2. `chord_similarity` - Chord-to-chord SSM (use for heatmap!)
3. `chord_statistics` - Aggregated stats

See `LOOKER_STUDIO_GUIDE.md` for complete dashboard setup.

---

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Test chord aggregation
python scripts/analysis/test_chord_aggregation.py 100

# 3. Export to BigQuery
export GCP_PROJECT_ID="your-project-id"
export GCP_CREDENTIALS="path/to/key.json"
./run_chord_analysis.sh

# 4. Create Looker Studio dashboard
# Follow instructions in LOOKER_STUDIO_GUIDE.md
```

---

## File Structure

```
scripts/analysis/
├── test_single_song.py          # Frame-level quick test
├── test_chord_aggregation.py    # Chord-level test
├── visualize_ssm.py             # Local visualization
└── README.md                    # This file

scripts/export/
└── export_chord_level_to_bigquery.py  # Main export script

src/analysis/
├── ssm.py                       # SSM computation utilities
└── chord_aggregation.py         # Frame → chord aggregation
```

---

## Chord Quality Filtering

Parsed automatically from chord labels:

| Chord Label | Root | Quality | Bass |
|-------------|------|---------|------|
| C | C | maj | None |
| Am | A | min | None |
| G7 | G | 7 | None |
| D/F# | D | maj | F# |
| E:min | E | :min | None |

**In Looker Studio, filter by:**
- `chord_quality = 'maj'` → Only major chords
- `chord_quality = 'min'` → Only minor chords
- `chord_quality = '7'` → Only 7th chords

This enables analysis like:
- "How similar are major chords to each other?"
- "Do major→minor transitions have low similarity?"
- "Which layer best separates major from minor?"

---

## Troubleshooting

**Error: No MERT embeddings found**
```bash
# Run MERT extraction first
python scripts/dataset/cache_mert_layers.py \
    --manifest data/processed/manifest.csv \
    --layers -1 12 18
```

**Error: No chord labels found**
```bash
# Run label alignment first
python scripts/dataset/align_labels.py
```

**BigQuery authentication error**
- Ensure service account has BigQuery Admin role
- Check credentials file path

---

## Next Steps

1. ✅ Test chord aggregation locally
2. ✅ Export to BigQuery
3. ✅ Create Looker Studio dashboard (see LOOKER_STUDIO_GUIDE.md)
4. 📊 Analyze which layers capture chord structure
5. 📝 Export results for paper

**Main insight:** Chord-level analysis is manageable and meaningful for understanding how MERT represents harmonic structure!
