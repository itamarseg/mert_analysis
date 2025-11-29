#!/bin/bash
# Chord-level analysis pipeline for MERT embeddings
# Exports manageable chord-to-chord data to BigQuery for Looker Studio

set -e

echo "🎵 MERT Chord-Level Analysis Pipeline"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Configuration
MERT_LAYERS_DIR="${1:-data/processed/mert_layers}"
DATA_DIR="${2:-data/processed}"
GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_CREDENTIALS="${GCP_CREDENTIALS:-}"

echo "📁 Configuration:"
echo "  - MERT layers: $MERT_LAYERS_DIR"
echo "  - Data directory: $DATA_DIR"
echo ""

# Step 1: Test on one song (optional, for verification)
echo "🧪 Step 1: Testing chord aggregation on song 100..."
python scripts/analysis/test_chord_aggregation.py 100 --data_dir "$DATA_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Test successful!"
else
    echo "❌ Test failed!"
    exit 1
fi
echo ""

# Step 2: Export to BigQuery (if configured)
if [ -n "$GCP_PROJECT_ID" ]; then
    echo "☁️  Step 2: Exporting chord-level data to BigQuery..."
    
    if [ -n "$GCP_CREDENTIALS" ]; then
        python scripts/export/export_chord_level_to_bigquery.py \
            --mert_layers_dir "$MERT_LAYERS_DIR" \
            --data_dir "$DATA_DIR" \
            --project_id "$GCP_PROJECT_ID" \
            --credentials "$GCP_CREDENTIALS"
    else
        python scripts/export/export_chord_level_to_bigquery.py \
            --mert_layers_dir "$MERT_LAYERS_DIR" \
            --data_dir "$DATA_DIR" \
            --project_id "$GCP_PROJECT_ID"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ BigQuery export complete!"
    else
        echo "❌ BigQuery export failed!"
        exit 1
    fi
else
    echo "⏭️  Step 2: Skipping BigQuery export (GCP_PROJECT_ID not set)"
fi
echo ""

echo "🎉 Pipeline complete!"
echo ""
echo "📊 Next steps:"
if [ -n "$GCP_PROJECT_ID" ]; then
    echo "  1. Go to Looker Studio: https://lookerstudio.google.com/"
    echo "  2. Create new report → Add BigQuery data source"
    echo "  3. Connect to: $GCP_PROJECT_ID.mert_analysis.chord_similarity"
    echo "  4. Create chord SSM heatmap (see LOOKER_STUDIO_GUIDE.md)"
    echo ""
    echo "📈 Tables created:"
    echo "  - chord_segments (individual chord occurrences)"
    echo "  - chord_similarity (chord-to-chord SSM) ← Use this for heatmap!"
    echo "  - chord_statistics (aggregated stats)"
else
    echo "  1. Set up GCP credentials:"
    echo "     export GCP_PROJECT_ID='your-project-id'"
    echo "     export GCP_CREDENTIALS='path/to/key.json'"
    echo "  2. Run this script again to export to BigQuery"
fi
echo ""
echo "📖 See LOOKER_STUDIO_GUIDE.md for dashboard creation"

