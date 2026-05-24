#!/bin/bash
set -e

source .venv/bin/activate
export PYTHONPATH=.

DATA_DIR="data/TESS Toronto emotional speech set data"

echo "=== Training Speech Model ==="
python3 models/speech_pipeline/train.py --data_dir "$DATA_DIR" --epochs 10

echo "=== Testing Speech Model ==="
python3 models/speech_pipeline/test.py --data_dir "$DATA_DIR"

echo "=== Training Text Model ==="
python3 models/text_pipeline/train.py --data_dir "$DATA_DIR" --epochs 10

echo "=== Testing Text Model ==="
python3 models/text_pipeline/test.py --data_dir "$DATA_DIR"

echo "=== Training Fusion Model ==="
python3 models/fusion_pipeline/train.py --data_dir "$DATA_DIR" --epochs 10

echo "=== Testing Fusion Model ==="
python3 models/fusion_pipeline/test.py --data_dir "$DATA_DIR"

echo "=== All Done ==="
