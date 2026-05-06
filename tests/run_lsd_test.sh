#!/usr/bin/env bash
# LSD min_length + fuse 순서 테스트
set -e

INPUT="tests/data/test_acicular.png"
BASE_ARGS="--input $INPUT --measure_mode lsd --particle_type acicular \
  --scale_pixels 147 --scale_um 1 \
  --roi_x_min 0 --roi_y_min 0 --roi_x_max 1024 --roi_y_max 768 \
  --no-auto_center_crop \
  --model dummy --model_cfg dummy"

echo "========================================"
echo "CASE 1: min_length=0, no fuse  (모든 컨투어 통과)"
echo "========================================"
conda run -n measure python primary_measure.py $BASE_ARGS \
  --min_length 0 --output_dir tests/out/case1 2>&1 | grep -E "^\[|num_total|====="

echo ""
echo "========================================"
echo "CASE 2: min_length=50, no fuse  (짧은 것 탈락)"
echo "========================================"
conda run -n measure python primary_measure.py $BASE_ARGS \
  --min_length 50 --output_dir tests/out/case2 2>&1 | grep -E "^\[|num_total|====="

echo ""
echo "========================================"
echo "CASE 3: min_length=50, --fuse  (융합 후 min_length → B 통과해야 함)"
echo "========================================"
conda run -n measure python primary_measure.py $BASE_ARGS \
  --min_length 50 --fuse --output_dir tests/out/case3 2>&1 | grep -E "^\[|num_total|====="
