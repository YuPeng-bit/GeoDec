#!/bin/bash

# 定义输出目录
OUT_DIR="benchmark_results_stage2_scan"

# 1. 跑 Baseline (Control Scale = 0.0)
# 这相当于纯 MV-Adapter，但走了同样的 Stage 2 Pipeline，作为最严格的基准
echo "Running Scale 0.0..."
python evaluation/run_benchmark_stage2.py --control_scale 0.0 --output_dir $OUT_DIR

echo "Running Scale 0.1..."
python evaluation/run_benchmark_stage2.py --control_scale 0.1 --output_dir $OUT_DIR

echo "Running Scale 0.2..."
python evaluation/run_benchmark_stage2.py --control_scale 0.2 --output_dir $OUT_DIR

echo "Running Scale 0.3..."
python evaluation/run_benchmark_stage2.py --control_scale 0.3 --output_dir $OUT_DIR
#echo "Running Scale 0.4..."
#python evaluation/run_benchmark_stage2.py --control_scale 0.4 --output_dir $OUT_DIR
# 2. 跑中间值
echo "Running Scale 0.5..."
python evaluation/run_benchmark_stage2.py --control_scale 0.5 --output_dir $OUT_DIR

echo "Running Scale 0.6..."
python evaluation/run_benchmark_stage2.py --control_scale 0.6 --output_dir $OUT_DIR

# 3. 跑强约束
echo "Running Scale 0.7..."
python evaluation/run_benchmark_stage2.py --control_scale 0.7 --output_dir $OUT_DIR

echo "Running Scale 0.8..."
python evaluation/run_benchmark_stage2.py --control_scale 0.8 --output_dir $OUT_DIR

echo "Running Scale 0.9..."
python evaluation/run_benchmark_stage2.py --control_scale 0.9 --output_dir $OUT_DIR

echo "Running Scale 1.0..."
python evaluation/run_benchmark_stage2.py --control_scale 1.0 --output_dir $OUT_DIR