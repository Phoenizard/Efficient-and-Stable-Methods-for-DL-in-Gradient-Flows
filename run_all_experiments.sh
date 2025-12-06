#!/bin/bash

# Run all experiments for the SAV-based methods paper
# This script will:
# 1. Generate all experimental data
# 2. Run all experiments
# 3. Save results and plots

echo "=========================================="
echo "Running All Experiments"
echo "=========================================="

# Step 1: Generate data
echo ""
echo "Step 1: Generating experimental data..."
echo "=========================================="
python generate_data.py

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed!"
    exit 1
fi

# Step 2: Run experiments
echo ""
echo "Step 2: Running experiments..."
echo "=========================================="

# Experiment 1: Sin + Cos Regression
echo ""
echo "Running Experiment 1: Sin + Cos Regression..."
python experiment_regression_1.py

if [ $? -ne 0 ]; then
    echo "Warning: Experiment 1 failed!"
fi

# Experiment 2: Quadratic Regression
echo ""
echo "Running Experiment 2: Quadratic Regression..."
python experiment_regression_2.py

if [ $? -ne 0 ]; then
    echo "Warning: Experiment 2 failed!"
fi

# Experiment 3: Gaussian Regression
echo ""
echo "Running Experiment 3: Gaussian Regression..."
python experiment_regression_3.py

if [ $? -ne 0 ]; then
    echo "Warning: Experiment 3 failed!"
fi

# Experiment 4: MNIST Classification
echo ""
echo "Running Experiment 4: MNIST Classification..."
python experiment_classification_mnist.py

if [ $? -ne 0 ]; then
    echo "Warning: Experiment 4 failed (MNIST data may not be available)"
fi

# Summary
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - results/experiment_1/"
echo "  - results/experiment_2/"
echo "  - results/experiment_3/"
echo "  - results/experiment_mnist/"
echo ""
echo "See USAGE_GUIDE.md for more details."
