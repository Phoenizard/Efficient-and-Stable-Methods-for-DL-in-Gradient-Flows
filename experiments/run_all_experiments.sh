#!/bin/bash
# Comprehensive experiment script for all optimization methods
# This script runs experiments for SAV, ExpSAV, IEQ, SGD, and Adam on both regression and classification tasks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Starting All Experiments"
echo "=========================================="

# Create results directory structure
echo "Creating results directories..."
mkdir -p results/SAV
mkdir -p results/ESAV
mkdir -p results/IEQ
mkdir -p results/SGD
mkdir -p results/Adam

# Check if MNIST data exists, if not generate it
if [ ! -f "data/MNIST_train_data.pt" ]; then
    echo -e "${YELLOW}Generating MNIST data...${NC}"
    (cd data/MNIST && python MNIST.py)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}MNIST data generated successfully${NC}"
    else
        echo -e "${RED}Failed to generate MNIST data${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}MNIST data already exists${NC}"
fi

# ==================== Regression Experiments ====================
echo ""
echo -e "${GREEN}=========================================="
echo "Running Regression Experiments"
echo "==========================================${NC}"

# SAV Regression
echo -e "${YELLOW}Running SAV Regression...${NC}"
if python SAV_Regression.py > results/SAV/regression_log.txt 2>&1; then
    echo -e "${GREEN}✓ SAV Regression completed${NC}"
else
    echo -e "${RED}✗ SAV Regression failed${NC}"
fi

# ExpSAV Regression
echo -e "${YELLOW}Running ExpSAV Regression...${NC}"
if python ESAV_Regression.py > results/ESAV/regression_log.txt 2>&1; then
    echo -e "${GREEN}✓ ExpSAV Regression completed${NC}"
else
    echo -e "${RED}✗ ExpSAV Regression failed${NC}"
fi

# IEQ Regression (Full Jacobian)
echo -e "${YELLOW}Running IEQ Regression (Full Jacobian)...${NC}"
if python IEQ_Regression.py > results/IEQ/regression_full_log.txt 2>&1; then
    echo -e "${GREEN}✓ IEQ Regression (Full) completed${NC}"
else
    echo -e "${RED}✗ IEQ Regression (Full) failed${NC}"
fi

# IEQ Regression (Adaptive)
echo -e "${YELLOW}Running IEQ Regression (Adaptive)...${NC}"
if python IEQ_Regression_Adaptive.py > results/IEQ/regression_adaptive_log.txt 2>&1; then
    echo -e "${GREEN}✓ IEQ Regression (Adaptive) completed${NC}"
else
    echo -e "${RED}✗ IEQ Regression (Adaptive) failed${NC}"
fi

# SGD Regression
echo -e "${YELLOW}Running SGD Regression...${NC}"
if python SGD_Regression.py > results/SGD/regression_log.txt 2>&1; then
    echo -e "${GREEN}✓ SGD Regression completed${NC}"
else
    echo -e "${RED}✗ SGD Regression failed${NC}"
fi

# Adam Regression (if exists)
if [ -f "Adam_Regression.py" ]; then
    echo -e "${YELLOW}Running Adam Regression...${NC}"
    python Adam_Regression.py > results/Adam/regression_log.txt 2>&1
    echo -e "${GREEN}✓ Adam Regression completed${NC}"
fi

# ==================== Classification Experiments ====================
echo ""
echo -e "${GREEN}=========================================="
echo "Running Classification Experiments"
echo "==========================================${NC}"

# SAV Classification
echo -e "${YELLOW}Running SAV Classification...${NC}"
if python SAV_Classification.py > results/SAV/classification_log.txt 2>&1; then
    echo -e "${GREEN}✓ SAV Classification completed${NC}"
else
    echo -e "${RED}✗ SAV Classification failed${NC}"
fi

# ExpSAV Classification
echo -e "${YELLOW}Running ExpSAV Classification...${NC}"
if python ESAV_Classification.py > results/ESAV/classification_log.txt 2>&1; then
    echo -e "${GREEN}✓ ExpSAV Classification completed${NC}"
else
    echo -e "${RED}✗ ExpSAV Classification failed${NC}"
fi

# IEQ Classification (Full Jacobian)
echo -e "${YELLOW}Running IEQ Classification (Full Jacobian)...${NC}"
if python IEQ_Classification.py > results/IEQ/classification_full_log.txt 2>&1; then
    echo -e "${GREEN}✓ IEQ Classification (Full) completed${NC}"
else
    echo -e "${RED}✗ IEQ Classification (Full) failed${NC}"
fi

# IEQ Classification (Adaptive)
echo -e "${YELLOW}Running IEQ Classification (Adaptive)...${NC}"
if python IEQ_Classification_Adaptive.py > results/IEQ/classification_adaptive_log.txt 2>&1; then
    echo -e "${GREEN}✓ IEQ Classification (Adaptive) completed${NC}"
else
    echo -e "${RED}✗ IEQ Classification (Adaptive) failed${NC}"
fi

# SGD Classification
echo -e "${YELLOW}Running SGD Classification...${NC}"
if python SGD_Classification.py > results/SGD/classification_log.txt 2>&1; then
    echo -e "${GREEN}✓ SGD Classification completed${NC}"
else
    echo -e "${RED}✗ SGD Classification failed${NC}"
fi

# Adam Classification (if exists)
if [ -f "Adam.py" ]; then
    echo -e "${YELLOW}Running Adam Classification...${NC}"
    python Adam.py > results/Adam/classification_log.txt 2>&1
    echo -e "${GREEN}✓ Adam Classification completed${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "All Experiments Completed Successfully!"
echo "Results saved in ./results/ directory"
echo "==========================================${NC}"

# Generate summary
echo ""
echo "Generating results summary..."
python experiments/summarize_results.py
