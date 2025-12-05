# Experiments Documentation

This folder contains scripts and configurations for running comprehensive experiments on all optimization methods.

## Contents

- `experiment_config.yaml` - Complete hyperparameter configurations for all methods
- `run_all_experiments.sh` - Shell script to run all experiments automatically
- `summarize_results.py` - Python script to parse and summarize experimental results

## Quick Start

### Running All Experiments

To run all experiments (both regression and classification tasks for all methods):

```bash
# Make the script executable
chmod +x experiments/run_all_experiments.sh

# Run all experiments
./experiments/run_all_experiments.sh
```

This will:
1. Generate MNIST data if not present
2. Run all regression experiments (SAV, ExpSAV, IEQ, SGD, Adam)
3. Run all classification experiments
4. Save logs to `results/` directory
5. Generate summary statistics

### Running Individual Experiments

To run specific experiments:

```bash
# Regression experiments
python SAV_Regression.py
python ESAV_Regression.py
python IEQ_Regression.py
python IEQ_Regression_Adaptive.py
python SGD_Regression.py

# Classification experiments
python SAV_Classification.py
python ESAV_Classification.py
python IEQ_Classification.py
python IEQ_Classification_Adaptive.py
python SGD_Classification.py
python Adam.py
```

## Experiment Configuration

All hyperparameters are documented in `experiment_config.yaml`. Key configurations:

### Regression Task
- **Dataset:** Gaussian data with $y = \exp(-x^2) + \epsilon$
- **Model:** 2-layer network (1 → 100 → 1)
- **Epochs:** 50,000
- **Batch Size:** 256

### Classification Task
- **Dataset:** MNIST (28×28 handwritten digits)
- **Model:** 2-layer network (784 → 100 → 10)
- **Epochs:** 100
- **Batch Size:** 256

## Results Structure

After running experiments, results are organized as:

```
results/
├── SAV/
│   ├── regression_log.txt
│   └── classification_log.txt
├── ESAV/
│   ├── regression_log.txt
│   └── classification_log.txt
├── IEQ/
│   ├── regression_full_log.txt
│   ├── regression_adaptive_log.txt
│   ├── classification_full_log.txt
│   └── classification_adaptive_log.txt
├── SGD/
│   ├── regression_log.txt
│   └── classification_log.txt
├── Adam/
│   ├── regression_log.txt
│   └── classification_log.txt
└── summary.json
```

## Analyzing Results

After experiments complete, view the summary:

```bash
python experiments/summarize_results.py
```

This generates:
- Console output with formatted tables
- `results/summary.json` with detailed metrics
- Comparative insights on best performers

## Expected Performance

### Regression Task
| Method | Expected Test Loss | Stability |
|--------|-------------------|-----------|
| ExpSAV | ~10^-4 - 10^-5 | High |
| IEQ Adaptive | ~10^-4 - 10^-5 | High |
| IEQ Full | ~10^-5 - 10^-6 | Very High |
| SAV | ~10^-4 | Medium |
| Adam | ~10^-4 | High |
| SGD | ~10^-3 | Medium |

### Classification Task
| Method | Expected Accuracy | Training Speed |
|--------|------------------|----------------|
| ExpSAV | 90-95% | Medium |
| IEQ Adaptive | 90-95% | Fast |
| IEQ Full | 90-95% | Slow |
| SAV | 88-93% | Medium |
| Adam | 95%+ | Fast |
| SGD | 85-90% | Medium |

## Computational Complexity

| Method | Per-Iteration Cost | Memory |
|--------|-------------------|---------|
| ExpSAV | O(n²) | O(n²) |
| IEQ Full | O(n³) | O(batch²) |
| IEQ Adaptive | O(n) | O(n) |
| SAV | O(n²) | O(n²) |
| Adam | O(n) | O(n) |
| SGD | O(n) | O(n) |

## Reproducibility

All experiments use:
- Random seed: 0
- Torch seed: 0
- NumPy seed: 0
- Deterministic mode when possible

## Troubleshooting

### MNIST Data Not Found
Run the data generation script:
```bash
cd data/MNIST
python MNIST.py
cd ../..
```

### Out of Memory
- Reduce batch_size in scripts
- Use smaller model (reduce m from 100 to 50)
- Run classification experiments with fewer epochs

### Experiments Taking Too Long
For quick testing:
- Reduce num_epochs in scripts
- Use IEQ_Adaptive instead of IEQ_Full
- Run only classification tasks (100 epochs vs 50,000)

## Citation

If you use these experimental setups in your research, please cite:

```bibtex
@article{your_paper,
  title={Efficient and Stable Methods for Deep Learning in Gradient Flows},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## Notes

- All plots are generated with matplotlib and displayed during training
- Set `isRecord = True` in scripts to enable Weights & Biases logging
- Modify hyperparameters in individual scripts or use `experiment_config.yaml` as reference
- GPU is automatically used if available (CUDA)
