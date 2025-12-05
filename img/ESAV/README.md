# ExpSAV (Exponential SAV) Results

This folder contains experimental results for the ExpSAV optimization method.

## Algorithm

ExpSAV uses an exponential auxiliary variable $r = C \cdot \exp(L(w))$ to avoid numerical instabilities in the original SAV method.

**Key features:**
- Stable formulation without $r^{-n}$ terms
- Recursive update: $r^{n+1} = \frac{r^n}{1 + \Delta t S^n}$
- Energy monotonicity guaranteed

## Experiments

### Regression Task
- **Script:** `ESAV_Regression.py`
- **Dataset:** Gaussian data with $y = \exp(-x^2) + \epsilon$
- **Model:** Two-layer network with ReLU (m=100)
- **Hyperparameters:**
  - $C = 1.0$
  - $\lambda = 1.0$
  - $\Delta t = 0.1$

### Classification Task
- **Script:** `ESAV_Classification.py`
- **Dataset:** MNIST (28x28 handwritten digits)
- **Model:** 784 → 100 → 10 with ReLU
- **Hyperparameters:**
  - $C = 1.0$
  - $\lambda = 0.0$
  - $\Delta t = 0.1$
  - Epochs: 100

## Results

### Expected Performance
- **Regression:** Smooth loss decay, stable training
- **Classification:** ~90-95% test accuracy on MNIST
- **Comparison:** More stable than SAV, competitive with Adam/SGD

### Usage

To reproduce results:
```bash
# Regression
python ESAV_Regression.py

# Classification
python ESAV_Classification.py
```

## Notes

- The auxiliary variable `r` is initialized at the first batch and persists throughout training
- Do not reset `r` between epochs
- Monitor loss curves for energy dissipation property
