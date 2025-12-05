# IEQ (Invariant Energy Quadratization) Results

This folder contains experimental results for both IEQ optimization methods.

## Algorithms

IEQ transforms the loss function into quadratic form using auxiliary variables $q = f(w) - y$.

### Version A: Full Jacobian Method
- **Scripts:** `IEQ_Regression.py`, `IEQ_Classification.py`
- **Method:** Exact solution of $(I + \Delta t J J^T)^{-1}$
- **Complexity:** $O(n^3)$ per iteration
- **Accuracy:** High precision

### Version B: Adaptive Step Size Method
- **Scripts:** `IEQ_Regression_Adaptive.py`, `IEQ_Classification_Adaptive.py`
- **Method:** Simplified adaptive scaling factor
- **Complexity:** $O(n)$ per iteration
- **Efficiency:** Much faster than Full Jacobian

## Experiments

### Regression Task
**Full Jacobian:**
- **Script:** `IEQ_Regression.py`
- **Dataset:** Gaussian data with $y = \exp(-x^2) + \epsilon$
- **Model:** Two-layer network with ReLU (m=100)
- **Hyperparameters:** $\Delta t = 0.1$

**Adaptive:**
- **Script:** `IEQ_Regression_Adaptive.py`
- **Dataset:** Same as above
- **Hyperparameters:** $\Delta t = 0.1$, $\epsilon = 10^{-8}$

### Classification Task
**Full Jacobian:**
- **Script:** `IEQ_Classification.py`
- **Dataset:** MNIST (28x28 handwritten digits)
- **Model:** 784 → 100 → 10 with ReLU
- **Hyperparameters:** $\Delta t = 0.1$, Epochs: 100

**Adaptive:**
- **Script:** `IEQ_Classification_Adaptive.py`
- **Dataset:** Same as above
- **Hyperparameters:** $\Delta t = 0.1$, $\epsilon = 10^{-8}$, Epochs: 100

## Mathematical Formulation

### Full Jacobian Method
$$
\begin{aligned}
q^{n+1} &= (I + \Delta t J J^T)^{-1} q^n \\
w^{n+1} &= w^n - \Delta t J^T q^{n+1}
\end{aligned}
$$

### Adaptive Method
$$
\begin{aligned}
\alpha^n &= \frac{1}{1 + \Delta t \frac{\|\nabla_w L(w^n)\|^2}{\|q^n\|^2 + \epsilon}} \\
w^{n+1} &= w^n - \Delta t \alpha^n \nabla_w L(w^n) \\
q^{n+1} &= \frac{q^n}{1 + \Delta t \frac{\|\nabla_w L(w^n)\|^2}{\|q^n\|^2 + \epsilon}}
\end{aligned}
$$

## Results Comparison

### Expected Performance
- **Full Jacobian:** Higher precision but slower computation
- **Adaptive:** Faster training with comparable accuracy
- **Trade-off:** Adaptive method is recommended for large-scale problems

### Computational Cost
| Method | Per-iteration Cost | Memory Usage |
|--------|-------------------|--------------|
| Full Jacobian | $O(n^3)$ | $O(n^2)$ |
| Adaptive | $O(n)$ | $O(n)$ |

## Usage

To reproduce results:
```bash
# Full Jacobian - Regression
python IEQ_Regression.py

# Full Jacobian - Classification
python IEQ_Classification.py

# Adaptive - Regression
python IEQ_Regression_Adaptive.py

# Adaptive - Classification
python IEQ_Classification_Adaptive.py
```

## Notes

- The auxiliary variable `q` is initialized at the first batch
- For classification tasks, use one-hot encoding for $y$
- The $\epsilon$ parameter prevents division by zero in adaptive method
- Monitor the adaptive factor $\alpha^n$ to ensure stability
