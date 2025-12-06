# Efficient and Stable Methods for Deep Learning in Gradient Flows

This repository implements efficient and stable optimization algorithms based on gradient flow methods for deep learning. The algorithms are designed to maintain energy dissipation properties while providing numerical stability and computational efficiency.

## Overview

We implement three main approaches:
1. **SAV (Scalar Auxiliary Variable)** - Original method with $r = \sqrt{L(w) + C}$
2. **ExpSAV (Exponential SAV)** - Improved stability using exponential auxiliary variable
3. **IEQ (Invariant Energy Quadratization)** - Quadratization-based method with adaptive step size

All methods are based on the gradient flow framework for optimization in deep learning:

$$
\frac{\partial w}{\partial t} = -\nabla_w E(w)
$$

where $E(w)$ is the energy functional (typically loss function + regularization).

---

## Algorithms

### 1. SAV Schema (Scalar Auxiliary Variable)

The original SAV method introduces an auxiliary variable $r = \sqrt{L(w) + C}$ to stabilize the gradient flow.

**Update equations:**

$$
\begin{aligned}
w^{n+1,*} &= -\Delta t (I + \Delta t\mathcal{L})^{-1}\nabla_w L(w^n) \\
S^n &= \langle \nabla_w L(w^n), (I + \Delta t\mathcal{L})^{-1}\nabla_w L(w^n) \rangle \\
r^{n+1} &= \frac{r^n}{1 + \Delta t \frac{S^n}{2(L(w^n) + C)}} \\
w^{n+1} &= w^n + \frac{r^{n+1}}{r^n} w^{n+1,*}
\end{aligned}
$$

**Properties:**
- Unconditionally energy stable
- Requires $C > 0$ to ensure $L(w) + C > 0$
- May encounter numerical issues when $L(w) \approx 0$

---

### 2. ExpSAV Schema (Exponential SAV)

ExpSAV uses an exponential auxiliary variable to avoid gradient vanishing/explosion issues in the original SAV formulation.

**Auxiliary variable:**
$$
r = C \cdot \exp(L(w))
$$

**Stable update scheme (without $r^{-n}$ terms):**

$$
\begin{aligned}
w^{n+1,*} &= -\Delta t (I + \Delta t\mathcal{L})^{-1}\nabla_w L(w^n) \\
S^n &= \langle \nabla_w L(w^n), (I + \Delta t\mathcal{L})^{-1}\nabla_w L(w^n) \rangle \\
r^{n+1} &= \frac{r^n}{1 + \Delta t S^n} \\
w^{n+1} &= w^n + \frac{r^{n+1}}{r^n} w^{n+1,*}
\end{aligned}
$$

**Key advantages:**
- **Improved stability:** Removes the $r^{-n}$ term that can cause gradient vanishing/explosion
- **Natural scaling:** Exponential form provides better numerical behavior across different loss scales
- **Energy monotonicity:** Since $\mathcal{L}$ is positive semi-definite, $S^n \geq 0$ ensures $r^{n+1} \leq r^n$

**Implementation notes:**
- Initialize: $r^0 = C \cdot \exp(L(w^0))$ where $C > 0$ is a scaling constant
- The auxiliary variable $r$ must persist across epochs (do not reset between epochs)
- Simplified Hessian approximation: $\mathcal{L} \approx \lambda I$ for efficiency

---

### 3. IEQ Schema (Invariant Energy Quadratization)

IEQ transforms the loss function into a quadratic form using auxiliary variables.

**System transformation:**
- Define scalar auxiliary variable: $q = f(w) - y$
- Transform loss to quadratic form: $L = \frac{1}{2}q^2$

#### Version A: Full Jacobian Method (High Precision)

**Update equations:**

$$
\begin{aligned}
J &= \nabla_w f(w^n) \quad \text{(Jacobian matrix)} \\
q^{n+1} &= (I + \Delta t J J^T)^{-1} q^n \\
w^{n+1} &= w^n - \Delta t J^T q^{n+1}
\end{aligned}
$$

**Computational complexity:** $O(n^3)$ due to matrix inversion

#### Version B: Adaptive Step Size Method (Efficient)

**Simplified approximation:**

$$
\|g\|^2 \approx \frac{\|\nabla_w L(w^n)\|^2}{\|q^n\|^2}
$$

**Update equations:**

$$
\begin{aligned}
q^{n+1} &= \frac{q^n}{1 + \Delta t \frac{\|\nabla_w L(w^n)\|^2}{\|q^n\|^2 + \epsilon}} \\
w^{n+1} &= w^n - \Delta t \alpha^n \nabla_w L(w^n)
\end{aligned}
$$

where the adaptive scaling factor is:

$$
\alpha^n = \frac{1}{1 + \Delta t \frac{\|\nabla_w L(w^n)\|^2}{\|q^n\|^2 + \epsilon}}
$$

**Key advantages:**
- **Computational efficiency:** $O(n)$ complexity
- **Adaptive behavior:** Automatically adjusts step size based on loss magnitude
- **Numerical stability:** $\epsilon$ regularization prevents division by zero

---

## Unified Relaxed Scheme

The relaxed approach combines dynamically computed intermediate values with ideal values based on definitions.

**Core idea:** For generalized auxiliary variable $v$ (corresponding to $r$ in SAV/ExpSAV or $q$ in IEQ):
- **Ideal value** $\hat{v}^{n+1}$: Computed directly from updated parameters $w^{n+1}$
- **Intermediate value** $\tilde{v}^{n+1}$: Computed from recursive formulas in base algorithms

**Relaxation update:**

$$
v^{n+1} = \xi_0 \hat{v}^{n+1} + (1-\xi_0)\tilde{v}^{n+1}
$$

where $\xi_0 \in [0,1]$ is determined by:

$$
\xi_0 = \min_{\xi \in [0,1]} \xi \quad \text{subject to energy dissipation constraints}
$$

**Benefits:**
- Maintains energy dissipation properties
- Corrects numerical drift in auxiliary variables
- Balances accuracy and stability

---

## Project Structure

```
.
├── README.md                       # This file
├── model/
│   └── LinearModel.py             # Neural network models
├── data/
│   ├── data_generate.py           # Data generation utilities
│   ├── plot_data.py               # Visualization tools
│   └── MNIST/                     # MNIST dataset handling
├── img/                           # Results and figures
│   ├── SAV/                       # SAV method results
│   ├── ESAV/                      # ExpSAV method results
│   ├── IEQ/                       # IEQ method results
│   ├── SGD/                       # SGD baseline results
│   └── Adam/                      # Adam baseline results
├── utilize.py                     # Helper functions
├── data_loader.py                 # Data loading utilities
│
├── SAV_Regression.py              # SAV for regression tasks
├── SAV_Classification.py          # SAV for classification tasks
├── ESAV_Regression.py             # ExpSAV for regression tasks
├── ESAV_Classification.py         # ExpSAV for classification tasks
├── IEQ_Regression.py              # IEQ for regression (Full Jacobian)
├── IEQ_Classification.py          # IEQ for classification (Full Jacobian)
├── IEQ_Regression_Adaptive.py     # IEQ for regression (Adaptive method)
├── IEQ_Classification_Adaptive.py # IEQ for classification (Adaptive method)
├── SGD_Regression.py              # SGD baseline
├── SGD_Classification.py          # SGD baseline
└── Adam.py                        # Adam baseline
```

---

## Usage

### Regression Example (ExpSAV)

```python
from model import LinearModel
import torch
import torch.nn as nn

# Load data
(x_train, y_train) = torch.load('data/Gaussian_train_data.pt')

# Initialize model
model = LinearModel.SinCosModel(m=100)
criterion = nn.MSELoss()

# ExpSAV hyperparameters
C = 1.0
lambda_ = 1.0
dt = 0.1

# Initialize auxiliary variable (persist across training!)
loss = criterion(model(x_train), y_train)
r = C * math.exp(loss.item())

# Training loop
for epoch in range(num_epochs):
    for X, Y in train_loader:
        # ... (see ESAV_Regression.py for complete implementation)
```

### Classification Example (IEQ - Adaptive)

```python
# IEQ with adaptive step size
for epoch in range(num_epochs):
    for X, Y in train_loader:
        pred = model(X)
        loss = criterion(pred, Y)

        # Compute gradient
        model.zero_grad()
        loss.backward()
        grad_norm = torch.norm(flatten_grad(model))

        # Update auxiliary variable q
        q = pred - Y  # f(w) - y
        q_norm = torch.norm(q)
        alpha = 1.0 / (1.0 + dt * (grad_norm**2) / (q_norm**2 + 1e-8))

        # Update parameters
        with torch.no_grad():
            theta = flatten_params(model.W, model.a)
            theta_new = theta - dt * alpha * flatten_grad(model)
            model.W, model.a = unflatten_params(theta_new, ...)
```

---

## Experiments

### Quick Start

To run all experiments:
```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
chmod +x experiments/run_all_experiments.sh
./experiments/run_all_experiments.sh

# View results summary
python experiments/summarize_results.py
```

For detailed experimental results and analysis, see **[EXPERIMENTS.md](EXPERIMENTS.md)**.

### Regression Task
- **Dataset:** Gaussian noise data, $y = \exp(-x^2) + \epsilon$
- **Model:** Two-layer neural network with ReLU activation (m=100 neurons)
- **Metrics:** MSE loss on train/test sets
- **Expected:** ExpSAV and IEQ achieve test loss ~10⁻⁵

### Classification Task
- **Dataset:** MNIST handwritten digits
- **Model:** Two-layer neural network (784 → 100 → 10)
- **Metrics:** Cross-entropy loss and accuracy
- **Expected:** ExpSAV and IEQ achieve 92-94% accuracy

### Results
- **Detailed Analysis:** See [EXPERIMENTS.md](EXPERIMENTS.md)
- **Experiment Configs:** See [experiments/experiment_config.yaml](experiments/experiment_config.yaml)
- **Implementation Results:** See `img/` folder for method-specific results:
  - `img/SAV/` - Original SAV results
  - `img/ESAV/` - ExpSAV results
  - `img/IEQ/` - IEQ (Full & Adaptive) results
  - `img/SGD/` - SGD baseline
  - `img/Adam/` - Adam baseline

---

## References

1. Shen, J., Xu, J., & Yang, J. (2018). The scalar auxiliary variable (SAV) approach for gradient flows. *Journal of Computational Physics*.

2. Huang, F., Shen, J., & Yang, Z. (2020). A highly efficient and accurate new scalar auxiliary variable approach for gradient flows. *SIAM Journal on Scientific Computing*.

3. Yang, X., & Zhao, J. (2017). On linear and unconditionally energy stable algorithms for variable mobility Cahn-Hilliard type equation with logarithmic Flory-Huggins potential. *Communications in Computational Physics*.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{esav_ieq_2025,
  title={Efficient and Stable Methods for Deep Learning in Gradient Flows},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Efficient-and-Stable-Methods-for-DL-in-Gradient-Flows}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions and feedback, please open an issue on GitHub.
