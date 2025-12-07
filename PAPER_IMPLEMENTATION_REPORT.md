# Paper Implementation Report

## Reference Paper
**Title:** Efficient and stable SAV-based methods for gradient flows arising from deep learning
**Authors:** Ziqi Ma, Zhiping Mao, Jie Shen
**Journal:** Journal of Computational Physics 505 (2024) 112911
**DOI:** 10.1016/j.jcp.2024.112911

## Summary

This repository implements the SAV (Scalar Auxiliary Variable) based optimization methods described in the paper for training neural networks. The methods treat neural network training as solving gradient flows from a continuous point of view.

---

## Algorithms from the Paper

### Core SAV Methods

#### 1. **Algorithm 2: Vanilla SAV Scheme** ✅ IMPLEMENTED
- **Location:** `SAV_Regression.py`, `SAV_Classification.py`
- **Key Equations:** (17a-b), (19-20)
- **Description:** Introduces auxiliary variable `r = √(I(θ) + C)` to stabilize the gradient flow
- **Implementation Details:**
  ```python
  # Lines 68-80 in SAV_Regression.py
  θ^(n+1,1) = θ^n
  θ^(n+1,2) = -Δt/√(I(θ^n)+C) * (I + ΔtL)^(-1) * ∇I(θ^n)
  r^(n+1) = r^n / (1 + Δt * (∇I, (I+ΔtL)^(-1)∇I) / (2(I+C)))
  θ^(n+1) = θ^(n+1,1) + r^(n+1) * θ^(n+1,2)
  ```
- **Status:** ✅ Correctly matches paper formulation

#### 2. **Algorithm 3: Restart SAV Scheme** ✅ NEWLY IMPLEMENTED
- **Location:** `ResSAV_Regression.py`
- **Key Feature:** Resets `r̂^n = √(I(θ^n) + C)` at each step
- **Purpose:** Prevents `r^n` from decaying to 0 too rapidly
- **Advantages:** Better accuracy and correct steady state solution
- **Status:** ✅ Implemented according to paper specification

#### 3. **Algorithm 4: Relaxed SAV Scheme** ✅ NEWLY IMPLEMENTED
- **Location:** `RelSAV_Regression.py`, `RelSAV_Classification.py`
- **Key Equations:** (22a-c), (23-24)
- **Key Feature:** Uses relaxation parameter ξ₀ computed from optimization
  ```python
  r^(n+1) = ξ₀ * r̃^(n+1) + (1-ξ₀) * r̂^(n+1)
  ξ₀ = max{0, (-b - √(b²-4ac)) / (2a)}
  ```
- **Purpose:** Combines vanilla SAV stability with restart SAV accuracy
- **Advantages:** Unconditionally energy stable + links r^n directly to r(t^n)
- **Status:** ✅ Implemented with relaxation parameter η=0.99 (default from paper)

#### 4. **Algorithm 5: Adaptive SAV Scheme** ✅ NEWLY IMPLEMENTED
- **Location:** `AdaptiveSAV_Regression.py`
- **Key Equations:** (3-4) from Adam + SAV combination
- **Key Feature:** Combines SAV with Adam's adaptive learning rate strategy
  ```python
  # Adam momentum and variance
  m^(n+1) = β₁m^n + (1-β₁)∇I(θ^n)
  v^(n+1) = β₂v^n + (1-β₂)||∇I(θ^n)||²

  # Bias correction
  m̂ = m / (1 - β₁^n)
  v̂ = v / (1 - β₂^n)

  # Adaptive step size
  Δ̂t = Δt / √(v̂ + ε)
  ```
- **Purpose:** Improves efficiency for complex problems
- **Parameters:** β₁=0.9, β₂=0.999, ε=1e-8 (standard Adam values)
- **Status:** ✅ Implemented combining SAV scheme with adaptive strategy

### Space Discretization Methods

#### 5. **Algorithm 1: Smoothed Particle Method (SPM)** ✅ NEWLY IMPLEMENTED
- **Location:** `SPM_SAV_Regression.py`
- **Key Equations:** (10), (12)
- **Key Feature:** Uses smooth kernel φₕ(θ - θₖ) instead of Dirac delta δ(θ - θₖ)
  ```python
  π̃(θ) = (1/m) Σₖ φₕ(θ - θₖ)
  # Monte Carlo integration over ξ ~ N(0, I)
  ```
- **Implementation:**
  - Smoothing parameter: h = 0.0001 (default from paper)
  - Monte Carlo samples: J = 10 (default from paper)
  - Uses Gaussian perturbations: ξ ~ N(0, h²I)
- **Purpose:** Better accuracy than Particle Method (PM)
- **Status:** ✅ Implemented with Monte Carlo integration

---

## Comparison with Existing Code

### ✅ What Was Already Correct

| File | Method | Paper Section | Status |
|------|--------|---------------|---------|
| `SAV_Regression.py` | Vanilla SAV | Algorithm 2 | ✅ Correct |
| `SAV_Classification.py` | Vanilla SAV | Algorithm 2 | ✅ Correct |
| `SGD_Regression.py` | SGD Baseline | Section 2 | ✅ Baseline |
| `model/LinearModel.py` | One-hidden-layer NN | Section 2 | ✅ Correct |

### 🆕 What Was Missing (Now Implemented)

| File | Method | Paper Section | Implementation Date |
|------|--------|---------------|-------------------|
| `ResSAV_Regression.py` | Restart SAV | Algorithm 3 | 2025-12-05 |
| `RelSAV_Regression.py` | Relaxed SAV | Algorithm 4 | 2025-12-05 |
| `RelSAV_Classification.py` | Relaxed SAV (MNIST) | Algorithm 4 | 2025-12-05 |
| `AdaptiveSAV_Regression.py` | Adaptive SAV | Algorithm 5 | 2025-12-05 |
| `SPM_SAV_Regression.py` | SPM + SAV | Algorithm 1+2 | 2025-12-05 |

### ⚠️ What's in Code But NOT in Paper

| File | Method | Note |
|------|--------|------|
| `ESAV_Regression.py` | Exponential SAV | Uses `r = C*exp(I(θ))` - different formulation |
| `ESAV_Classification.py` | Exponential SAV | Not described in the paper |
| `IEQ_Regression.py` | Implicit Euler | Uses Jacobian - not in paper |
| `IEQ_Classification.py` | Implicit Euler | Not in paper |

---

## Key Parameters from Paper

### Default Parameters (Table 1, Section 3)

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Number of neurons | m | 100-10000 | Hidden layer width |
| SAV constant | C | 1-100 | Ensures I(θ)+C ≥ 0 |
| Linear operator coefficient | λ | 0-10 | For L(θ) = λθ |
| Learning rate | Δt (lr) | 0.01-1.0 | Time step size |
| Batch size | l | 64-256 | Mini-batch size |
| Relaxation parameter | η | 0.99 | For RelSAV method |
| SPM smoothing | h | 0.0001 | Smoothing bandwidth |
| SPM samples | J | 10 | Monte Carlo samples |
| Adam β₁ | β₁ | 0.9 | First moment decay |
| Adam β₂ | β₂ | 0.999 | Second moment decay |

---

## Numerical Examples from Paper

### Example 1 (Section 3.1.1)
**Target Function:**
```
f*(x₁,...,xᴰ) = sin(Σpᵢxᵢ) + cos(Σqᵢxᵢ)
```
- **Dataset:** Random data in (0,1)^D
- **Dimensions tested:** D = 20, 40
- **Key finding:** SAV methods work with larger learning rates

### Example 2 (Section 3.1.2)
**Target Function:**
```
f*(x₁,...,xᴰ) = Σ cᵢxᵢ²
```
- **Domain:** [0,5]^D
- **Key finding:** Adaptive SAV necessary for complex problems

### Example 3 (Section 3.1.3)
**Target Function:**
```
f*(x) = exp(-10||x||²)
```
- **Dataset:** Non-uniform, x ~ N(0, 0.2)
- **Challenge:** Sharp gradients near origin
- **Key finding:** Adaptive RelSAV outperforms Adam

### Example 4 (Section 3.2)
**Dataset:** MNIST (60000 training, 10000 test)
- **Architecture:** [784,1] → [W,a] with ReLU, m=100
- **Key finding:** SPM slightly better than PM for classification

---

## How to Reproduce Paper Results

### 1. Setup Environment
```bash
pip install torch torchvision matplotlib numpy
```

### 2. Generate Data
```bash
cd data
python data_generate.py  # For regression data
python MNIST/MNIST.py    # For MNIST data
```

### 3. Run Experiments

#### Vanilla SAV (Baseline)
```bash
python SAV_Regression.py       # Regression
python SAV_Classification.py   # MNIST classification
```

#### Restart SAV (Better Accuracy)
```bash
python ResSAV_Regression.py
```

#### Relaxed SAV (Best Balance)
```bash
python RelSAV_Regression.py
python RelSAV_Classification.py  # MNIST
```

#### Adaptive SAV (For Complex Problems)
```bash
python AdaptiveSAV_Regression.py
```

#### SPM (Higher Accuracy Space Discretization)
```bash
python SPM_SAV_Regression.py
```

### 4. Comparison with Baselines
```bash
python SGD_Regression.py        # Standard SGD
python Adam.py                  # Standard Adam (if available)
```

---

## Expected Results (From Paper)

### Stability
- **SAV methods** converge with learning rate lr=0.5-1.0
- **SGD/Adam** fail or oscillate with lr>0.1

### Efficiency
- **SAV methods** achieve 2-3 orders of magnitude better loss with same epochs
- **Adaptive SAV** converges faster than vanilla SAV for complex problems

### Accuracy
- **SPM** slightly better than **PM** (especially for classification)
- **RelSAV** better than **SAV** and **ResSAV**

### Key Findings (Figure References)
- Fig 1: Energy dissipation with full batch
- Fig 2: SPM vs PM accuracy comparison
- Fig 3: Different learning rates (lr=0.5, lr=1)
- Fig 4: Adaptive vs fixed learning rate
- Fig 10: Adaptive SAV comparison with Adam/Adagrad/RMSprop

---

## Code Structure

```
Efficient-and-Stable-Methods-for-DL-in-Gradient-Flows/
├── model/
│   └── LinearModel.py              # One-hidden-layer neural network
├── data/
│   ├── data_generate.py            # Generate regression data
│   ├── MNIST/MNIST.py              # Load MNIST data
│   └── *.pt                        # Saved datasets
├── utilize.py                      # Helper functions (flatten/unflatten params)
│
├── SAV_Regression.py               # ✅ Algorithm 2 (Vanilla SAV)
├── SAV_Classification.py           # ✅ Algorithm 2 (MNIST)
├── ResSAV_Regression.py            # 🆕 Algorithm 3 (Restart SAV)
├── RelSAV_Regression.py            # 🆕 Algorithm 4 (Relaxed SAV)
├── RelSAV_Classification.py        # 🆕 Algorithm 4 (MNIST)
├── AdaptiveSAV_Regression.py       # 🆕 Algorithm 5 (Adaptive SAV)
├── SPM_SAV_Regression.py           # 🆕 Algorithm 1+2 (SPM)
│
├── SGD_Regression.py               # Baseline: Standard SGD
├── SGD_Classification.py           # Baseline: Standard SGD (MNIST)
├── Adam.py                         # Baseline: Adam optimizer
│
├── ESAV_Regression.py              # ⚠️ NOT in paper (Exponential SAV)
├── ESAV_Classification.py          # ⚠️ NOT in paper
├── IEQ_Regression.py               # ⚠️ NOT in paper (Implicit Euler)
└── IEQ_Classification.py           # ⚠️ NOT in paper
```

---

## Implementation Details

### 1. Vanilla SAV (Algorithm 2)
**Mathematical Formulation:**
```
θ^(n+1) - θ^n
─────────────── + L(θ^(n+1)) + (∇I(θ^n)/√(I(θ^n)+C)) * r^(n+1) - L(θ^n) = 0
      Δt

r^(n+1) - r^n       1
─────────────── = ─────────────── * (∇I(θ^n), (θ^(n+1)-θ^n)/Δt)
      Δt        2√(I(θ^n)+C)
```

**Efficient Implementation:**
```python
θ^(n+1,1) = θ^n
θ^(n+1,2) = -(Δt/√(I(θ^n)+C)) * (I + ΔtL)^(-1) * ∇I(θ^n)
r^(n+1) = r^n / (1 + Δt*(∇I, (I+ΔtL)^(-1)∇I)/(2(I+C)))
θ^(n+1) = θ^(n+1,1) + r^(n+1) * θ^(n+1,2)
```

### 2. Energy Stability
**Theorem 1:** Vanilla SAV and Relaxed SAV are unconditionally energy stable:
```
(r^(n+1))² - (r^n)² ≤ 0  for any Δt > 0
```

### 3. Linear Operator
```python
L(θ) = λ * (-Δ)^k * θ
```
Most commonly: `L(θ) = λθ` (k=0), where λ ≥ 0

---

## Performance Tuning Guide

### Choosing SAV Parameters

#### SAV Constant C
- **Small C (1-10):** For well-scaled problems
- **Large C (100-1000):** For problems with large loss values
- **Rule:** Ensure `I(θ) + C > 0` always

#### Linear Operator λ
- **λ = 0:** Pure SAV (no damping)
- **λ = 1-4:** Light damping (paper default)
- **λ = 10+:** Strong damping (very smooth convergence)

#### Learning Rate Δt
- **Fixed:** 0.1 - 1.0 (much larger than SGD!)
- **Adaptive:** Start with 0.1, let Adam-style adaptation handle rest

### Method Selection Guide

| Problem Type | Recommended Method | Why |
|--------------|-------------------|-----|
| Simple regression | Vanilla SAV | Fast, simple |
| Complex regression | Adaptive SAV or RelSAV | Better convergence |
| Classification (MNIST) | RelSAV | Best balance |
| High accuracy needed | SPM + RelSAV | SPM gives better accuracy |
| Unstable training | RelSAV | Guarantees energy stability |

---

## Validation Checklist

- [x] Vanilla SAV matches Algorithm 2
- [x] Restart SAV matches Algorithm 3
- [x] Relaxed SAV matches Algorithm 4
- [x] Adaptive SAV matches Algorithm 5
- [x] SPM matches Algorithm 1
- [x] Energy dissipation verified (Fig 1)
- [ ] Reproduce Fig 2 (PM vs SPM accuracy)
- [ ] Reproduce Fig 3 (learning rate comparison)
- [ ] Reproduce Fig 4 (adaptive comparison)
- [ ] Reproduce MNIST results (Fig 11)

---

## Future Work

### To Fully Reproduce Paper:
1. ✅ Implement all missing algorithms
2. ⏳ Run comprehensive experiments matching all paper figures
3. ⏳ Create benchmark comparison plots
4. ⏳ Add multi-layer neural network support (paper Example 5)
5. ⏳ Implement PDE solving examples (Burgers equation, Example 5)

### Additional Enhancements:
- [ ] Add learning rate schedulers
- [ ] Support for more activation functions (tanh, sigmoid)
- [ ] GPU optimization for large-scale problems
- [ ] Automatic parameter tuning (C, λ)
- [ ] Visualization tools for energy evolution

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{ma2024efficient,
  title={Efficient and stable SAV-based methods for gradient flows arising from deep learning},
  author={Ma, Ziqi and Mao, Zhiping and Shen, Jie},
  journal={Journal of Computational Physics},
  volume={505},
  pages={112911},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2024.112911}
}
```

---

## Contact

For questions about the implementation, please refer to:
- **Paper:** https://doi.org/10.1016/j.jcp.2024.112911
- **Code Repository:** [Current Repository]
- **Authors:** Ziqi Ma (maziqi@stu.xmu.edu.cn), Zhiping Mao (zpmao@xmu.edu.cn), Jie Shen (jshen@eitech.edu.cn)

---

*Last Updated: 2025-12-05*
*Implementation Status: ✅ All core algorithms from paper implemented*
