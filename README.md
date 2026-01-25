# Battery-SOH-estimation

## Week 1: Physics-Informed Neural Networks (PINNs)

### Task 1: Zero-Shot Learning of Exponential Decay

**Objective:** Train a neural network to learn `f(x) = e^(-x)` using only physics constraints, without any training data.

**Approach:**
- Implemented a PINN with 2 hidden layers (20 neurons each) using `tanh` activation
- Physics constraint: Enforced differential equation `f'(x) = -f(x)`
- Initial condition: `f(0) = 1`
- Training domain: `x ∈ [-2, 2]` with 100 collocation points

**Results:**
- Successfully learned the exponential function without labeled data
- Mean Absolute Error: ~0.0004 
- Demonstrates PINN's ability to learn from governing equations alone

**Key Insight:** PINNs can solve differential equations by encoding physical laws directly into the loss function, eliminating the need for extensive labeled datasets.

---

### Task 2: Simple Harmonic Motion with Sparse Data

**Objective:** Compare PINN vs regular neural network performance when extrapolating from minimal training data.

**Setup:**
- Training data: Only 3 points from `sin(t)` at `t = 0, 1, 2`
- Test domain: Extrapolate to `t ∈ [0, 10]`
- Physics constraint: SHM equation `y'' + y = 0`

**Models Compared:**
1. **PINN with tanh activation** - Physics + data loss
2. **PINN with sigmoid activation** - Physics + data loss
3. **Regular NN** - Data loss only

**Results:**
| Model | Mean Absolute Error |
|-------|---------------------|
| PINN (tanh) | ~0.32 |
| PINN (sigmoid) | ~0.45 |
| Regular NN | ~0.62 |

**Discussion:**
- **PINN (tanh)** accurately extrapolated the sine wave beyond training data by leveraging the physics of simple harmonic motion
- **Tanh activation** significantly outperformed sigmoid due to better gradient flow and symmetric range
- **Regular NN** collapsed to a constant value outside the training domain—classic overfitting to sparse data
- Physics constraints act as a powerful regularizer, enabling generalization with minimal data

---

## Week 2: Battery Modeling with PyBaMM

### Task 1: Lithium Concentration Dynamics in Battery Particles

**Objective:** Visualize lithium concentration gradients during 1C discharge using the Single Particle Model (SPM).

**Implementation:**
- Simulated 1C discharge to 2.5V cutoff
- Extracted concentration profiles at 5 timestamps: 0s, 900s, 1800s, 2700s, 3600s
- Plotted **Concentration vs Radius** across particle geometry

**Results:**
- **Early discharge (t=0s):** Nearly uniform lithium distribution
- **Mid-discharge (t=1800s):** Clear concentration gradient forms
- **Late discharge (t=3600s):** Steep gradient with surface depletion

**Physical Interpretation:**
- Lithium is extracted from the particle **surface** during discharge
- **Diffusion limitations** prevent fast replenishment from particle center
- Steeper gradients → higher internal resistance → voltage drop
- This explains why batteries deliver less capacity at high C-rates

**Practical Implications:**
- Smaller particles reduce diffusion distances → better rate capability
- Tradeoff: Smaller particles have lower volumetric energy density
- Optimal particle size balances power and energy density

---

### Task 2: SOC-to-Voltage Mapping Function

**Objective:** Develop a fast mathematical function to convert State of Charge (SOC) to Open Circuit Voltage (OCV).

**Approach:**
- Extracted OCV data from PyBaMM's **Chen2020** parameter set
- Used stoichiometry-based electrode potentials: `OCV = U_pos - U_neg`
- Fitted a **5th-degree polynomial** to the SOC-OCV curve
- Generated 100 data points across SOC ∈ [0, 1]

**Function:**
```python
def soc_to_voltage(soc):
    return np.polyval(coeffs, soc)
```

**Results:**
- Polynomial fit closely matches PyBaMM OCV data
- Example: SOC = 0.5 → Voltage ≈ 3.524V
- Fast computation: Significantly faster than full physics simulation

**Applications:**
- **Battery Management Systems (BMS):** Real-time SOC estimation
- **Control algorithms:** Voltage prediction for charging/discharging strategies
- **Fast lookup tables:** Replace computationally expensive physics models in production


# Week 3 — Physics-Constrained Transformer for Battery SoH Estimation

This week focuses on building an **end-to-end physics-constrained Transformer pipeline** for lithium-ion battery **State of Health (SoH)** estimation using the NASA battery dataset.

The key idea is to **separate physics from learning**:
- Use **high-resolution physics data** to enforce conservation laws
- Use **low-resolution binned sequences** for efficient Transformer training

---

## 1. Dataset Overview

**Source**  
NASA Li-ion Battery Aging Dataset (via Kaggle, cleaned version).

**Signals Used**
- Voltage (V)
- Current (A)
- Temperature (°C)
- Time (s)

**Cycle Type**
- Discharge cycles only (most informative for degradation)

---

## 2. Preprocessing Pipeline

The preprocessing stage converts raw discharge cycles into **two synchronized datasets**.

### 2.1 Cycle Cleaning

For each discharge cycle:

1. **Sort by time** and remove duplicate timestamps  
2. **Apply voltage cutoff**
   - Remove all points below **2.7 V**
   - Avoids non-linear and unsafe discharge region  
3. **Coulomb counting**: $Q = \int I(t)\, dt$

5. **Capacity filtering**
   - Discard cycles with:
     - `capacity < 1 Ah` (partial / corrupted cycles)
     - `capacity > 2 Ah` (non-physical)

6. **State of Charge (SoC)**:
   
    $\text{SoC}(t) = 100 \times \left(1 + \frac{\text{CumQ}(t)}{Q_{\text{cycle}}}\right)$

7. **State of Health (SoH)**:

    $\mathrm{SoH} = 100 \times \frac{Q_{\text{cycle}}}{Q_{\text{nominal}}}$

This ensures:
- No aging information leakage
- Physically meaningful labels

---

## 3. Dual-Resolution Dataset Construction

### 3.1 Physics Dataset — `phys200`

Each cleaned cycle is **resampled to 200 uniform time points**.

**Purpose**
- Accurate numerical integration
- Stable computation of:
  - Total discharged charge
  - Total discharged energy

**Method**
- Linear interpolation over a uniform time grid
- Preserves the original discharge curve shape

**Used for**
- Physics supervision only (not model input)

---

### 3.2 ML Dataset — `main20`

Each cycle is **compressed into 20 time bins**.

**Binning Strategy**
- Split the resampled cycle into 20 equal time segments
- Aggregate each bin using:
  - Mean Voltage
  - Mean Current
  - Mean Temperature
  - Mean SoC
  - Total `dt`

**Why 20 bins?**
- Fixed-length sequences for Transformers
- Removes noise
- Retains low-frequency degradation patterns
- Efficient training

Each cycle becomes:
(20 timesteps × 5 features)


---

## 4. Model Architecture — Physics-Constrained Transformer

### 4.1 Transformer Encoder

**Input**

Features per bin:
- Voltage
- Current
- Temperature
- SoC
- dt (acts as continuous positional encoding)

**Architecture**
- Linear embedding layer
- Multi-head self-attention encoder (stacked layers)
- No explicit sinusoidal encoding (time is included as a feature)

---

### 4.2 Output Heads

The Transformer predicts **three physically meaningful quantities**:

1. **SoH (normalized, bounded)**
   - Output range: `[0, 1]` (via `Sigmoid`)
2. **Total discharged charge**
3. **Total discharged energy**

These outputs allow physics constraints to be enforced **globally**, not point-wise.

---

## 5. Physics-Constrained Loss Function

The total loss is:

$$
\mathcal{L} =
\underbrace{\text{MSE}(\widehat{\text{SoH}}, \text{SoH})}_{\text{Data loss}}
+
\lambda_Q \underbrace{\text{MSE}(\widehat{Q}, Q)}_{\text{Charge conservation}}
+
\lambda_E \underbrace{\text{MSE}(\widehat{E}, E)}_{\text{Energy conservation}}
$$
### Key Design Choices

- **Targets are normalized**
  - SoH ∈ [0, 1]
  - Charge normalized by nominal capacity
  - Energy normalized by a reference value
- Physics losses act as **regularization**, not dominance
- No PDE residuals → fast training

---

## 6. Training Strategy

- Optimizer: **Adam**
- Batch training with `DataLoader`
- Progress tracked using `tqdm`
- Model checkpoints saved every 5 epochs
- Training can be resumed from any checkpoint (optimizer state included)

---

---

## 7. Results

- Model successfully learns a **monotonic relationship** between discharge behavior and SoH
- No prediction collapse
- Physics constraints stabilize training
- Predictions are conservative but physically plausible
---

## 8. Key Takeaways

- Separating **physics resolution** and **learning resolution** is critical
- Global physics constraints are sufficient and efficient
- Transformers can learn degradation patterns when sequences are well-structured


---





