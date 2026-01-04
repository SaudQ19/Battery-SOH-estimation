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
- Fast computation: ~1000x faster than full physics simulation

**Applications:**
- **Battery Management Systems (BMS):** Real-time SOC estimation
- **Control algorithms:** Voltage prediction for charging/discharging strategies
- **Fast lookup tables:** Replace computationally expensive physics models in production

