def dissipation_explanation() -> str:
    return r"""
### What are these energy dissipation plots?

These plots show **how the total energy of the system changes over time**.
They are used to understand whether the **numerical method** conserves or dissipates energy.

Two views are shown because they answer **two slightly different questions**.

---

#### Energy vs time: *absolute energy*

This plot shows the **raw value of the total energy** `E(t)`.

It answers:
- "Is the total energy increasing, decreasing, or staying constant?"
- "Is the behavior smooth or irregular?"

How to read it:
- A **flat curve** $\to$ energy is conserved.
- A **slowly decreasing curve** $\to$ energy dissipation (physical or numerical).
- A **rapid drop** $\to$ excessive numerical dissipation.
- An **increasing curve** $\to$ non-physical behavior (instability).

This plot is useful to see the **actual magnitude** of the energy.

---

#### Normalized energy $E / E_0$ vs time: *relative energy*

This plot shows the energy **relative to the initial one**.

- `E / E₀ = 1` $\to$ energy equal to the initial value.
- `E / E₀ < 1` $\to$ energy loss.
- `E / E₀ > 1` $\to$ energy growth (problematic).

This plot is important because:
- It removes scaling effects.
- It makes **different simulations directly comparable**.
- Small energy losses become easier to see.

Even a small drift away from 1 can indicate **numerical dissipation**.

---

### How to interpret both plots together

✅ **Correct behavior**
  - `E(t)` roughly flat
  - `E / E₀` stays close to **1**
Energy is conserved.

🟡 **Expected dissipation**
  - `E(t)` slowly decreases
  - `E / E₀` smoothly goes below 1
Physical or mild numerical dissipation.

⚠️ **Too much dissipation**
  - `E(t)` drops quickly
  - `E / E₀` rapidly goes toward 0
Time step too large or overly dissipative scheme.

❌ **Wrong behavior**
  - `E(t)` increases
  - `E / E₀` goes above 1
Instability or implementation issue.

---

In general, energy should **never grow**, it should either stay constant or decrease smoothly, depending on the model.
"""

def dissipation_explanation_for_dummies() -> str:
    return r"""
### What do these plots indicate? (For dummies)

These plots show **whether the simulation is behaving correctly or not**.

- The first plot shows **how much total energy the system has**.
- The second plot shows the **same energy compared to the initial one**.

We **must look at both** plots to get the full picture.

---

### How to read them quickly

✅ **Everything is OK**
  - Energy stays almost constant;
  - Normalized energy stays close to **1** $\to$ The method is stable and correct.

🟡 **Some energy loss**
  - Energy slowly decreases;
  - Normalized energy goes slightly below 1 $\to$ The method is dissipating energy (often acceptable).

⚠️ **Too much energy loss**
  - Energy drops very fast;
  - Normalized energy quickly goes toward 0 $\to$ The method is too dissipative (time step may be too large).

❌ **Something is wrong**
  - Energy increases;
  - Normalized energy goes above 1 $\to$ The simulation is unstable or incorrect.
  
---

In summary, we want energy to stay constant or decrease slowly. If it increases, something is definitely wrong.
"""