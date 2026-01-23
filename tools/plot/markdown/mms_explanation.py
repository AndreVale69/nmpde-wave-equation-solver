def mms_explanation() -> str:
    return r"""
### What do these plots indicate?

This section reports **verification errors** obtained with the
**Method of Manufactured Solutions (MMS)**.

In MMS, an exact analytical solution is prescribed and the forcing term is built
so that the PDE is satisfied by that solution. The solver output is then
compared against the exact fields, and the resulting differences are measured
as error norms.

---

#### Snapshot at the last available sample

This table shows the **error norms at the final recorded time/step**.
It provides a compact “end-of-run” accuracy indicator (useful to compare runs).

---

#### Peak error over the run (max over time)

This table reports the **maximum value reached by each error norm** during the
simulation.

It is mainly used to detect:
- transient spikes;
- loss of stability;
- phases where the numerical method temporarily degrades.

---

#### MMS error history plot vs time/step

This plot shows the **time evolution of the error norms**.

Typical interpretations:
- bounded, smooth error curves $\to$ stable behavior at the chosen resolution;
- persistent growth in time $\to$ accumulating error or instability;
- oscillatory errors $\to$ under-resolution (time step too large and/or mesh too coarse).

This plot does not by itself prove correctness; the strongest validation comes
from combining MMS with a convergence study across `h` and/or `dt`.
"""

def mms_explanation_for_dummies() -> str:
    return r"""
### What do these plots indicate? (For dummies)

These plots show **how far the numerical solution is from the exact one**.

Because an exact solution is known (MMS), the error can be measured directly.

---

### Snapshot at the last sample

This table shows **the error at the end of the simulation**.

- Smaller values $\to$ more accurate solution
- Larger values $\to$ less accurate solution

It is useful to quickly compare different runs.

---

### Peak error over the run

This table shows the **largest error that occurred at any time**.

- If the peak is much larger than the final error, the solver had difficulties during the simulation.
- Very large peaks can indicate instability.

---

### Error history plot

This plot shows **how the error changes over time**.

How to read it:
- Smooth and bounded curves $\to$ solver behaves well
- Errors that grow over time $\to$ problem (instability or too large time step)
- Strong oscillations $\to$ solution is under-resolved

---

In general, errors should stay **small and under control** for the whole simulation.
"""