def convergence_explanation() -> str:
    return r"""
### What do these plots mean?

This section evaluates the **convergence properties** of the numerical method.

The goal is to verify that the error decreases in a predictable way when the
discretization is refined (smaller mesh size `h` or time step `dt`).

---

#### Convergence plot (log-log)

The log-log plot shows how different error norms scale with the refinement
parameter.

- A straight line indicates power-law convergence.
- The slope of the line corresponds to the order of accuracy.
- Steeper lines mean faster error reduction.

This plot provides a **global view** of convergence behavior.

---

#### Fitted slopes (observed order)

This table reports the slope obtained by fitting the log-log curves.

- It represents the observed convergence order.
- Values close to the theoretical order indicate a correct implementation.
- Significant deviations suggest pre-asymptotic effects or modeling issues.

---

#### Per-step observed orders

This table shows the convergence order computed between consecutive refinement
levels.

- It helps identify when the asymptotic regime is reached.
- Stable values at fine resolutions are the most reliable indicators of
convergence.
"""

def convergence_explanation_for_dummies() -> str:
    return r"""
### What do these plots indicate? (For dummies)

This section checks **if our solver becomes more accurate when we refine the discretization**.

We are looking for one simple thing:

> **When `h` (mesh size) or `dt` (time step) gets smaller, the error should go down.**

---

#### 1) Convergence plot (log-log)

- Each curve shows how an error measure decreases as we refine.
- The plot uses log-log scales, so **a straight line means clean convergence**.

✅ Good sign:
- lines go **down** when `h`/`dt` decreases
- lines are roughly straight

❌ Bad sign:
- lines are flat (error not improving)
- lines go up (error gets worse)

---

#### 2) Fitted slopes (observed order)

This table estimates **how fast the error decreases**.

- A slope around **2** means “error shrinks like squared refinement” (very good).
- A slope around **1** means “linear improvement”.
- A slope around **0** means “no improvement”.

We want slopes that are:
- positive;
- stable across norms;
- close to what we expect from our method.

---

#### 3) Per-step observed orders

This table shows the order **between each pair of refinement steps**.

✅ Good sign:
- the order becomes stable in the last refinements  
  (the finest runs are the most trustworthy)

⚠️ Normal:
- coarse levels may look messy  
  (the solver is not yet in the “asymptotic” regime)

---

In general, trust the last refinement levels the most: if the last orders are stable and positive, convergence is working.
"""